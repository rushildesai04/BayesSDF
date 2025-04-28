#!/usr/bin/env python

"""
render_uncertainty.py
"""

from __future__ import annotations
import json
import os
import struct
import shutil
import sys
import types
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union
import mediapy as media
import numpy as np
import torch
import tyro
import nerfstudio
import pkg_resources
from jaxtyping import Float
from rich import box, style
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from torch import Tensor
from typing_extensions import Annotated
import pickle

from nerfstudio.cameras.camera_paths import (get_interpolated_camera_path, get_path_from_json, get_spiral_path) # type:ignore
from nerfstudio.cameras.cameras import Cameras, CameraType # type:ignore
# from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManager # type:ignore
from nerfstudio.data.scene_box import SceneBox # type:ignore
from nerfstudio.model_components import renderers # type:ignore
# from nerfstudio.pipelines.base_pipeline import Pipeline # type:ignore
from nerfstudio.utils import colormaps, install_checks, colors # type:ignore
from nerfstudio.utils.eval_utils import eval_setup # type:ignore
from nerfstudio.utils.rich_utils import CONSOLE, ItersPerSecColumn # type:ignore
from nerfstudio.utils.scripts import run_command # type:ignore
from bayessdf.scripts.output_uncertainty import get_output_fn, get_uncertainty

if pkg_resources.get_distribution("nerfstudio").version >= "0.3.1":
    from nerfstudio.utils.colormaps import ColormapOptions # type:ignore

import ipdb

# Same script as render.py but first adding uncertainty output to model.get_output output dict and then rendering

def _render_trajectory_video(
    pipeline,
    # pipeline: Pipeline,
    cameras: Cameras,
    output_filename: Path,
    rendered_output_names: List[str],
    crop_data: Optional[CropData] = None,
    rendered_resolution_scaling_factor: float = 1.0,
    seconds: float = 5.0,
    output_format: Literal["images", "video"] = "video",
    image_format: Literal["jpeg", "png"] = "png",
    jpeg_quality: int = 100,
    depth_near_plane: Optional[float] = None,
    depth_far_plane: Optional[float] = None,
    gt_visibility_path = None,
    colormap_options = None
) -> None:
    
    """
    Helper function to create a video of the spiral trajectory.
    Args:
        pipeline: Pipeline to evaluate with.
        cameras: Cameras to render.
        output_filename: Name of the output file.
        rendered_output_names: List of outputs to visualise.
        crop_data: Crop data to apply to the rendered images.
        rendered_resolution_scaling_factor: Scaling factor to apply to the camera image resolution.
        seconds: Length of output video.
        output_format: How to save output data.
        colormap_options: Options for colormap.
    """

    CONSOLE.print("[bold green]Creating trajectory " + output_format)
    cameras.rescale_output_resolution(rendered_resolution_scaling_factor)
    cameras = cameras.to(pipeline.device)
    fps = len(cameras) / seconds

    uncertainty_images = []
    rgb_images = []

    progress = Progress(TextColumn(":movie_camera: Rendering :movie_camera:"), 
                        BarColumn(), 
                        TaskProgressColumn(text_format="[progress.percentage]{task.completed}/{task.total:>.0f}({task.percentage:>3.1f}%)", show_speed=True), 
                        ItersPerSecColumn(suffix="fps"), 
                        TimeRemainingColumn(compact=True, elapsed_when_finished=True), TimeElapsedColumn())
    
    output_image_dir = output_filename.parent / output_filename.stem

    if output_format == "images":
        output_image_dir.mkdir(parents=True, exist_ok=True)

    if output_format == "video":
        output_filename.parent.mkdir(parents=True, exist_ok=True)
        
    with ExitStack() as stack:
        writer = None

        with progress:
            for camera_idx in progress.track(range(cameras.size), description=""):
                if gt_visibility_path:
                    pseudo_gt_visibility = media.read_image(str(gt_visibility_path)+"/{:05d}.png".format(camera_idx))
                    pseudo_gt_visibility = (pseudo_gt_visibility[..., 0] >= 1).astype('float')[...,None]
                
                else:
                    pseudo_gt_visibility = 1

                aabb_box = None

                if crop_data is not None:
                    bounding_box_min = crop_data.center - crop_data.scale / 2.0
                    bounding_box_max = crop_data.center + crop_data.scale / 2.0
                    aabb_box = SceneBox(torch.stack([bounding_box_min, bounding_box_max]).to(pipeline.device))

                # camera_ray_bundle = cameras.generate_rays(camera_indices=camera_idx, aabb_box=aabb_box)
                camera_ray_bundle = cameras.generate_rays(camera_indices=camera_idx)

                if crop_data is not None:
                    with renderers.background_color_override_context(crop_data.background_color.to(pipeline.device)), torch.no_grad():
                        outputs = pipeline.model.get_outputs_for_camera_ray_bundle(camera_ray_bundle)
                
                else:
                    with torch.no_grad():
                        outputs = pipeline.model.get_outputs_for_camera_ray_bundle(camera_ray_bundle)

                render_image = []

                output_image = outputs['rgb'].cpu().numpy()
                rgb_images.append(output_image)

                for rendered_output_name in rendered_output_names:
                    if rendered_output_name not in outputs:
                        CONSOLE.rule("Error", style="red")
                        CONSOLE.print(f"Could not find {rendered_output_name} in the model outputs", justify="center")
                        CONSOLE.print(f"Please set --rendered_output_name to one of: {outputs.keys()}", justify="center")
                        sys.exit(1)

                    output_image = outputs[rendered_output_name]

                    if rendered_output_name == 'uncertainty':
                        if pkg_resources.get_distribution("nerfstudio").version >= "0.3.1":
                            colormap_options1 = ColormapOptions(colormap='inferno', 
                                                                normalize=colormap_options.normalize, 
                                                                colormap_min=colormap_options.colormap_min, 
                                                                colormap_max=colormap_options.colormap_max, 
                                                                invert=colormap_options.invert)
                            output_image = colormaps.apply_colormap(image=output_image, colormap_options=colormap_options1).cpu().numpy() * pseudo_gt_visibility
                        
                        else:
                            output_image = colormaps.apply_colormap(output_image, cmap="inferno").cpu().numpy() * pseudo_gt_visibility

                        uncertainty_images.append(output_image)
                    
                    elif rendered_output_name == 'rgb':
                        output_image = (output_image).cpu().numpy() * pseudo_gt_visibility
                    
                    else:
                        output_image = colormaps.apply_depth_colormap(output_image, near_plane=depth_near_plane, far_plane=depth_far_plane).cpu().numpy() * pseudo_gt_visibility

                    render_image.append(output_image)

                render_image = np.concatenate(render_image, axis=1)

                if output_format == "images":
                    if image_format == "png":
                        media.write_image(output_image_dir / f"{camera_idx:05d}.png", render_image, fmt="png")

                    if image_format == "jpeg":
                        media.write_image(output_image_dir / f"{camera_idx:05d}.jpg", render_image, fmt="jpeg", quality=jpeg_quality)

                if output_format == "video":
                    if writer is None:
                        render_width = int(render_image.shape[1])
                        render_height = int(render_image.shape[0])
                        writer = stack.enter_context(media.VideoWriter(path=output_filename, shape=(render_height, render_width), fps=fps))
                    
                    writer.add_image(render_image)

    rgb_images_np = np.stack(rgb_images, axis=0)
    np.save(output_filename.with_name("rgb.npy"), rgb_images_np)
    print(f"RGB Image Saved With Shape {rgb_images_np.shape}")

    uncertainty_images_np = np.stack(uncertainty_images, axis=0)
    np.save(output_filename.with_suffix('.npy'), uncertainty_images_np)
    print(f"Uncertainty Image Saved With Shape {uncertainty_images_np.shape}")
    
    table = Table(title=None, show_header=False, box=box.MINIMAL, title_style=style.Style(bold=True))

    if output_format == "video":
        if cameras.camera_type[0] == CameraType.EQUIRECTANGULAR.value:
            CONSOLE.print("Adding spherical camera data")
            insert_spherical_metadata_into_file(output_filename)
        table.add_row("Video", str(output_filename))

    else:
        table.add_row("Images", str(output_image_dir))

    CONSOLE.print(Panel(table, title="[bold][green]:tada: Render Complete :tada:[/bold]", expand=False))

def insert_spherical_metadata_into_file(
    output_filename: Path,
) -> None:
    
    """
    Inserts spherical metadata into MP4 video file in-place.
    Args:
        output_filename: Name of the (input and) output file.
    """
    
    spherical_uuid = b"\xff\xcc\x82\x63\xf8\x55\x4a\x93\x88\x14\x58\x7a\x02\x52\x1f\xdd"
    spherical_metadata = bytes(
        """<rdf:SphericalVideo
        xmlns:rdf='http://www.w3.org/1999/02/22-rdf-syntax-ns#'
        xmlns:GSpherical='http://ns.google.com/videos/1.0/spherical/'>
        <GSpherical:ProjectionType>equirectangular</GSpherical:ProjectionType>
        <GSpherical:Spherical>True</GSpherical:Spherical>
        <GSpherical:Stitched>True</GSpherical:Stitched>
        <GSpherical:StitchingSoftware>nerfstudio</GSpherical:StitchingSoftware>
        </rdf:SphericalVideo>""",
        "utf-8",
    )

    insert_size = len(spherical_metadata) + 8 + 16

    with open(output_filename, mode="r+b") as mp4file:
        try:
            mp4file_size = os.stat(output_filename).st_size

            while True:
                pos = mp4file.tell()
                size, tag = struct.unpack(">I4s", mp4file.read(8))

                if tag == b"moov":
                    break

                mp4file.seek(pos + size)

            if pos + size != mp4file_size:
                raise Exception("moov container not at end of file")
            
            mp4file.seek(pos)
            mp4file.write(struct.pack(">I", size + insert_size))
            mp4file.seek(pos + 8)

            while True:
                pos = mp4file.tell()
                size, tag = struct.unpack(">I4s", mp4file.read(8))

                if tag == b"trak":
                    break

                mp4file.seek(pos + size)

            mp4file.seek(pos)
            mp4file.write(struct.pack(">I", size + insert_size))
            mp4file.seek(pos + size)

            rest_of_file = mp4file.read(mp4file_size - pos - size)

            mp4file.seek(pos + size)
            mp4file.write(struct.pack(">I4s16s", insert_size, b"uuid", spherical_uuid))
            mp4file.write(spherical_metadata)
            mp4file.write(rest_of_file)

        finally:
            mp4file.close()

@dataclass
class CropData:

    """Data for cropping an image."""
    background_color: Float[Tensor, "3"] = torch.Tensor([0.0, 0.0, 0.0])
    """background color""" 
    center: Float[Tensor, "3"] = torch.Tensor([0.0, 0.0, 0.0])
    """center of the crop"""
    scale: Float[Tensor, "3"] = torch.Tensor([2.0, 2.0, 2.0])
    """scale of the crop"""
    
def get_crop_from_json(camera_json: Dict[str, Any]) -> Optional[CropData]:
    """
    Load crop data from a camera path JSON
    Args:
        camera_json: camera path data

    Returns:
        Crop data
    """

    if "crop" not in camera_json or camera_json["crop"] is None:
        return None

    bg_color = camera_json["crop"]["crop_bg_color"]

    return CropData(
        background_color=torch.Tensor([bg_color["r"] / 255.0, bg_color["g"] / 255.0, bg_color["b"] / 255.0]),
        center=torch.Tensor(camera_json["crop"]["crop_center"]),
        scale=torch.Tensor(camera_json["crop"]["crop_scale"]),
    )

@dataclass
class BaseRender:

    """Base class for rendering."""
    load_config: Path
    """Path to config YAML file."""    
    output_path: Path = Path("renders/output.mp4")
    """Path to output video file."""
    unc_path: Path = Path("unc.npy")
    """Path to output video file."""
    visibility_path: Path = None
    """Path to nerfbuster visibility masks, if needed to be rendered."""
    image_format: Literal["jpeg", "png"] = "png"
    """Image format"""
    jpeg_quality: int = 100
    """JPEG quality"""
    downscale_factor: float = 1.0
    """Scaling factor to apply to the camera image resolution."""
    rendered_output_names: List[str] = field(default_factory=lambda: ["rgb", "uncertainty", "depth"])
    """Name of the renderer outputs to use. rgb, depth, etc. concatenates them along y axis"""
    eval_num_rays_per_chunk: Optional[int] = None
    """Specifies number of rays per chunk during eval. If None, use the value in the config file."""
    if pkg_resources.get_distribution("nerfstudio").version >= "0.3.1":
        colormap_options: colormaps.ColormapOptions = colormaps.ColormapOptions()
    else:
        colormap_options = None
    """Colormap options."""
    depth_near_plane: Optional[float] = None
    """Closest depth to consider when using the colormap for depth. If None, use min value."""
    depth_far_plane: Optional[float] = None
    """Furthest depth to consider when using the colormap for depth. If None, use max value."""
    filter_out: bool = False
    """Render with filtering"""
    filter_thresh: float = 0.5
    """filter threshold"""
    white_bg: bool = False
    """ Render empty space as white when filtering"""
    black_bg: bool = True
    """ Render empty space as black when filtering"""


@dataclass
class RenderCameraPath(BaseRender):
    
    """Render a camera path generated by the viewer or blender add-on."""
    camera_path_filename: Path = Path("camera_path.json")
    """Filename of the camera path to render."""
    output_format: Literal["images", "video"] = "video"
    """How to save output data."""
    
    def main(self) -> None:

        """Main function."""

        if pkg_resources.get_distribution("nerfstudio").version >= "0.3.1":
            _, pipeline, _, _ = eval_setup(
                self.load_config,
                eval_num_rays_per_chunk=self.eval_num_rays_per_chunk,
                test_mode="inference",
            )

        else:
            _, pipeline, _ = eval_setup(
                self.load_config,
                eval_num_rays_per_chunk=self.eval_num_rays_per_chunk,
                test_mode="inference",
            )
        
        self.device = pipeline.device
        pipeline.model.filter_out = self.filter_out
        pipeline.model.filter_thresh = self.filter_thresh
        pipeline.model.hessian = torch.tensor(np.load(str(self.unc_path))).to(self.device)
        pipeline.model.lod = np.log2(round(pipeline.model.hessian.shape[0]**(1/3))-1)
        pipeline.model.get_uncertainty = types.MethodType(get_uncertainty, pipeline.model)
        pipeline.model.white_bg = self.white_bg
        pipeline.model.black_bg = self.black_bg
        pipeline.model.N =  4096*1000
        new_method = get_output_fn(pipeline.model)
        pipeline.model.get_outputs = types.MethodType(new_method, pipeline.model)

        install_checks.check_ffmpeg_installed()

        with open(self.camera_path_filename, "r", encoding="utf-8") as f:
            camera_path = json.load(f)
        
        seconds = camera_path["seconds"]
        crop_data = get_crop_from_json(camera_path)
        camera_path = get_path_from_json(camera_path)

        if camera_path.camera_type[0] == CameraType.OMNIDIRECTIONALSTEREO_L.value:
            temp_folder_path = self.output_path.parent / (self.output_path.stem + "_temp")

            Path(temp_folder_path).mkdir(parents=True, exist_ok=True)
            left_eye_path = temp_folder_path / "ods_render_Left.mp4"

            self.output_path = left_eye_path

            CONSOLE.print("[bold green]:goggles: Omni-directional Stereo VR :goggles:")
            CONSOLE.print("Rendering left eye view")

        if self.output_format == "video" and str(self.output_path.suffix) == "":
            self.output_path = self.output_path.with_suffix(".mp4")

        _render_trajectory_video(
            pipeline,
            camera_path,
            output_filename=self.output_path,
            rendered_output_names=self.rendered_output_names,
            rendered_resolution_scaling_factor=1.0 / self.downscale_factor,
            crop_data=crop_data,
            seconds=seconds,
            output_format=self.output_format,
            image_format=self.image_format,
            jpeg_quality=self.jpeg_quality,
            depth_near_plane=self.depth_near_plane,
            depth_far_plane=self.depth_far_plane,
            colormap_options=self.colormap_options
        )

        if camera_path.camera_type[0] == CameraType.OMNIDIRECTIONALSTEREO_L.value:
            left_eye_path = self.output_path
            right_eye_path = left_eye_path.parent / "ods_render_Right.mp4"

            self.output_path = right_eye_path
            camera_path.camera_type[0] = CameraType.OMNIDIRECTIONALSTEREO_R.value

            CONSOLE.print("Rendering right eye view")

            _render_trajectory_video(
                pipeline,
                camera_path,
                output_filename=self.output_path,
                rendered_output_names=self.rendered_output_names,
                rendered_resolution_scaling_factor=1.0 / self.downscale_factor,
                crop_data=crop_data,
                seconds=seconds,
                output_format=self.output_format,
                image_format=self.image_format,
                jpeg_quality=self.jpeg_quality,
                depth_near_plane=self.depth_near_plane,
                depth_far_plane=self.depth_far_plane,
                gt_visibility_path=self.visibility_path,
                colormap_options=self.colormap_options
            )

            self.output_path = Path(str(left_eye_path.parent)[:-5] + ".mp4")
            ffmpeg_ods_command = ""

            if self.output_format == "video":
                ffmpeg_ods_command = f'ffmpeg -y -i "{left_eye_path}" -i "{right_eye_path}" -filter_complex "[0:v]pad=iw:2*ih[int];[int][1:v]overlay=0:h" -c:v libx264 -crf 23 -preset veryfast "{self.output_path}"'
                run_command(ffmpeg_ods_command, verbose=False)
            
            if self.output_format == "images":
                self.output_path = Path(str(left_eye_path.parent)[:-5])
                self.output_path.mkdir(parents=True, exist_ok=True)

                if self.image_format == "png":
                    ffmpeg_ods_command = f'ffmpeg -y -pattern_type glob -i "{str(left_eye_path.with_suffix("") / "*.png")}"  -pattern_type glob -i "{str(right_eye_path.with_suffix("") / "*.png")}" -filter_complex vstack -start_number 0 "{str(self.output_path)+"//%05d.png"}"'
                
                elif self.image_format == "jpeg":
                    ffmpeg_ods_command = f'ffmpeg -y -pattern_type glob -i "{str(left_eye_path.with_suffix("") / "*.jpg")}"  -pattern_type glob -i "{str(right_eye_path.with_suffix("") / "*.jpg")}" -filter_complex vstack -start_number 0 "{str(self.output_path)+"//%05d.jpg"}"'
                
                run_command(ffmpeg_ods_command, verbose=False)

            if str(left_eye_path.parent)[-5:] == "_temp":
                shutil.rmtree(left_eye_path.parent, ignore_errors=True)
            
            CONSOLE.print("[bold green]Final ODS Render Complete")


@dataclass
class RenderInterpolated(BaseRender):
    
    """Render a trajectory that interpolates between training or eval dataset images."""
    pose_source: Literal["eval", "train"] = "eval"
    """Pose source to render."""
    interpolation_steps: int = 10
    """Number of interpolation steps between eval dataset cameras."""
    order_poses: bool = False
    """Whether to order camera poses by proximity."""
    frame_rate: int = 30
    """Frame rate of the output video."""
    output_format: Literal["images", "video"] = "video"
    """How to save output data."""
    
    def main(self) -> None:

        """Main function."""

        if pkg_resources.get_distribution("nerfstudio").version >= "0.3.1":
            _, pipeline, _, _ = eval_setup(
                self.load_config,
                eval_num_rays_per_chunk=self.eval_num_rays_per_chunk,
                test_mode="test",
            )
        else:
             _, pipeline, _ = eval_setup(
                self.load_config,
                eval_num_rays_per_chunk=self.eval_num_rays_per_chunk,
                test_mode="test",
            )

        self.device = pipeline.device
        pipeline.model.filter_out = self.filter_out
        pipeline.model.filter_thresh = self.filter_thresh
        pipeline.model.hessian = torch.tensor(np.load(str(self.unc_path))).to(self.device)
        pipeline.model.lod = np.log2(round(pipeline.model.hessian.shape[0]**(1/3))-1)
        pipeline.model.get_uncertainty = types.MethodType(get_uncertainty, pipeline.model)
        pipeline.model.white_bg = self.white_bg
        pipeline.model.black_bg = self.black_bg
        pipeline.model.N =  4096*1000

        new_method = get_output_fn(pipeline.model)
        pipeline.model.get_outputs = types.MethodType(new_method, pipeline.model)


        install_checks.check_ffmpeg_installed()

        if self.pose_source == "eval":
            assert pipeline.datamanager.eval_dataset is not None
            cameras = pipeline.datamanager.eval_dataset.cameras
        
        else:
            assert pipeline.datamanager.train_dataset is not None
            cameras = pipeline.datamanager.train_dataset.cameras

        seconds = self.interpolation_steps * len(cameras) / self.frame_rate

        if pkg_resources.get_distribution("nerfstudio").version >= "0.3.1":
            camera_path = get_interpolated_camera_path(
                cameras=cameras,
                steps=self.interpolation_steps,
                order_poses=self.order_poses,
            )

        else:
             camera_path = get_interpolated_camera_path(cameras=cameras, steps=self.interpolation_steps)
             
        Ks = cameras.get_intrinsics_matrices()

        if self.interpolation_steps == 1:
            camera_path = pipeline.datamanager.fixed_indices_eval_dataloader.cameras


        _render_trajectory_video(
            pipeline,
            camera_path,
            output_filename=self.output_path,
            rendered_output_names=self.rendered_output_names,
            rendered_resolution_scaling_factor=1.0 / self.downscale_factor,
            seconds=seconds,
            output_format=self.output_format,
            depth_near_plane=self.depth_near_plane,
            depth_far_plane=self.depth_far_plane,
            gt_visibility_path=self.visibility_path,
            colormap_options=self.colormap_options
        )

@dataclass
class RenderSpiral(BaseRender):

    """Render a spiral trajectory (often not great)."""
    seconds: float = 3.0
    """How long the video should be."""
    output_format: Literal["images", "video"] = "video"
    """How to save output data."""
    frame_rate: int = 30
    """Frame rate of the output video (only for interpolate trajectory)."""
    radius: float = 0.1
    """Radius of the spiral."""

    def main(self) -> None:

        """Main function."""

        if pkg_resources.get_distribution("nerfstudio").version >= "0.3.1":
            _, pipeline, _, _ = eval_setup(
                self.load_config,
                eval_num_rays_per_chunk=self.eval_num_rays_per_chunk,
                test_mode="test",
            )
        
        else:
             _, pipeline, _ = eval_setup(
                self.load_config,
                eval_num_rays_per_chunk=self.eval_num_rays_per_chunk,
                test_mode="test",
            )

        self.device = pipeline.device

        pipeline.model.filter_out = self.filter_out
        pipeline.model.filter_thresh = self.filter_thresh
        pipeline.model.hessian = torch.tensor(np.load(str(self.unc_path))).to(self.device)
        pipeline.model.lod = np.log2(round(pipeline.model.hessian.shape[0]**(1/3))-1)
        pipeline.model.get_uncertainty = types.MethodType(get_uncertainty, pipeline.model)
        pipeline.model.white_bg = self.white_bg
        pipeline.model.black_bg = self.black_bg
        pipeline.model.N = 4096 * 1000

        new_method = get_output_fn(pipeline.model)
        pipeline.model.get_outputs = types.MethodType(new_method, pipeline.model)

        install_checks.check_ffmpeg_installed()

        # assert isinstance(pipeline.datamanager, VanillaDataManager)
        steps = int(self.frame_rate * self.seconds)

        camera_start = pipeline.datamanager.eval_dataloader.get_camera(image_idx=0).flatten()
        # camera_start = pipeline.datamanager.eval_dataloader.get_camera(image_idx=0)[0]
        camera_path = get_spiral_path(camera_start, steps=steps, radius=self.radius)

        _render_trajectory_video(
            pipeline,
            camera_path,
            output_filename=self.output_path,
            rendered_output_names=self.rendered_output_names,
            rendered_resolution_scaling_factor=1.0 / self.downscale_factor,
            seconds=self.seconds,
            output_format=self.output_format,
            depth_near_plane=self.depth_near_plane,
            depth_far_plane=self.depth_far_plane,
            gt_visibility_path=self.visibility_path,
            colormap_options=self.colormap_options
        )


Commands = tyro.conf.FlagConversionOff[
    Union[
        Annotated[RenderCameraPath, tyro.conf.subcommand(name="camera-path")],
        Annotated[RenderInterpolated, tyro.conf.subcommand(name="interpolate")],
        Annotated[RenderSpiral, tyro.conf.subcommand(name="spiral")],
    ]
]


def entrypoint():

    """Entrypoint for use with pyproject scripts."""

    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(Commands).main()


if __name__ == "__main__":
    entrypoint()


def get_parser_fn():

    """Get the parser function for the sphinx docs."""

    return tyro.extras.get_parser(Commands)
