#!/usr/bin/env python

from __future__ import annotations # type:ignore


import time # type:ignore
import tyro # type:ignore
import torch # type:ignore
import numpy as np # type:ignore
import pkg_resources # type:ignore
import json # type:ignore
import types # type:ignore
import nerfstudio # type:ignore
import nerfacc # type:ignore
import ipdb # type:ignore
from dataclasses import dataclass # type:ignore
from pathlib import Path # type:ignore
from nerfstudio.utils.eval_utils import eval_setup # type:ignore
from nerfstudio.utils.rich_utils import CONSOLE # type:ignore
from nerfstudio.models.neus_facto import NeuSFactoModel # type:ignore
from nerfstudio.field_components.field_heads import FieldHeadNames # type:ignore
from nerfstudio.field_components.encodings import HashEncoding # type:ignore
from bayessdf.utils.utils import normalize_point_coords, find_grid_indices, get_gaussian_blob_new # type:ignore

@dataclass
class ComputeUncertainty:
    """Load a checkpoint, compute uncertainty, and save it to a npy file."""

    # Path to config YAML file.
    load_config: Path
    # Name of the output file.
    output_path: Path = Path("unc.npy")
    # Uncertainty level of detail (log2 of it)
    lod: int = 8
    # number of iterations on the trainset    
    iters: int = 1000         
    
    def find_uncertainty(self, points, rgb, outputs, distortion):
        inds, coeffs = find_grid_indices(points, self.aabb, distortion, self.lod, self.device)

        #because deformation params are detached for each point on each ray from the grid, summation does not affect derivative
        colors = torch.sum(rgb, dim=0)

        colors[0].backward(retain_graph=True)

        # r = outputs['offsets'].grad.clone().detach().view(-1,3)
        # r = outputs['eik_grad'].clone().detach().view(-1,3)
        r = outputs['offsets'].clone().detach().view(-1,3)
        
        # outputs['offsets'].grad.zero_()
        # outputs['eik_grad'].zero_()
        outputs['offsets'].zero_()

        colors[1].backward(retain_graph=True)

        # g = outputs['offsets'].grad.clone().detach().view(-1,3)
        # g = outputs['eik_grad'].clone().detach().view(-1,3)
        g = outputs['offsets'].clone().detach().view(-1,3)

        # outputs['offsets'].grad.zero_()
        # outputs['eik_grad'].zero_()
        outputs['offsets'].zero_()

        colors[2].backward()

        # b = outputs['offsets'].grad.clone().detach().view(-1,3)
        # b = outputs['eik_grad'].clone().detach().view(-1,3)
        b = outputs['offsets'].clone().detach().view(-1,3)

        # outputs['offsets'].grad.zero_()
        # outputs['eik_grad'].zero_()
        outputs['offsets'].zero_()
        
        dmy = (torch.arange(points.shape[0])[...,None]).repeat((1,points.shape[1])).flatten().to(self.device)
        first = True

        for corner in range(8):
            if first:
                all_ind = torch.cat((dmy.unsqueeze(-1),inds[corner].unsqueeze(-1)), dim=-1) 
                all_r = coeffs[corner].unsqueeze(-1)*r
                all_g = coeffs[corner].unsqueeze(-1)*g
                all_b = coeffs[corner].unsqueeze(-1)*b
                first = False
            else:
                all_ind = torch.cat((all_ind, torch.cat((dmy.unsqueeze(-1),inds[corner].unsqueeze(-1)), dim=-1)), dim=0)
                all_r = torch.cat((all_r, coeffs[corner].unsqueeze(-1)*r), dim=0)
                all_g = torch.cat((all_g, coeffs[corner].unsqueeze(-1)*g), dim=0)
                all_b = torch.cat((all_b, coeffs[corner].unsqueeze(-1)*b ), dim=0)

        keys_all, inds_all = torch.unique(all_ind, dim=0, return_inverse=True)
        grad_r_1 = torch.bincount(inds_all, weights=all_r[...,0]) #for first element of deformation field
        grad_g_1 = torch.bincount(inds_all, weights=all_g[...,0])
        grad_b_1 = torch.bincount(inds_all, weights=all_b[...,0])
        grad_r_2 = torch.bincount(inds_all, weights=all_r[...,1]) #for second element of deformation field
        grad_g_2 = torch.bincount(inds_all, weights=all_g[...,1])
        grad_b_2 = torch.bincount(inds_all, weights=all_b[...,1])
        grad_r_3 = torch.bincount(inds_all, weights=all_r[...,2]) #for third element of deformation field
        grad_g_3 = torch.bincount(inds_all, weights=all_g[...,2])
        grad_b_3 = torch.bincount(inds_all, weights=all_b[...,2])
        grad_1 = grad_r_1**2+grad_g_1**2+grad_b_1**2
        grad_2 = grad_r_2**2+grad_g_2**2+grad_b_2**2
        grad_3 = grad_r_3**2+grad_g_3**2+grad_b_3**2 #will consider the trace of each submatrix for each deformation
        
        #vector as indicator of hessian wrt the whole vector
        grads_all = torch.cat((keys_all[:,1].unsqueeze(-1), (grad_1+grad_2+grad_3).unsqueeze(-1)), dim=-1)
        hessian = torch.zeros(((2**self.lod)+1)**3).to(self.device)
        hessian = hessian.put((grads_all[:,0]).long(), grads_all[:,1], True)

        return hessian
    
    
    def get_unc_neusfacto(self, ray_bundle, model):
        ''' reimplementation of get_output function from models because of lack of proper interface to ray_samples'''

        if model.collider is not None:
            ray_bundle = model.collider(ray_bundle)

        ray_samples, weights_list, ray_samples_list = model.proposal_sampler(ray_bundle, density_fns=model.density_fns)

        points = ray_samples.frustums.get_positions()
        pos, _ = normalize_point_coords(points, self.aabb, model.field.spatial_distortion)

        #find offset value in the deformation field
        offsets = self.deform_field(pos).clone().detach()
        # offsets = self.deform_field(pos)
        # offsets.requires_grad = True
        # offsets.retain_grad()

        ray_samples.frustums.set_offsets(offsets)    
        field_outputs = model.field(ray_samples, return_alphas=True)

        if model.config.background_model != "none":
            field_outputs = model.forward_background_field_and_merge(ray_samples, field_outputs)

        weights = ray_samples.get_weights_and_transmittance_from_alphas(field_outputs[FieldHeadNames.ALPHA])[0]

        weights_list.append(weights)
        ray_samples_list.append(ray_samples)

        samples_and_field_outputs = {
            "ray_samples": ray_samples,
            "field_outputs": field_outputs,
            "weights": weights,
            "weights_list": weights_list,
            "ray_samples_list": ray_samples_list,
        }

        # Shortcuts
        field_outputs = samples_and_field_outputs["field_outputs"]
        ray_samples = samples_and_field_outputs["ray_samples"]
        weights = samples_and_field_outputs["weights"]

        rgb = model.renderer_rgb(rgb=field_outputs[FieldHeadNames.RGB], weights=weights)
        depth = model.renderer_depth(weights=weights, ray_samples=ray_samples)

        # the rendered depth is point-to-point distance and we should convert to depth
        # depth = depth / ray_bundle.directions_norm
        depth = depth / ray_bundle.metadata["directions_norm"] # NERFSTUDIO

        # remove the rays that don't intersect with the surface
        # hit = (field_outputs[FieldHeadNames.SDF] > 0.0).any(dim=1) & (field_outputs[FieldHeadNames.SDF] < 0).any(dim=1)
        # depth[~hit] = 10000.0

        # normal = model.renderer_normal(semantics=field_outputs[FieldHeadNames.NORMAL], weights=weights)
        normal = model.renderer_normal(semantics=field_outputs[FieldHeadNames.NORMALS], weights=weights) # NERFSTUDIO
        accumulation = model.renderer_accumulation(weights=weights)

        # TODO add a flat to control how the background model are combined with foreground sdf field

        # background model
        if model.config.background_model != "none" and "bg_transmittance" in samples_and_field_outputs:
            bg_transmittance = samples_and_field_outputs["bg_transmittance"]

            # sample inversely from far to 1000 and points and forward the bg model
            ray_bundle.nears = ray_bundle.fars
            ray_bundle.fars = torch.ones_like(ray_bundle.fars) * model.config.far_plane_bg
            ray_samples_bg = model.sampler_bg(ray_bundle)

            # use the same background model for both density field and occupancy field
            field_outputs_bg = model.field_background(ray_samples_bg)
            weights_bg = ray_samples_bg.get_weights(field_outputs_bg[FieldHeadNames.DENSITY])

            rgb_bg = model.renderer_rgb(rgb=field_outputs_bg[FieldHeadNames.RGB], weights=weights_bg)

            # merge background color to forgound color
            rgb = rgb + bg_transmittance * rgb_bg

        outputs = {
            "rgb": rgb,
            "accumulation": accumulation,
            "depth": depth,
            "normal": normal,
            "weights": weights,
            "ray_points": model.scene_contraction(ray_samples.frustums.get_start_positions()), # used for creating visiblity mask
            # "directions_norm": ray_bundle.directions_norm,  # used to scale z_vals for free space and sdf loss
            "directions_norm": ray_bundle.metadata["directions_norm"], # NERFSTUDIO
        }

        if True:
            grad_points = field_outputs[FieldHeadNames.GRADIENT]

            points_norm = pos
            outputs.update({"eik_grad": grad_points, "points_norm": points_norm, "offsets": offsets})

            # TODO volsdf use different point set for eikonal loss
            # grad_points = self.field.gradient(eik_points)
            # outputs.update({"eik_grad": grad_points})

            outputs.update(samples_and_field_outputs)

        # TODO how can we move it to neus_facto without out of memory
        if "weights_list" in samples_and_field_outputs:
            weights_list = samples_and_field_outputs["weights_list"]
            ray_samples_list = samples_and_field_outputs["ray_samples_list"]

            for i in range(len(weights_list) - 1):
                outputs[f"prop_depth_{i}"] = model.renderer_depth(
                    weights=weights_list[i], ray_samples=ray_samples_list[i]
                )
        # this is used only in viewer
        outputs["normal_vis"] = (outputs["normal"] + 1.0) / 2.0
        
        return outputs, points, offsets
    
    def get_sdf(self, model, pipeline):
        x = np.linspace(-1.0, 1.0, 128)
        y = np.linspace(-1.0, 1.0, 128)
        z = np.linspace(-1.0, 1.0, 128)
        xv, yv, zv = np.meshgrid(x, y, z)
        query_points = np.stack([xv, yv, zv], axis=-1).reshape(-1, 3)
        query_points_tensor = torch.tensor(query_points, dtype=torch.float32).to(self.device)
        # with torch.no_grad():
        #     sdf_outputs = model.field(query_points_tensor)
        #     sdf_values = sdf_outputs.cpu().numpy()
        # sdf_grid = sdf_values.reshape(xv.shape)
        # np.savez("/pscratch/sd/r/rushil/model_full.npz", sdf=sdf_grid)
        # print(f"Saved First SDF")
        sdf_list = []
        for step in range(pipeline.datamanager.train_dataset.__len__()):
            ray_bundle, _ = pipeline.datamanager.next_train(step)
            if model.collider is not None:
                ray_bundle = model.collider(ray_bundle)
            ray_samples, _, _ = model.proposal_sampler(ray_bundle, density_fns=model.density_fns)
            field_outputs = model.field(ray_samples, return_alphas=True)
            sdf_list.append(field_outputs[FieldHeadNames.SDF])
        sdf_arrays = [sdf.cpu().numpy() for sdf in sdf_list]
        stacked_sdf = np.stack(sdf_arrays, axis=0)
        # np.savez("/pscratch/sd/r/rushil/model_ray.npz", sdfs=stacked_sdf)
        print(f"Saved Second SDF")
        
    def get_output_fn(self, model):
        if isinstance(model, NeuSFactoModel):
            return self.get_unc_neusfacto
        else:
            raise Exception("Sorry, this model is not currently supported.")
            
    def main(self) -> None:
        """Main function."""
        
        if pkg_resources.get_distribution("nerfstudio").version >= "0.3.1":
            config, pipeline, checkpoint_path, _ = eval_setup(self.load_config)
        else:
            config, pipeline, checkpoint_path = eval_setup(self.load_config)
        
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        start_time = time.time()
        
        self.device = pipeline.device
        self.aabb = pipeline.model.scene_box.aabb.to(self.device)
        self.hessian = torch.zeros(((2**self.lod)+1)**3).to(self.device)
        self.deform_field = HashEncoding(num_levels = 1, 
                            min_res = 2**self.lod,
                            max_res = 2**self.lod,
                            log2_hashmap_size = self.lod*3+1, #simple regular grid (hash table size > grid size)
                            features_per_level = 3,
                            # hash_init_scale = 0.,
                            hash_init_scale = 1.,
                            # hash_init_scale = 50.,
                            implementation = "torch",
                            interpolation = "Linear")
        self.deform_field.to(self.device)
        self.deform_field.scalings = torch.tensor([2**self.lod]).to(self.device)
        
        pipeline.eval()
        len_train = max(pipeline.datamanager.train_dataset.__len__(), self.iters)
        # self.get_sdf(pipeline.model, pipeline)
        for step in range(len_train):
            print(f'Step: [{step}/{len_train}]')
            ray_bundle, batch = pipeline.datamanager.next_train(step)
            output_fn = self.get_output_fn(pipeline.model)
            if True: #not isinstance(pipeline.model, MipNerfModel):
                outputs, points, offsets = output_fn(ray_bundle, pipeline.model)
                hessian = self.find_uncertainty(points, outputs['rgb'], outputs, pipeline.model.field.spatial_distortion)    
                self.hessian += hessian.clone().detach()
            else:
                outputs, points_fine, offsets_fine, points_coarse, offsets_coarse = output_fn(ray_bundle, pipeline.model)
                hessian = self.find_uncertainty(points_fine, offsets_fine, outputs['rgb_fine'], pipeline.model.field.spatial_distortion)    
                self.hessian += hessian.clone().detach()
                hessian = self.find_uncertainty(points_coarse, offsets_coarse, outputs['rgb_coarse'], pipeline.model.field.spatial_distortion)    
                self.hessian += hessian.clone().detach()
        end_time = time.time()    
        print("Done")
        with open(str(self.output_path), 'wb') as f:
            np.save(f, self.hessian.cpu().numpy())
        execution_time = end_time - start_time
        print(f"Execution time: {execution_time:.6f} seconds")    


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(ComputeUncertainty).main()


if __name__ == "__main__":
    entrypoint()

# For sphinx docs
get_parser_fn = lambda: tyro.extras.get_parser(ComputeUncertainty)  # noqa
