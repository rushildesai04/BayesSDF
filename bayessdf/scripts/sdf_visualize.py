import trimesh
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from mpl_toolkits.mplot3d import Axes3D
from skimage.measure import marching_cubes
from scipy.ndimage import zoom
import argparse
import os
import ipdb

def visualize_2d_slices(sdf, num_slices=5):
    """Visualize 2D slices along the Z-axis."""
    fig, axes = plt.subplots(1, num_slices, figsize=(15, 5))
    slice_indices = np.linspace(0, sdf.shape[2] - 1, num_slices, dtype=int)
    for i, idx in enumerate(slice_indices):
        axes[i].imshow(sdf[:, :, idx], cmap='viridis')
        axes[i].set_title(f"Slice {idx}")
        axes[i].axis('off')
    plt.show()

def visualize_3d_isosurface(sdf, isomin=-0.1, isomax=0.1):
    """Visualize the 3D isosurface of the SDF."""
    fig = go.Figure(data=go.Isosurface(
        x=np.linspace(0, sdf.shape[0], sdf.shape[0]),
        y=np.linspace(0, sdf.shape[1], sdf.shape[1]),
        z=np.linspace(0, sdf.shape[2], sdf.shape[2]),
        value=sdf.flatten(),
        isomin=isomin,
        isomax=isomax,
        surface_count=1,
        colorscale="Viridis"
    ))
    fig.update_layout(scene=dict(
        xaxis_title="X",
        yaxis_title="Y",
        zaxis_title="Z"
    ))
    fig.show()

def visualize_marching_cubes(sdf, level=0.0, out='model.ply'):
    """Generate a mesh using Marching Cubes and visualize it."""
    ipdb.set_trace()

    sdf = sdf.reshape((int(np.cbrt(sdf.size)),) * 3)
    zoom_factors = [t / o for t, o in zip(target_shape, sdf.shape)]
    sdf = zoom(sdf, zoom_factors, order=1)

    # if sdf.shape[2] == 1:
    #     z_factor = 32 / sdf.shape[2]
    #     sdf = zoom(sdf, (1, 1, z_factor), order=1)

    verts, faces, _, _ = marching_cubes(sdf, level=level)

    fig = go.Figure(data=go.Mesh3d(x=verts[:, 0], y=verts[:, 1], z=verts[:, 2], i=faces[:, 0], j=faces[:, 1], k=faces[:, 2], 
                                   color='cyan', opacity=0.5))
    fig.update_layout(scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"))
    fig.show()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2], cmap="viridis", edgecolor="none")
    plt.show()

    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    mesh.export(out)

def visualize_multiple_sdfs(sdf_files, num_slices=5):
    """Visualize 2D slices from multiple SDF files for comparison, handling both 2D and 3D SDFs."""
    fig, axes = plt.subplots(len(sdf_files), num_slices, figsize=(15, 5 * len(sdf_files)))
    for i, file_path in enumerate(sdf_files):
        sdf_data = np.load(file_path)
        print(file_path)
        sdf = sdf_data['sdf']
        if sdf.ndim == 2:
            for j in range(num_slices):
                axes[i, j].imshow(sdf, cmap='viridis')
                axes[i, j].set_title(f"{os.path.basename(file_path)} - 2D SDF")
                axes[i, j].axis('off')
        elif sdf.ndim == 3:
            slice_indices = np.linspace(0, sdf.shape[2] - 1, num_slices, dtype=int)
            for j, idx in enumerate(slice_indices):
                axes[i, j].imshow(sdf[:, :, idx], cmap='viridis')
                axes[i, j].set_title(f"{os.path.basename(file_path)} - Slice {idx}")
                axes[i, j].axis('off')
    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="SDF Visualization Tool")
    parser.add_argument("--file", type=str, required=True, help="Path to the SDF .npz file or directory of SDF files.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to the output .ply object.")
    parser.add_argument("--method", type=str, choices=["2d", "3d", "compare", "marching_cubes"], required=True, help="Visualization method.")
    parser.add_argument("--num_slices", type=int, default=5, help="Number of slices for 2D and comparison methods.")
    parser.add_argument("--isomin", type=float, default=-0.1, help="Minimum isovalue for 3D isosurface.")
    parser.add_argument("--isomax", type=float, default=0.1, help="Maximum isovalue for 3D isosurface.")
    parser.add_argument("--level", type=float, default=0.0, help="Level value for Marching Cubes isosurface extraction.")
    
    args = parser.parse_args()

    if args.method == "2d":
        sdf_data = np.load(args.file)
        sdf = sdf_data['sdf']
        visualize_2d_slices(sdf, num_slices=args.num_slices)

    elif args.method == "3d":
        sdf_data = np.load(args.file)
        sdf = sdf_data['sdf']
        visualize_3d_isosurface(sdf, isomin=args.isomin, isomax=args.isomax)

    elif args.method == "marching_cubes":
        sdf_data = np.load(args.file)
        sdf = sdf_data['sdf']
        visualize_marching_cubes(sdf, level=args.level, out=args.output_path)

    elif args.method == "compare":
        if os.path.isdir(args.file):
            sdf_files = [os.path.join(args.file, f) for f in os.listdir(args.file) if f.endswith('.npz')]
        else:
            sdf_files = args.file.split(',')

        visualize_multiple_sdfs(sdf_files, num_slices=args.num_slices)

if __name__ == "__main__":
    main()