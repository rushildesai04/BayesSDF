import numpy as np
import trimesh
from scipy.ndimage import zoom
from skimage import measure
import argparse

def load_sdf(npy_file):
    """ Load an SDF from a .npy file. """
    sdf = np.load(npy_file)
    return sdf

def sdf_to_mesh(sdf, level=0.0):
    """
    Convert an SDF into a mesh using the Marching Cubes algorithm.

    Args:
        sdf (numpy.ndarray): The signed distance field (3D grid of SDF values).
        level (float): The isosurface level to extract (default 0 for surface).

    Returns:
        trimesh.Trimesh: Mesh object extracted from SDF.
    """
    verts, faces, normals, _ = measure.marching_cubes(sdf, level=level)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
    return mesh

def save_mesh(mesh, output_file):
    """ Save the mesh as a .ply file. """
    mesh.export(output_file)
    print(f"Saved mesh to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Convert a .npy SDF file to .ply mesh")
    parser.add_argument("--npy_file", type=str, help="Path to input .npy file")
    parser.add_argument("--output_ply", type=str, help="Path to output .ply file")
    parser.add_argument("--level", type=float, default=0.0, help="Isosurface level (default: 0.0)")

    args = parser.parse_args()

    print(f"Loading SDF from {args.npy_file}...")
    sdf = load_sdf(args.npy_file)
    sdf = sdf.reshape((257, 257, 257))

    print("Generating mesh from SDF...")
    mesh = sdf_to_mesh(sdf, level=args.level)

    print(f"Saving mesh to {args.output_ply}...")
    save_mesh(mesh, args.output_ply)

if __name__ == "__main__":
    main()
