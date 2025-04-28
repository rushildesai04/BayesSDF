import os
import argparse
import open3d as o3d
import trimesh
import skimage
import numpy as np
from scipy.spatial import cKDTree
from mesh_to_sdf import mesh_to_voxels, mesh_to_sdf, get_surface_point_cloud
import torch
import ipdb

os.environ["PYOPENGL_PLATFORM"] = "egl"


def run_vanilla(input_file, output_file, input_type, voxel_size=0.1):
    if input_type == 'mesh':
        mesh = trimesh.load(input_file)
        if not mesh.is_watertight:
            print("Warning: The input mesh is not watertight, which might affect SDF accuracy")
    elif input_type == 'pointcloud':
        pcd = o3d.io.read_point_cloud(input_file)
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
    else:
        raise ValueError("Invalid input type. Choose 'mesh' or 'pointcloud'")

    voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(mesh, voxel_size)

    voxel_centers = np.asarray([voxel.grid_index for voxel in voxel_grid.get_voxels()])
    voxel_centers = voxel_centers * voxel_grid.voxel_size + voxel_grid.origin
    vertices = np.asarray(mesh.vertices)
    trimesh_mesh = trimesh.Trimesh(vertices=vertices, faces=np.asarray(mesh.triangles))
    kdtree = cKDTree(trimesh_mesh.vertices)
    distances, _ = kdtree.query(voxel_centers)
    normals = trimesh_mesh.vertex_normals
    signs = np.sign(np.einsum('ij,ij->i', normals, voxel_centers - vertices.mean(axis=0)))
    signed_distances = signs * distances

    # voxel_centers = np.asarray([voxel.grid_index for voxel in voxel_grid.get_voxels()])
    # voxel_centers = voxel_centers * voxel_grid.voxel_size + voxel_grid.origin
    # vertices = np.asarray(mesh.vertices)
    # normals = np.asarray(mesh.vertex_normals)
    # kdtree = cKDTree(vertices)
    # distances, nearest_vertex_indices = kdtree.query(voxel_centers)
    # nearest_normals = normals[nearest_vertex_indices]
    # signs = np.sign(np.einsum('ij,ij->i', nearest_normals, voxel_centers - vertices.mean(axis=0)))
    # signed_distances = signs * distances

    np.savez(output_file, voxels=voxel_centers, sdf=signed_distances)
    print(f"SDF saved to {output_file}")

    return mesh


def run_mesh_to_sdf(input_file, output_file, voxel_resolution):
    mesh = trimesh.load(input_file)
    query_points = mesh.vertices + np.random.randn(*mesh.vertices.shape) * 0.01

    voxels = mesh_to_voxels(mesh, voxel_resolution, pad=False, surface_point_method='sample', sign_method='normal')
    queries = mesh_to_sdf(mesh, query_points, surface_point_method='sample', sign_method='normal')
    pc = get_surface_point_cloud(mesh, surface_point_method='sample')

    vertices, faces, normals, _ = skimage.measure.marching_cubes(voxels, level=0)
    new_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals)

    np.savez(output_file, sdf=voxels, queries=queries, point_cloud=pc.points, normals=pc.normals)
    new_mesh.export(output_file.replace('.npz', '.ply'))

    print(f"SDF, queries, and point cloud saved to {output_file}")
    print(f"Mesh saved as {output_file.replace('.npz', '.ply')}")


def run_trimesh(input_file, output_file):
    """
    Run the trimesh package method.
    """
    mesh = trimesh.load(input_file)
    query_points = mesh.vertices + np.random.randn(*mesh.vertices.shape) * 0.01
    sdf = trimesh.proximity.signed_distance(mesh, query_points)
    np.savez(output_file, sdf=sdf)
    print(f"SDF saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Convert Mesh or Point Cloud to SDF")
    parser.add_argument('--input_file', type=str, help="Path to .ply File")
    parser.add_argument('--output_file', type=str, help="Path to .npz SDF File.")
    parser.add_argument('--method', type=str, default='vanilla', choices=['vanilla', 'mesh-to-sdf', 'trimesh'],
                        help="Method to use for SDF Conversion: 'vanilla', 'mesh-to-sdf', or 'trimesh'")
    parser.add_argument('--input_type', type=str, default='mesh', choices=['mesh', 'pointcloud'],
                        help="Type of Input: 'mesh' or 'pointcloud'")
    parser.add_argument('--voxel_size', type=float, default=0.1, help="Voxel Size for Vanilla")
    parser.add_argument('--voxel_resolution', type=int, default=128, help="Voxel Resolution for mesh-to-sdf")

    args = parser.parse_args()

    if args.method == 'vanilla':
        run_vanilla(args.input_file, args.output_file, args.input_type, args.voxel_size)
    elif args.method == 'mesh-to-sdf':
        run_mesh_to_sdf(args.input_file, args.output_file, args.voxel_resolution)
    elif args.method == 'trimesh':
        run_trimesh(args.input_file, args.output_file)
    else:
        print("Invalid Method Selected.")


if __name__ == "__main__":
    main()
