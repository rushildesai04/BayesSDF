import numpy as np
import argparse
import os
import ipdb

def load_sdf_arrays(input_files):
    sdf_arrays = []
    for file in input_files:
        data = np.load(file)
        sdf_arrays.append(data)
    return np.stack(sdf_arrays, axis=0)

def calculate_statistics(sdf_arrays):
    var = np.var(sdf_arrays, axis=0)
    minimum = np.min(sdf_arrays, axis=0)
    maximum = np.max(sdf_arrays, axis=0)
    mean = np.mean(sdf_arrays, axis=0)
    return {'var': var, 'min': minimum, 'max': maximum, 'mean': mean}

def save_statistics(output_file, stats_dict):
    np.savez(output_file, **stats_dict)
    print(f"Statistics saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Calculate statistics across multiple SDFs.")
    parser.add_argument('--input_files', type=str, nargs='+', help="Paths to input .npz files containing SDF arrays.")
    parser.add_argument('--output_file', type=str, help="Path to save the calculated statistics as .npz.")

    args = parser.parse_args()

    if len(args.input_files) < 2:
        raise ValueError("At least two SDF files are required to calculate statistics.")

    sdf_arrays = load_sdf_arrays(args.input_files)
    print(f"Loaded {len(args.input_files)} SDF arrays with shape {sdf_arrays.shape[1:]}.")

    stats_dict = calculate_statistics(sdf_arrays)

    save_statistics(args.output_file, stats_dict)

if __name__ == "__main__":
    main()
