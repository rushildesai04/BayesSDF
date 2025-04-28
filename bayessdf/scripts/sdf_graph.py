import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
import torch
import ipdb

def calculate_grid(pred_sdf, gt_sdf, type):
    if pred_sdf.shape != gt_sdf.shape:
        raise ValueError("Predicted SDF and Ground Truth SDF Do Not Have Same Shape")
    if type == 'mae':
        return np.abs(pred_sdf - gt_sdf)
    elif type == 'mse':
        return np.abs(pred_sdf - gt_sdf) ** 2
    else:
        return np.sqrt(np.abs(pred_sdf - gt_sdf) ** 2)

def calculate_ensemble_curve(variance_grid, mae_grid):
    variances = variance_grid.flatten()
    mae_values = mae_grid.flatten()

    sorted_indices = np.argsort(-mae_values)
    sorted_values = mae_values[sorted_indices]

    sorted_var_indices = np.argsort(-variances)
    sorted_var_values = mae_values[sorted_var_indices]

    cumulative_mae = np.cumsum(sorted_values) / np.arange(1, len(sorted_values) + 1)
    cumulative_var_mae = np.cumsum(sorted_var_values) / np.arange(1, len(sorted_var_values) + 1)

    return cumulative_mae, cumulative_var_mae

def plot_ensemble_curve(variance_grid, mae_grid, output_file, parent_dir):
    os.makedirs(f"{output_file}/{parent_dir}", exist_ok=True)

    variances = variance_grid.flatten()
    mae_values = mae_grid.flatten()

    err_vec = torch.tensor(mae_values)
    unc_vec = torch.tensor(variances)
    err_type = 'mae'

    ratio_removed = np.linspace(0, 1, 100, endpoint=False)
    err_vec_sorted, _ = torch.sort(err_vec)
    n_valid_pixels = len(err_vec)
    ause_err = []

    for r in ratio_removed:
        err_slice = err_vec_sorted[0:int((1-r)*n_valid_pixels)]
        if err_type == 'rmse':
            ause_err.append(torch.sqrt(err_slice.mean()).cpu().numpy())
        elif err_type == 'mae' or err_type == 'mse':
            ause_err.append(err_slice.mean().cpu().numpy())
    
    _, var_vec_sorted_idxs = torch.sort(unc_vec)
    err_vec_sorted_by_var = err_vec[var_vec_sorted_idxs]
    ause_err_by_var = []

    for r in ratio_removed:
        err_slice = err_vec_sorted_by_var[0:int((1 - r) * n_valid_pixels)]
        if err_type == 'rmse':
            ause_err_by_var.append(torch.sqrt(err_slice.mean()).cpu().numpy())
        elif err_type == 'mae'or err_type == 'mse':
            ause_err_by_var.append(err_slice.mean().cpu().numpy())
    
    max_val = max(max(ause_err), max(ause_err_by_var))
    ause_err = ause_err / max_val
    ause_err = np.array(ause_err)
    
    ause_err_by_var = ause_err_by_var / max_val
    ause_err_by_var = np.array(ause_err_by_var)

    percentiles = ratio_removed
    cumulative_mae = ause_err
    cumulative_var_mae = ause_err_by_var
    cumulative_difference = abs(cumulative_mae - cumulative_var_mae)

    # sorted_indices = np.argsort(-mae_values)
    # sorted_values = mae_values[sorted_indices]
    
    # sorted_var_indices = np.argsort(-variances)
    # sorted_var_values = mae_values[sorted_var_indices]

    # cumulative_mae = np.cumsum(sorted_values) / np.arange(1, len(sorted_values) + 1)
    # cumulative_var_mae = np.cumsum(sorted_var_values) / np.arange(1, len(sorted_var_values) + 1)

    # cumulative_mae = sorted_values
    # cumulative_var_mae = np.convolve(sorted_var_values, np.ones(5000) / 5000, mode='same')

    # max_val = max(max(cumulative_mae), max(cumulative_var_mae))
    
    # cumulative_mae = cumulative_mae / max_val
    # cumulative_var_mae = cumulative_var_mae / max_val

    # cumulative_difference = abs(cumulative_mae - cumulative_var_mae)

    # percentiles = np.linspace(0, 100, len(cumulative_mae))

    plt.figure(figsize=(8, 5))
    plt.plot(percentiles, cumulative_mae, label="MAE SDF Sorted", color="purple", linewidth=2)
    plt.xlabel("SDF Grid Value Sparsification (Most Uncertain to Most Certain %)")
    plt.ylabel("Cumulative ΔMAE")
    plt.title("ΔMAE vs. SDF Grid Value Sparsification")
    plt.legend()
    plt.grid()
    if output_file:
        plt.savefig(f"{output_file}/{parent_dir}/mae_norm.png")
    else:
        plt.show()
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(percentiles, cumulative_var_mae, label="MAE SDF Sorted By Var", color="blue", linewidth=2)
    plt.xlabel("SDF Grid Value Sparsification (Most Uncertain to Most Certain %)")
    plt.ylabel("Cumulative ΔMAE")
    plt.title("ΔMAE vs. SDF Grid Value Sparsification")
    plt.legend()
    plt.grid()
    if output_file:
        plt.savefig(f"{output_file}/{parent_dir}/mae_var.png")
    else:
        plt.show()
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(percentiles, cumulative_difference, label="MAE SDF Difference", color="red", linewidth=2)
    plt.xlabel("SDF Grid Value Sparsification (Most Uncertain to Most Certain %)")
    plt.ylabel("Cumulative ΔMAE")
    plt.title("ΔMAE vs. SDF Grid Value Sparsification")
    plt.legend()
    plt.grid()
    if output_file:
        plt.savefig(f"{output_file}/{parent_dir}/mae_diff.png")
    else:
        plt.show()
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(percentiles, cumulative_mae, label="MAE SDF Sorted", color="purple", linewidth=2)
    plt.plot(percentiles, cumulative_var_mae, label="MAE SDF Sorted By Var", color="blue", linewidth=2)
    plt.plot(percentiles, cumulative_difference, label="MAE SDF Difference", color="red", linewidth=2)
    plt.xlabel("SDF Grid Value Sparsification (Most Uncertain to Most Certain %)")
    plt.ylabel("Cumulative ΔMAE")
    plt.title("ΔMAE vs. SDF Grid Value Sparsification")
    plt.legend()
    plt.grid()
    if output_file:
        plt.savefig(f"{output_file}/{parent_dir}/mae_cumul.png")
    else:
        plt.show()
    plt.close()

def plot_render_curve(variance_grid, mae_grid, output_file, parent_dir):
    os.makedirs(f"{output_file}/{parent_dir}", exist_ok=True)

    variances = variance_grid.flatten()
    mae_values = mae_grid.flatten()

    err_vec = torch.tensor(mae_values)
    unc_vec = torch.tensor(variances)
    err_type = 'mae'

    ratio_removed = np.linspace(0, 1, 100, endpoint=False)
    err_vec_sorted, _ = torch.sort(err_vec)
    n_valid_pixels = len(err_vec)
    ause_err = []

    for r in ratio_removed:
        err_slice = err_vec_sorted[0:int((1-r)*n_valid_pixels)]
        if err_type == 'rmse':
            ause_err.append(torch.sqrt(err_slice.mean()).cpu().numpy())
        elif err_type == 'mae' or err_type == 'mse':
            ause_err.append(err_slice.mean().cpu().numpy())
    
    _, var_vec_sorted_idxs = torch.sort(unc_vec)
    err_vec_sorted_by_var = err_vec[var_vec_sorted_idxs]
    ause_err_by_var = []

    for r in ratio_removed:
        err_slice = err_vec_sorted_by_var[0:int((1 - r) * n_valid_pixels)]
        if err_type == 'rmse':
            ause_err_by_var.append(torch.sqrt(err_slice.mean()).cpu().numpy())
        elif err_type == 'mae'or err_type == 'mse':
            ause_err_by_var.append(err_slice.mean().cpu().numpy())
    
    max_val = max(max(ause_err), max(ause_err_by_var))
    ause_err = ause_err / max_val
    ause_err = np.array(ause_err)
    
    ause_err_by_var = ause_err_by_var / max_val
    ause_err_by_var = np.array(ause_err_by_var)

    percentiles = ratio_removed
    cumulative_mae = ause_err
    cumulative_var_mae = ause_err_by_var
    cumulative_difference = abs(cumulative_mae - cumulative_var_mae)

    # sorted_indices = np.argsort(-mae_values)
    # sorted_values = mae_values[sorted_indices]
    
    # sorted_var_indices = np.argsort(-variances)
    # sorted_var_values = mae_values[sorted_var_indices]

    # cumulative_mae = np.cumsum(sorted_values) / np.arange(1, len(sorted_values) + 1)
    # cumulative_var_mae = np.cumsum(sorted_var_values) / np.arange(1, len(sorted_var_values) + 1)

    # cumulative_mae = sorted_values
    # cumulative_var_mae = np.convolve(sorted_var_values, np.ones(5000) / 5000, mode='same')

    # max_val = max(max(cumulative_mae), max(cumulative_var_mae))
    
    # cumulative_mae = cumulative_mae / max_val
    # cumulative_var_mae = cumulative_var_mae / max_val
    # cumulative_difference = abs(cumulative_mae - cumulative_var_mae)

    percentiles = np.linspace(0, 100, len(cumulative_mae))

    plt.figure(figsize=(8, 5))
    plt.plot(percentiles, cumulative_mae, label="MAE Render Sorted", color="purple", linewidth=2)
    plt.xlabel("Rendering Pixel Sparsification (Most Uncertain to Most Certain %)")
    plt.ylabel("Cumulative ΔMAE")
    plt.title("ΔMAE vs. Rendering Pixel Sparsification")
    plt.legend()
    plt.grid()
    if output_file:
        plt.savefig(f"{output_file}/{parent_dir}/mae_norm.png")
    else:
        plt.show()
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(percentiles, cumulative_var_mae, label="MAE Render Sorted By Var", color="blue", linewidth=2)
    plt.xlabel("Rendering Pixel Sparsification (Most Uncertain to Most Certain %)")
    plt.ylabel("Cumulative ΔMAE")
    plt.title("ΔMAE vs. Rendering Pixel Sparsification")
    plt.legend()
    plt.grid()
    if output_file:
        plt.savefig(f"{output_file}/{parent_dir}/mae_var.png")
    else:
        plt.show()
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(percentiles, cumulative_difference, label="MAE Render Difference", color="red", linewidth=2)
    plt.xlabel("Rendering Pixel Sparsification (Most Uncertain to Most Certain %)")
    plt.ylabel("Cumulative ΔMAE")
    plt.title("ΔMAE vs. Rendering Pixel Sparsification")
    plt.legend()
    plt.grid()
    if output_file:
        plt.savefig(f"{output_file}/{parent_dir}/mae_diff.png")
    else:
        plt.show()
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(percentiles, cumulative_mae, label="MAE Render Sorted", color="purple", linewidth=2)
    plt.plot(percentiles, cumulative_var_mae, label="MAE Render Sorted By Var", color="blue", linewidth=2)
    plt.plot(percentiles, cumulative_difference, label="MAE Render Difference", color="red", linewidth=2)
    plt.xlabel("Rendering Pixel Sparsification (Most Uncertain to Most Certain %)")
    plt.ylabel("Cumulative ΔMAE")
    plt.title("ΔMAE vs. Rendering Pixel Sparsification")
    plt.legend()
    plt.grid()
    if output_file:
        plt.savefig(f"{output_file}/{parent_dir}/mae_cumul.png")
    else:
        plt.show()
    plt.close()

def plot_box_whisker_sdfs(gt_sdf, variance_sdf, mean_sdf, ensemble_sdfs, output_file=None):
    gt_sdf_flat = gt_sdf.flatten()
    variance_sdf_flat = variance_sdf.flatten()
    mean_sdf_flat = mean_sdf.flatten()
    ensemble_sdfs_flat = [sdf.flatten() for sdf in ensemble_sdfs]

    model_labels = [f"Model {i+1}" for i in range(len(ensemble_sdfs))]
    data = ensemble_sdfs_flat

    plt.figure(figsize=(12, 6))
    plt.boxplot(data, vert=True, patch_artist=True, labels=model_labels)

    plt.axhline(np.mean(mean_sdf_flat), color='blue', linestyle='--', label="Mean SDF")
    plt.axhline(np.mean(variance_sdf_flat), color='red', linestyle='-.', label="Variance SDF")
    plt.axhline(np.mean(gt_sdf_flat), color='green', linestyle='-', label="Ground Truth SDF Mean")

    plt.title("Box And Whisker Plot of Ensemble SDFs")
    plt.ylabel("SDF Values")
    plt.xlabel("Ensemble Models")
    plt.legend(loc="upper right")
    plt.grid(True)

    if output_file:
        plt.savefig(output_file)
        print(f"Box-and-whisker plot saved to {output_file}")
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description="Calculate MAE grid and plot ensemble curve.")
    parser.add_argument('--pred_sdf', type=str, required=True, help="Path to the predicted SDF file")
    parser.add_argument('--gt_sdf', type=str, required=True, help="Path to the ground truth SDF file")
    parser.add_argument('--stats_sdf', type=str, required=True, help="Path to the ensemble stats SDF file")
    parser.add_argument('--pred_render', type=str, required=True, help="Path to the predicted Rendering file")
    parser.add_argument('--gt_render', type=str, required=True, help="Path to the ground truth Rendering file")
    parser.add_argument('--stats_render', type=str, required=True, help="Path to the ensemble stats Rendering file")
    parser.add_argument('--output_plot', type=str, default=None, help="Path to save the output SDF plot")
    parser.add_argument('--output_render', type=str, default=None, help="Path to save the output Rendering plot")
    parser.add_argument('--output_whisker', type=str, default=None, help="Path to save the output file")
    parser.add_argument('--emsemble_sdfs', type=str, nargs='+', default=None, help="Paths to ensemble models")
    args = parser.parse_args()

    pred_sdf = np.load(args.pred_sdf)['values'].reshape((128, 128, 128))
    gt_sdf = np.load(args.gt_sdf)['sdf']
    stats_sdf = np.load(args.stats_sdf)
    var_sdf = stats_sdf['var']
    mean_sdf = stats_sdf['mean']
    mae_sdf = calculate_grid(pred_sdf, gt_sdf, 'mae')
    ensemble_sdfs = [np.load(pth)['values'] for pth in args.ensemble_sdfs]

    output_plot = args.output_plot
    output_render = args.output_render
    output_whisker = args.output_whisker

    # calculate_ensemble_curve(var_sdf, mae_sdf)
    plot_ensemble_curve(var_sdf, mae_sdf, output_plot, 'sdf_plots')
    plot_box_whisker_sdfs(gt_sdf, var_sdf, mean_sdf, ensemble_sdfs, output_whisker)

    print("Created SDF Graphs")

    pred_render = np.load(args.pred_render)[0]
    gt_render = np.load(args.gt_render)['mean'][0]
    stats_render = np.load(args.stats_render)
    var_render = stats_render['var'][0]
    mae_render = calculate_grid(pred_render, gt_render, 'mae')
    plot_render_curve(var_render, mae_render, output_render, 'render_plots')

    print("Created Render Graphs")

if __name__ == "__main__":
    main()
