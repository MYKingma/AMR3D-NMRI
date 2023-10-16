# Script to evaluate the SVRTK output, comparing it to the ground truth and the manually upsampled ground truth
#
# Author: Maurice Kingma

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import ttest_rel
from scipy.ndimage import binary_erosion
import seaborn as sns

from utils.nifti import load_nifti_file
from utils.plot import plot_slice, plot_scrollable_volume, get_p_value_significance_symbols, create_visual_inspection_plots
from utils.mask import compute_brain_mask
from utils.preprocessing import normalize_volume
from utils.metrics import calculate_blur_effect_per_slice, calculate_ssim_per_slice


def export_binary_erosion_comparison(recon_data, manual_data, init_data, args, recon_file, gt_data=None):
    # Plot slice of coronal axis for all volumes side by side
    plot = plt.figure(figsize=(15, 5))
    if gt_data is not None:
        plot.add_subplot(3, 4, 1)
        plt.imshow(recon_data[:, :, 100], cmap="gray")
        plt.title("Recon")
        plot.add_subplot(3, 4, 2)
        plt.imshow(gt_data[:, :, 100], cmap="gray")
        plt.title("GT")
        plot.add_subplot(3, 4, 3)
        plt.imshow(manual_data[:, :, 100], cmap="gray")
        plt.title("Manual")
        plot.add_subplot(3, 4, 4)
        plt.imshow(init_data[:, :, 100], cmap="gray")
        plt.title("Init")
        plot.add_subplot(3, 4, 5)
        plt.imshow(recon_data[:, 100, :], cmap="gray")
        plot.add_subplot(3, 4, 6)
        plt.imshow(gt_data[:, 100, :], cmap="gray")
        plot.add_subplot(3, 4, 7)
        plt.imshow(manual_data[:, 100, :], cmap="gray")
        plot.add_subplot(3, 4, 8)
        plt.imshow(init_data[:, 100, :], cmap="gray")
        plot.add_subplot(3, 4, 9)
        plt.imshow(recon_data[100, :, :], cmap="gray")
        plot.add_subplot(3, 4, 10)
        plt.imshow(gt_data[100, :, :], cmap="gray")
        plot.add_subplot(3, 4, 11)
        plt.imshow(manual_data[100, :, :], cmap="gray")
        plot.add_subplot(3, 4, 12)
        plt.imshow(init_data[100, :, :], cmap="gray")
    else:
        plot.add_subplot(3, 3, 1)
        plt.imshow(recon_data[:, :, 100], cmap="gray")
        plt.title("Recon")
        plot.add_subplot(3, 3, 2)
        plt.imshow(manual_data[:, :, 100], cmap="gray")
        plt.title("Manual")
        plot.add_subplot(3, 3, 3)
        plt.imshow(init_data[:, :, 100], cmap="gray")
        plt.title("Init")
        plot.add_subplot(3, 3, 4)
        plt.imshow(recon_data[:, 100, :], cmap="gray")
        plot.add_subplot(3, 3, 5)
        plt.imshow(manual_data[:, 100, :], cmap="gray")
        plot.add_subplot(3, 3, 6)
        plt.imshow(init_data[:, 100, :], cmap="gray")
        plot.add_subplot(3, 3, 7)
        plt.imshow(recon_data[100, :, :], cmap="gray")
        plot.add_subplot(3, 3, 8)
        plt.imshow(manual_data[100, :, :], cmap="gray")
        plot.add_subplot(3, 3, 9)
        plt.imshow(init_data[100, :, :], cmap="gray")

    # Save plot
    eval_type = "HCP" if args.ground_truth_path else "UTRECHT"
    plot_filename = f"man_comparison_{args.binary_erosion}_iter.png"
    plt.savefig(os.path.join(
        "/Users/Maurice/Downloads/compare", recon_file.replace(".nii.gz", f"_{eval_type}_{plot_filename}")))
    plt.close()


def evaluate_scores(eval_type, score_type, recon_scores, manual_scores, init_scores, gt_scores=np.array([]), visual_evaluation_slice=None, box_y_lim=(0, 1)):
    # Get scores for visual evaluation slice
    visual_evaliation_scores = {}
    if visual_evaluation_slice is not None:
        visual_evaliation_scores["Slice-to-volume"] = recon_scores[visual_evaluation_slice]
        visual_evaliation_scores["B-spline"] = manual_scores[visual_evaluation_slice]
        visual_evaliation_scores["Gaussian"] = init_scores[visual_evaluation_slice]
        if len(gt_scores) > 0:
            visual_evaliation_scores["Ground truth"] = gt_scores[visual_evaluation_slice]

    # Get all nan indices for all arrays
    nan_indices_recon = np.argwhere(np.isnan(recon_scores))
    nan_indices_manual = np.argwhere(np.isnan(manual_scores))
    nan_indices_init = np.argwhere(np.isnan(init_scores))
    nan_indices_gt = np.argwhere(np.isnan(gt_scores))

    # Creata an array containing nan indices present in all arrays
    nan_indices_arrays = (nan_indices_recon, nan_indices_manual, nan_indices_init, nan_indices_gt) if len(
        gt_scores) > 0 else (nan_indices_recon, nan_indices_manual, nan_indices_init)
    nan_indices = np.concatenate(
        nan_indices_arrays, axis=0)
    nan_indices = np.unique(nan_indices, axis=0)

    # Get non nan scores using the nan indices
    recon_scores = np.delete(recon_scores, nan_indices)
    manual_scores = np.delete(manual_scores, nan_indices)
    init_scores = np.delete(init_scores, nan_indices)
    gt_scores = np.delete(gt_scores, nan_indices) if len(
        gt_scores) > 0 else gt_scores

    # Add scores to list
    score_list = [
        {"type": "Slice-to-volume reconstruction", "scores": recon_scores},
        {"type": "B-spline interpolation", "scores": manual_scores},
        {"type": "Gaussian reconstruction", "scores": init_scores},
        {"type": "Ground truth", "scores": gt_scores}
    ]

    print("-------------------------------------")

    # Print average scores
    for score in score_list:
        if score["scores"].size == 0:
            continue
        average_score = sum(score["scores"]) / len(score["scores"])
        print()
        print(f"{score_type.upper()} SCORES {score['type']}:")
        print(f"Average {score_type} score: ", average_score, "Standard deviation: ", np.std(
            score["scores"]), "Amount of slices:", len(score["scores"]))

    # Perform a paired t-test
    t_statistic, p_value = ttest_rel(recon_scores, manual_scores)
    print()
    print("Paired t-test Slice-to-volume reconstruction vs B-spline interpolation:")
    print("T-statistic:", t_statistic)
    print("P-value:", f"{p_value}")

    # Print significant difference
    print("Significant difference:", p_value < 0.05)
    print("-------------------------------------")

    # Create a box plot for each group
    figsize = (5, 4)  # if len(gt_scores) > 0 else (6.4, 4.8)
    fig, ax = plt.subplots(figsize=figsize)
    boxplot = ax.boxplot([recon_scores, manual_scores],
                         patch_artist=True, widths=0.5)

    # Change the background colour of the boxes to Seaborn's 'pastel' palette
    colors = sns.color_palette("pastel")
    for patch, color in zip(boxplot['boxes'], colors):
        patch.set_facecolor(color)

    # Add labels and title
    ax.set_xticklabels(
        ["Slice-to-volume", "B-spline"], fontsize=13)

    # Set y limit from 0 to 1
    ax.set_ylim(box_y_lim)

    # Round y ticks to 1 decimal
    ax.set_yticklabels([round(tick, 1) for tick in ax.get_yticks()])

    if "blur" in score_type:
        ax.set_ylabel("Estimated blur strength", fontsize=13)
    elif score_type == "ssim":
        ax.set_ylabel("SSIM index")

    # Get the y-axis limits
    bottom, top = ax.get_ylim()
    y_range = top - bottom

    # Get the significance symbol
    significance_symbol = get_p_value_significance_symbols(p_value)

    # Add p-value to plot
    bar_height = (y_range * 0.02) + top
    bar_tips = bar_height + (y_range * 0.02)
    ax.plot([1, 1, 2, 2], [bar_height, bar_tips, bar_tips, bar_height],
            linewidth=1.5, color="black")
    text_height = bar_height + (y_range * 0.05)

    # Add annotation text
    ax.text(1.5, text_height, significance_symbol, horizontalalignment="center",
            verticalalignment="bottom", fontsize=11)

    # Increase y axis limit
    ax.set_ylim(bottom, top + (y_range * 0.2))

    # Colour of the median lines
    plt.setp(boxplot['medians'], color='k')

    # Save plot
    plt.savefig(os.path.join(
        "/Users/Maurice/Downloads/plots", f"{eval_type}_boxplot_{score_type}.png"))
    plt.close()

    return visual_evaliation_scores


def main(args):
    blur_gt = []
    blur_manual = []
    blur_recon = []
    blur_init = []
    ssim_manual = []
    ssim_recon = []
    ssim_init = []

    # Set evaluation type
    eval_type = "HCP" if args.ground_truth_path else "UTRECHT"

    # List files in data directory, not containing 'init' in name
    recon_files = [file for file in os.listdir(
        args.data_path) if 'init' not in file]

    recon_files = recon_files[:1] if args.visual_evaluation and args.ground_truth_path else recon_files

    # Loop over files in data directory
    for recon_file in tqdm(recon_files):
        if args.pipeline_scan and args.pipeline_scan != recon_file.split(".")[0]:
            continue

        # Check if file is nifti file
        if not recon_file.endswith(".nii.gz") or recon_file.endswith(".nii"):
            continue

        # Get path to recon and init file
        recon_file_path = os.path.join(args.data_path, recon_file)
        init_file_path = os.path.join(
            args.data_path, recon_file.replace(".nii.gz", "_init.nii.gz"))

        # Get ground truth file and manually upsampled ground truth file inside groud truth directory and manual directory, same name as recon file
        if args.ground_truth_path:
            gt_file_path = os.path.join(
                args.ground_truth_path, recon_file.replace(".gz", ""))
        if args.ground_truth_path:
            manual_file_path = os.path.join(
                args.manual_path, recon_file.replace(".nii.gz", "_downsampled_0.nii"))
        else:
            manual_file_path = os.path.join(
                args.manual_path, recon_file.replace(".nii.gz", "_input_tans.nii"))

        # Read all nii files
        recon_data, _, _ = load_nifti_file(
            recon_file_path)
        if args.ground_truth_path:
            gt_data, _, _ = load_nifti_file(gt_file_path)
        manual_data, _, _ = load_nifti_file(manual_file_path)
        init_data, _, _ = load_nifti_file(init_file_path)

        # Convert -1 values to 1 in recon data
        recon_data[recon_data == -1] = 0
        manual_data[manual_data == -1] = 0
        init_data[init_data == -1] = 0

        # Normalize data
        recon_data = normalize_volume(recon_data)
        if args.ground_truth_path:
            gt_data = normalize_volume(gt_data)
        manual_data = normalize_volume(manual_data)
        init_data = normalize_volume(init_data)

        visual_inspection_plot_data = [{"name": "Slice-to-volume", "volume_data": np.copy(recon_data)}, {"name": "B-spline", "volume_data": np.copy(manual_data)}, {
                                       "name": "Gaussian", "volume_data": np.copy(init_data)}]
        if args.ground_truth_path:

            # Add ground truth to visual inspection plot data, to the front
            visual_inspection_plot_data.insert(
                0, {"name": "Ground truth", "volume_data": np.copy(gt_data)})

        # Calculate mask for gt, or for recon if no gt is present
        if args.ground_truth_path:
            mask = compute_brain_mask(
                np.copy(gt_data), threshold=0.1, mask_per_slice=True, axis=2)
        else:
            mask = compute_brain_mask(
                np.copy(recon_data), threshold=0.1, mask_per_slice=True, axis=2)

        # Shrink mask using binary erosion
        mask = binary_erosion(mask, iterations=15)

        # Apply mask
        if args.ground_truth_path:
            gt_data[~mask.astype(bool)] = 0
        manual_data[~mask.astype(bool)] = 0
        recon_data[~mask.astype(bool)] = 0
        init_data[~mask.astype(bool)] = 0

        if args.evaluate_binary_erosion:
            export_binary_erosion_comparison(
                recon_data, manual_data, init_data, args, recon_file, gt_data if args.ground_truth_path else None)

        # Calculate blur per slice
        blur_recon_per_slice = calculate_blur_effect_per_slice(
            recon_data, args.coronal_axis, args.h_size)
        if args.ground_truth_path:
            blur_gt_per_slice = calculate_blur_effect_per_slice(
                gt_data, args.coronal_axis, args.h_size)
        blur_manual_per_slice = calculate_blur_effect_per_slice(
            manual_data, args.coronal_axis, args.h_size)
        blur_init_per_slice = calculate_blur_effect_per_slice(
            init_data, args.coronal_axis, args.h_size)

        # Concat list of scores with previous scores
        blur_recon += blur_recon_per_slice
        if args.ground_truth_path:
            blur_gt += blur_gt_per_slice
        blur_manual += blur_manual_per_slice
        blur_init += blur_init_per_slice

        if args.ground_truth_path:
            # Calculate SSIM between gt and recon, gt and manual
            ssim_recon_per_slice = calculate_ssim_per_slice(
                recon_data, gt_data, args.coronal_axis)
            ssim_manual_per_slice = calculate_ssim_per_slice(
                manual_data, gt_data, args.coronal_axis)
            ssim_init_per_slice = calculate_ssim_per_slice(
                init_data, gt_data, args.coronal_axis)

            # Concat list of scores with previous scores
            ssim_recon += ssim_recon_per_slice
            ssim_manual += ssim_manual_per_slice
            ssim_init += ssim_init_per_slice
        else:

            # Evaluate scores
            print(f"Evaluating {recon_file}")
            blur_score_slice = evaluate_scores(eval_type, f"blur_{recon_file.split('.')[0]}", blur_recon_per_slice,
                                               blur_manual_per_slice, blur_init_per_slice, visual_evaluation_slice=args.visual_evaluation_slice, box_y_lim=(0.4, 0.85))

            if args.visual_evaluation and args.pipeline_scan == "cirim-cirim":
                create_visual_inspection_plots(
                    visual_inspection_plot_data, 1, 100, (90, 130), (20, 60), 90, blur_score_slice, {}, name_suffix=f"coronal_Pipeline_{args.visual_evaluation_slice}_{recon_file.split('.')[0]}", x_offset_zoom=0.0, y_offset_zoom=0.02, offset_title=0.92)

                create_visual_inspection_plots(
                    visual_inspection_plot_data, 0, 100, (100, 150), (2, 52), 92, blur_scores_dict={}, ssim_scores_dict={}, name_suffix=f"sagittal_Pipeline_{args.visual_evaluation_slice}_{recon_file.split('.')[0]}", x_offset_zoom=0, y_offset_zoom=0.02, offset_title=0.92)

                create_visual_inspection_plots(
                    visual_inspection_plot_data, 2, 100, (85, 125), (50, 90), 90, blur_scores_dict={}, ssim_scores_dict={}, name_suffix=f"axial_Pipeline_{args.visual_evaluation_slice}_{recon_file.split('.')[0]}", x_offset_zoom=0.02, y_offset_zoom=0, offset_title=0.95)

            elif args.visual_evaluation and args.pipeline_scan == "cirim-dicom":
                create_visual_inspection_plots(
                    visual_inspection_plot_data, 1, 100, (50, 90), (15, 55), 90, blur_score_slice, {}, name_suffix=f"coronal_Pipeline_{args.visual_evaluation_slice}_{recon_file.split('.')[0]}", x_offset_zoom=0.0, y_offset_zoom=0.02, offset_title=0.95)

                create_visual_inspection_plots(
                    visual_inspection_plot_data, 0, 100, (60, 100), (15, 55), 90, blur_scores_dict={}, ssim_scores_dict={}, name_suffix=f"sagittal_Pipeline_{args.visual_evaluation_slice}_{recon_file.split('.')[0]}", x_offset_zoom=0, y_offset_zoom=0.02, offset_title=0.95)

                create_visual_inspection_plots(
                    visual_inspection_plot_data, 2, 100, (30, 70), (25, 65), 270, blur_scores_dict={}, ssim_scores_dict={}, name_suffix=f"axial_Pipeline_{args.visual_evaluation_slice}_{recon_file.split('.')[0]}", x_offset_zoom=0.02, y_offset_zoom=0, offset_title=0.95)
            elif args.visual_evaluation and args.pipeline_scan == "dicom-dicom":
                create_visual_inspection_plots(
                    visual_inspection_plot_data, 1, 100, (50, 90), (15, 55), 90, blur_score_slice, {}, name_suffix=f"coronal_Pipeline_{args.visual_evaluation_slice}_{recon_file.split('.')[0]}", x_offset_zoom=0.0, y_offset_zoom=0.02, offset_title=0.95)

                create_visual_inspection_plots(
                    visual_inspection_plot_data, 0, 100, (60, 100), (15, 55), 90, blur_scores_dict={}, ssim_scores_dict={}, name_suffix=f"sagittal_Pipeline_{args.visual_evaluation_slice}_{recon_file.split('.')[0]}", x_offset_zoom=0, y_offset_zoom=0.02, offset_title=0.95)

                create_visual_inspection_plots(
                    visual_inspection_plot_data, 2, 100, (30, 70), (25, 65), 270, blur_scores_dict={}, ssim_scores_dict={}, name_suffix=f"axial_Pipeline_{args.visual_evaluation_slice}_{recon_file.split('.')[0]}", x_offset_zoom=0.02, y_offset_zoom=0, offset_title=0.95)

            # Reset arrays
            blur_recon = []
            blur_manual = []
            blur_init = []

    if args.ground_truth_path:
        print(recon_file)

        # Evaluate scores
        blur_score_slice = evaluate_scores(eval_type, "blur", blur_recon,
                                           blur_manual, blur_init, blur_gt, visual_evaluation_slice=args.visual_evaluation_slice, box_y_lim=(0.2, 1))
        ssim_scores_slice = evaluate_scores(
            eval_type, "ssim", ssim_recon, ssim_manual, ssim_init, visual_evaluation_slice=args.visual_evaluation_slice, box_y_lim=(0.8, 1))

        if args.visual_evaluation:
            create_visual_inspection_plots(
                visual_inspection_plot_data, 1, 100, (160, 220), (90, 150), 90, blur_score_slice, ssim_scores_slice, name_suffix=f"coronal_HCP_{args.visual_evaluation_slice}",  offset_title=0.93)

            create_visual_inspection_plots(
                visual_inspection_plot_data, 0, 100, (100, 160), (2, 62), 90, blur_scores_dict={}, ssim_scores_dict={}, name_suffix=f"sagittal_HCP_{args.visual_evaluation_slice}", x_offset_zoom=0, y_offset_zoom=0.05,  offset_title=0.93)

            create_visual_inspection_plots(
                visual_inspection_plot_data, 2, 100, (160, 220), (90, 150), 90, blur_scores_dict={}, ssim_scores_dict={}, name_suffix=f"axial_HCP_{args.visual_evaluation_slice}", x_offset_zoom=0.05, y_offset_zoom=0,  offset_title=0.93)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_path", type=str, required=True,
                        help="Path to the directory containing the SVRTK output")
    parser.add_argument("-gt", "--ground_truth_path", type=str, required=False,
                        help="Path to the directory containing the ground truth")
    parser.add_argument("-m", "--manual_path", type=str, required=True,
                        help="Path to the directory containing the manually upsampled ground truth")
    parser.add_argument("-ca", "--coronal_axis", type=int, required=True,
                        help="Coronal axis of the volume, where the slices are taken from")
    parser.add_argument("-hs", "--h_size", type=int, default=11,
                        help="Size of the kernel used to calculate the blur score")
    parser.add_argument("-be", "--binary_erosion", type=int, default=15,
                        help="Amount of iterations used for binary erosion")
    parser.add_argument("-eb", "--evaluate_binary_erosion", action="store_true",
                        help="Whether to evaluate the binary erosion")
    parser.add_argument("-v", "--visual_evaluation", action="store_true",
                        help="Whether to create visual evaluation plots")
    parser.add_argument("-vsl", "--visual_evaluation_slice", type=int, default=100,
                        help="Slice to use for visual evaluation plots")
    parser.add_argument("-ps", "--pipeline_scan", type=str,
                        help="Pipeline scan to use for visual evaluation plots")

    args = parser.parse_args()

    # Print arguments
    print("Running SVRTK evaluation with the following arguments:")
    for arg in vars(args):
        print(f"{arg}:", getattr(args, arg))

    # Run main function
    main(args)
