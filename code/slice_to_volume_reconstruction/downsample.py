# Script for downsampling the HCP data in two orientations
#
# Outputs the downsampled data to a subject-specific directory
#
# Author: M.Y. Kingma

import os
import argparse
from utils.preprocessing import downsample_volume, normalize_volume
from utils.nifti import load_nifti_file, save_nifti
from utils.plot import plot_scrollable_volume
from utils.mask import compute_brain_mask


def main(args):
    # Get all the nifti files in the directory
    nifti_files = [f for f in os.listdir(
        args.data_path) if f.endswith('.nii.gz')]

    # Loop through each nifti file
    for file_name in nifti_files:
        # Load the nifti data
        nifti_data, _, nifti_transformation_matrix = load_nifti_file(
            os.path.join(args.data_path, file_name))

        # Normalize the nifti data
        nifti_data = normalize_volume(nifti_data)

        # Get the downsample axes
        downsample_axes = [0, 1, 2]
        downsample_axes.remove(args.coronal_axis)

        # Create a new directory for the downsampled files
        downsampled_directory_path = os.path.join(
            args.output_path, file_name.split('.')[0])
        os.makedirs(downsampled_directory_path, exist_ok=True)

        for index, axis in enumerate(downsample_axes):
            # Downsample the nifti data
            downsampled_nifti_data, downsampled_transformation_matrix = downsample_volume(
                nifti_data, nifti_transformation_matrix, axis, args.downsample_factor)

            # Generate filename
            downsampled_file_name = f"{file_name.split('.')[0]}_downsampled_{axis}.nii.gz"

            # Save the downsampled nifti
            save_nifti(downsampled_nifti_data,
                       downsampled_transformation_matrix, downsampled_directory_path, downsampled_file_name)

            # If first iteration, compute brain mask
            if index == 0 and args.mask:
                # Compute brain mask
                brain_mask = compute_brain_mask(
                    downsampled_nifti_data, mask_per_slice=args.mask_per_slice, axis=args.axial_axis, threshold=args.threshold)

                # Save brain mask
                mask_file_name = f"{file_name.split('.')[0]}_brain_mask.nii.gz"
                save_nifti(brain_mask, downsampled_transformation_matrix,
                           downsampled_directory_path, mask_file_name)


if __name__ == "__main__":
    # Create argument parser object
    parser = argparse.ArgumentParser(
        description="Downsample HCP data")

    # Add arguments
    parser.add_argument("-d", "--data-path", type=str, required=True,
                        help="Path to directory containing high-resolution NIfTI images")
    parser.add_argument("-df", "--downsample-factor", type=int, default=2,
                        help="Downsample factor")
    parser.add_argument("-ca", "--coronal-axis", type=int, required=True,
                        help="Coronal axis, axis that will not be downsampled")
    parser.add_argument("-o", "--output-path", type=str, required=True,
                        help="Path to output directory")
    parser.add_argument("-m", "--mask", action="store_true",
                        help="Compute brain mask")
    parser.add_argument("-mps", "--mask-per-slice", action="store_true",
                        help="Compute brain mask per slice")
    parser.add_argument("-aa", "--axial-axis", type=int, required=True,
                        help="Axial axis, axis that will be used for masking per slice")
    parser.add_argument("-t", "--threshold", type=float, default=None,
                        help="Threshold value for brain mask")

    # Parse the arguments
    args = parser.parse_args()

    # Print the arguments
    print("Running SVR preparation with the following arguments:")
    for arg in vars(args):
        print(f"{arg}:", getattr(args, arg))
    main(args)
