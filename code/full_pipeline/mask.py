# Script for preprocessing data for slice-to-volume reconstructions
# Generates a ROI mask and creates experiment directiries
#
# Author: M.Y. Kingma

import os
import argparse
from utils.nifti import load_nifti_file, save_nifti
from utils.mask import compute_brain_mask
from utils.plot import plot_scrollable_volume


def main(args):
    # Loop over directories in data directory
    for directory in os.listdir(args.data_path):

        # Check if directory is a directory
        if os.path.isdir(os.path.join(args.data_path, directory)):

            # Get files containing 'output' and 'init' in name
            for file in os.listdir(os.path.join(args.data_path, directory)):
                if 'input_trans' in file:

                    # Open nifit file
                    nifti_data, _, nifti_transformation_matrix = load_nifti_file(
                        os.path.join(args.data_path, directory, file))

                    # Compute brain mask
                    brain_mask = compute_brain_mask(
                        nifti_data, mask_per_slice=args.mask_per_slice, axis=args.axial_axis, threshold=args.threshold)

                    # Plot brain mask
                    plot_scrollable_volume(brain_mask)

                    # Save brain mask
                    mask_file_name = f"{file.split('.')[0]}_brain_mask.nii.gz"
                    save_nifti(brain_mask, nifti_transformation_matrix,
                               os.path.join(args.data_path, directory), mask_file_name)


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Rename and move files in data directory")
    parser.add_argument("-d", "--data_path", type=str,
                        help="path to data directory")
    parser.add_argument("-mps", "--mask-per-slice", action="store_true",
                        help="Compute brain mask per slice")
    parser.add_argument("-aa", "--axial-axis", type=int, required=True,
                        help="Axial axis, axis that will be used for masking per slice")
    parser.add_argument("-t", "--threshold", type=float, default=None,
                        help="Threshold value for brain mask")
    args = parser.parse_args()

    # Print arguments
    print("Collecting reconstructions with the following arguments:")
    for arg in vars(args):
        print(arg, getattr(args, arg))

    # Run main function
    main(args)
