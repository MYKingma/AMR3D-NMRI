# Script for preprocessing the UUMC data for SVR
#
# Author: M.Y. Kingma

import os
import numpy as np
import nibabel as nib

from utils.nifti import convert_hdf_file_to_nifti, load_nifti_file, is_hdf_file
from utils.plot import plot_scrollable_volume
from utils.preprocessing import create_transformation_matrix_nifti, normalize_volume
from utils.mask import compute_brain_mask


def save_stack_in_directory(volume_data, file_name, args):
    if not args.nifti_template and not args.nifti_template_sagittal:
        if args.debug:
            print("Volume data shape before permutation:", volume_data.shape)

        # Move slice dimension to the end (slice dimension is the dimension with the smallesr size)
        slice_dimension_index = np.argmin(volume_data.shape)

        if slice_dimension_index == 0:
            volume_data = np.transpose(volume_data, (2, 1, 0))
        elif slice_dimension_index == 1:
            volume_data = np.transpose(volume_data, (2, 0, 1))

        if args.debug:
            print("Volume data shape after permutation:", volume_data.shape)

    # Set the new voxel spacing if provided
    if new_voxel_spacing is not None:
        # Get index of the dimension with the smallest size
        min_dimension_index = np.argmin(volume_data.shape)

        # Get the highest value voxel spacing value
        max_voxel_value = np.max(new_voxel_spacing)

        # Move the highest voxel spacing value to the index of the dimension with the smallest size
        new_voxel_spacing = np.delete(new_voxel_spacing, 2)
        new_voxel_spacing = np.insert(
            new_voxel_spacing, min_dimension_index, max_voxel_value, axis=0)

        if args.debug:
            print("New voxel spacing:", new_voxel_spacing)

    # Create transformation matrix
    transformation_matrix = create_transformation_matrix_nifti(
        volume_data.shape, new_voxel_spacing, file_name, args)

    if args.int16:
        # Convert to int16
        volume_data = (volume_data * 32767).astype(np.int16)

    # Create nifti file with new voxel spacing
    nifti_data = nib.Nifti1Image(volume_data, transformation_matrix)

    if not args.nifti_template and not args.nifti_template_sagittal:
        # Set the new voxel spacing
        nifti_data.header.set_zooms(new_voxel_spacing)

        # Save the nifti file
        nib.save(nifti_data, os.path.join(args.data_path, file_name))


def preprocess_file(file_name, file_path, args):
    # Load the nifti file
    nifti_data, _, _ = load_nifti_file(
        file_path)

    # Normalize the volume
    nifti_data = normalize_volume(nifti_data, args.normalize_per_slice)

    if args.debug:
        print("Volume shape:", nifti_data.shape)
        print("Resolution:", args.resolution)
        print("Normalised volume min:", np.min(nifti_data))
        print("Normalised volume max:", np.max(nifti_data))

    # Save the volume
    save_stack_in_directory(
        nifti_data, file_name, args)

    if args.mask:
        # Compute the brain mask of first orientation
        brain_mask = compute_brain_mask(nifti_data, args)

        if args.debug and args.plot:
            plot_scrollable_volume(nifti_data)
            plot_scrollable_volume(brain_mask)
            print("Brain mask shape:", brain_mask.shape)

        # Create filename for mask (add _mask before extension)
        mask_filename = file_name.split(".")[0]
        mask_filename = file_name + "_mask.nii.gz"

        if args.debug:
            print("Mask voxel spacing:", args.resolution)

        # Save the mask
        save_stack_in_directory(
            brain_mask, mask_filename, args)


def main(args):
    # List all filenames in the data path
    filenames = os.listdir(args.data_path)

    # Loop over all filenames
    for file in filenames:
        # Get the file path
        file_path = os.path.join(args.data_path, file)

        # Check if file is a nifti file
        if is_hdf_file(file_path):

            # Convert the file to nifti
            convert_hdf_file_to_nifti(file_path, args.data_path, args)

            # Preprocess file
            preprocess_file(file, file_path, args)


if __name__ == "__main__":
    import argparse

    # Create argument parser object
    parser = argparse.ArgumentParser(
        description="Prepare data for SVR toolbox")

    def str_to_bool(value):
        if value.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif value.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    # Add arguments
    parser.add_argument("-d", "--data-path", type=str, required=True,
                        help="Path to directory containing high-resolution NIfTI images")
    parser.add_argument("-m", "--mask", action="store_true",
                        default=False, help="If provided, the mask will be created.")
    parser.add_argument("-nps", "--normalize-per-slice", action="store_true",
                        help="If provided, the volume will be normalized per slice. Otherwise, the volume will be normalized as a whole.")
    parser.add_argument("-r", "--resolution", type=float, nargs=3,
                        help="Resolution of the input data in mm.")
    parser.add_argument("-rs", "--resolution-sagittal", type=float, nargs=3,
                        help="Resolution of the input data in mm for the sagittal orientation.")
    parser.add_argument("-int16", "--int16", action="store_true",
                        default=False, help="If provided, the output files will be saved as int16. Otherwise, they will be saved as the data type provided.")
    parser.add_argument("-tp", "--transpose", type=int, nargs=3,
                        default=None, help="Transpose the volume. Default: None")
    parser.add_argument("-tps", "--transpose-sag", type=int, nargs=3,
                        default=None, help="Transpose the volume for the sagittal orientation. Default: None")
    parser.add_argument("-fl", "--flip", type=str_to_bool, nargs=3,
                        default=None, help="Flip the volume. Default: None")
    parser.add_argument("-fls", "--flip-sag", type=str_to_bool, nargs=3,
                        default=None, help="Flip the volume for the sagittal orientation. Default: None")
    parser.add_argument("-off", "--offset", type=float, nargs=3,
                        default=None, help="Offset the volume. Default: None")
    parser.add_argument("-offs", "--offset-sag", type=float, nargs=3,
                        default=None, help="Offset the volume for the sagittal orientation. Default: None")
    parser.add_argument("-db", "--debug", action="store_true",
                        help="Enable debug mode, only for single file processing (plots volume and mask)")
    parser.add_argument("-plt", "--plot", action="store_true",
                        help="Enable plotting of the volume and mask during debug mode")

    # Parse the arguments
    args = parser.parse_args()

    # Print the arguments
    print("Running SVR preparation with the following arguments:")
    for arg in vars(args):
        print(f"{arg}:", getattr(args, arg))
    main(args)
