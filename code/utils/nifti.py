# Helper functions for handling nifti files
#
# Author: M.Y. Kingma


from nibabel import nifti1 as nib
import numpy as np
import h5py
import os

from utils.plot import plot_abs_angle_real_imag_from_complex_volume


def load_nifti_file(nifti_path):
    """
    Load a nifti file and return the data and the voxel spacing.

    Inputs:
    - nifti_path: A string representing the path to the nifti file.

    Returns:
    - nifti_data: A 3D numpy array representing the nifti data.
    - nifti_voxel_spacing: A tuple representing the voxel spacing.
    - nifti_transformation_matrix: A 4x4 numpy array representing the transformation matrix.
    """

    # Open the nifti file
    nifti = nib.load(nifti_path)

    # Get the data from the nifti file
    nifti_data = nifti.get_fdata()

    # Get the voxel spacing from the nifti file
    nifti_voxel_spacing = nifti.header.get_zooms()

    # Get the transformation matrix from the nifti file
    nifti_transformation_matrix = nifti.affine

    return nifti_data, nifti_voxel_spacing, nifti_transformation_matrix


def convert_hdf_file_to_nifti(hdf_file_path, output_path, args):
    """
    This function converts a hdf file to a nifti file.

    Inputs:
    - hdf_file_path: A string representing the path to the hdf file.
    - output_path: A string representing the path to the output nifti file.
    - args: An argparse object representing the command line arguments.

    Returns:
    - None
    """

    # Open the hdf file
    hdf_data = np.array(h5py.File(hdf_file_path, "r")["reconstruction"])[
        ()
    ].squeeze()

    if args.debug:
        # Print information about the hdf file
        print("File", hdf_file_path)
        print("Volume min:", np.min(np.abs(hdf_data)))
        print("Volume max:", np.max(np.abs(hdf_data)))
        print("Volume shape:", hdf_data.shape)

        if args.plot:
            plot_abs_angle_real_imag_from_complex_volume(hdf_data)

    # Create nifti file with new voxel spacing
    nifti_data = nib.Nifti1Image(np.abs(hdf_data), np.eye(4))

    # Set the new voxel spacing if provided
    if args.used_resolution is not None:
        nifti_data.header.set_zooms(args.used_resolution)

    # Save the nifti file
    nib.save(nifti_data, output_path)


def is_hdf_file(file_path):
    """
    This function checks if a file is a hdf file.

    Inputs:
    - file_path: A string representing the path to the file.

    Returns:
    - is_hdf_file: A boolean representing whether the file is a hdf file.
    """
    try:
        with h5py.File(file_path, 'r'):
            return True
    except OSError:
        return False


def save_nifti(volume, transformation_matrix, output_path, filename):
    """
    This function saves a nifti file.

    Inputs:
    - volume: A 3D numpy array representing the volume.
    - transformation_matrix: A 4x4 numpy array representing the transformation matrix.
    - output_path: A string representing the path to the output nifti file.

    Returns:
    - None
    """

    # Create nifti file with new voxel spacing
    nifti_data = nib.Nifti1Image(volume, transformation_matrix)

    # Save the nifti file
    nib.save(nifti_data, os.path.join(output_path, filename))
