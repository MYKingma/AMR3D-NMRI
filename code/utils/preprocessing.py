# Helper functions for preprocessing data
#
# Author: M.Y. Kingma


import numpy as np
from skimage.measure import block_reduce


def normalize_volume(volume, normalize_per_slice=False, axis=0):
    """
    This function normalizes a 3D volume to be between 0 and 1.

    Inputs:
    - volume: A 3D numpy array representing the volume data
    - normalize_per_slice: A boolean representing whether to normalize each slice of the volume separately
    - axis: An integer representing the axis to normalize, if normalize_per_slice is True

    Returns:
    - volume: A 3D numpy array representing the normalized volume data
    """

    # Check and replace nan values
    volume = replace_nan(volume)

    if normalize_per_slice:
        # Normalize each slice of the volume separately
        for i in range(volume.shape[axis]):
            if axis == 0:
                volume[i, :, :] = np.min(np.abs(volume[i, :, :]))
            elif axis == 1:
                volume[:, i, :] = np.min(np.abs(volume[:, i, :]))
            elif axis == 2:
                volume[:, :, i] = np.min(np.abs(volume[:, :, i]))

    else:
        # Normalize the entire volume
        volume = volume - np.min(np.abs(volume))
        if np.max(np.abs(volume)) != 0:
            volume = volume / np.max(np.abs(volume))

    return volume


def replace_nan(volume):
    """
    This function replaces all nan values in a 3D volume with the average of neighboring values.

    Inputs:
    - volume: A 3D numpy array representing the volume data

    Returns:
    - volume: A 3D numpy array representing the volume data
    """

    # Get all nan indices
    nan_indices = np.isnan(volume)

    # Replace nan values with average of neighboring values
    for i, j, k in np.argwhere(nan_indices):
        # Get indices of neighboring pixels
        i_min = max(i-1, 0)
        i_max = min(i+2, volume.shape[0])
        j_min = max(j-1, 0)
        j_max = min(j+2, volume.shape[1])
        k_min = max(k-1, 0)
        k_max = min(k+2, volume.shape[2])

        # Compute average of neighboring values
        neighbors = volume[i_min:i_max, j_min:j_max, k_min:k_max]
        neighbors = neighbors[~np.isnan(neighbors)]
        if neighbors.size > 0:
            avg = np.mean(neighbors)
            volume[i, j, k] = avg

    return volume


def flip_transformation_matrix(transformation_matrix, filename, args):
    """
    This function flips the transformation matrix according to the flip arguments.

    Inputs:
    - transformation_matrix: A 4x4 numpy array representing the transformation matrix
    - filename: A string representing the filename of the nifti file
    - args: An argparse object representing the command line arguments

    Returns:
    - transformation_matrix: A 4x4 numpy array representing the flipped transformation matrix
    """

    # Get the flip arguments
    flip_arguments = args.flip if "sag" not in filename else args.flip_sag

    if not flip_arguments:
        return transformation_matrix

    # Flip the transformation matrix
    if bool(flip_arguments[0]):
        transformation_matrix[0, 0] = -1 * transformation_matrix[0, 0]
    if bool(flip_arguments[1]):
        transformation_matrix[1, 1] = -1 * transformation_matrix[1, 1]
    if bool(flip_arguments[2]):
        transformation_matrix[2, 2] = -1 * transformation_matrix[2, 2]

    if args.debug:
        print("Flipped transformation matrix:\n", transformation_matrix)

    return transformation_matrix


def transpose_transformation_matrix(transformation_matrix, filename, args):
    """
    This function transposes the transformation matrix according to the transpose arguments.

    Inputs:
    - transformation_matrix: A 4x4 numpy array representing the transformation matrix
    - filename: A string representing the filename of the nifti file
    - args: An argparse object representing the command line arguments

    Returns:
    - transformation_matrix: A 4x4 numpy array representing the transposed transformation matrix
    """

    # Get the transpose arguments
    transpose_args = args.transpose if "sag" not in filename else args.transpose_sag

    if not transpose_args:
        return transformation_matrix

    # Transpose the transformation matrix
    transposed_transformation_matrix = np.eye(4)
    transposed_transformation_matrix[0,
                                     :] = transformation_matrix[transpose_args[0], :]
    transposed_transformation_matrix[1,
                                     :] = transformation_matrix[transpose_args[1], :]
    transposed_transformation_matrix[2,
                                     :] = transformation_matrix[transpose_args[2], :]

    if args.debug:
        print("Transposed transformation matrix:\n",
              transposed_transformation_matrix)

    return transposed_transformation_matrix


def create_transformation_matrix_nifti(volume_shape, volume_spacing, filename, args):
    """
    This function creates a transformation matrix for a nifti file.

    Inputs:
    - volume_shape: A tuple representing the shape of the volume
    - volume_spacing: A tuple representing the voxel spacing of the volume
    - filename: A string representing the filename of the nifti file
    - args: An argparse object representing the command line arguments

    Returns:
    - transformation_matrix: A 4x4 numpy array representing the transformation matrix
    """

    # Create a transformation matrix to transform the nifti volume to the correct voxel spacing
    transformation_matrix = np.eye(4)

    # Set the diagonal values to the voxel spacing
    transformation_matrix[0, 0] = volume_spacing[0]
    transformation_matrix[1, 1] = volume_spacing[1]
    transformation_matrix[2, 2] = volume_spacing[2]

    if args.offset or args.offset_sag:
        used_offset = args.offset if "sag" not in filename else args.offset_sag

        if used_offset:
            transformation_matrix[0, 3] = used_offset[0]
            transformation_matrix[1, 3] = used_offset[1]
            transformation_matrix[2, 3] = used_offset[2]

    if args.debug:
        print("Volume shape:", volume_shape)
        print("Transformation matrix:\n", transformation_matrix)

    if args.flip or args.flip_sag:
        transformation_matrix = flip_transformation_matrix(
            transformation_matrix, filename, args)

    if args.transpose or args.transpose_sag:
        transformation_matrix = transpose_transformation_matrix(
            transformation_matrix, filename, args)

    return transformation_matrix


def downsample_volume(volume, transformation_matrix, downsample_axis, downsample_factor=2):
    """
    This function downsamples a 3D volume.

    Inputs:
    - volume: A 3D numpy array representing the volume data
    - spacing: A tuple representing the voxel spacing of the volume
    - axis: An integer representing the axis to downsample
    - args: An argparse object representing the command line arguments

    Returns:
    - downsampled_volume: A 3D numpy array representing the downsampled volume data
    - new_voxel_spacing: A tuple representing the new voxel spacing of the volume
    """

    # Define the block size
    block_size = [1, 1, 1]
    block_size[downsample_axis] = downsample_factor
    block_size = tuple(block_size)

    # Set new transformation matrix
    downsampled_transformation_matrix = np.copy(transformation_matrix)
    downsampled_transformation_matrix[downsample_axis,
                                      downsample_axis] = downsample_factor * transformation_matrix[downsample_axis, downsample_axis]

    # Apply the block_reduce function with a mean function to the volume
    downsampled_volume = block_reduce(
        volume, block_size=block_size, func=np.mean)  # type: ignore

    return downsampled_volume, downsampled_transformation_matrix


def readcfl(name):
    """
    Read in a .cfl file.

    Inputs:
    - name: Path to .cfl file

    Returns:
    numpy.array: Data stored in .cfl file
    """

    h = open(name + ".hdr", "r")
    h.readline()  # skip
    l = h.readline()
    h.close()
    dims = [int(i) for i in l.split()]

    # remove singleton dimensions from the end
    n = int(np.prod(dims))
    dims_prod = np.cumprod(dims)
    dims = dims[: np.searchsorted(dims_prod, n) + 1]

    # load data and reshape into dims
    d = open(name + ".cfl", "r")
    a = np.fromfile(d, dtype=np.complex64, count=n)
    d.close()
    a = a.reshape(dims, order="F")  # column-major
    return a


def get_slice_from_axis(volume, axis=0, slice_no=0) -> np.ndarray:
    """
    This function returns a slice from a 3D volume along a given axis.

    Inputs:
    - volume: A 3D numpy array representing the volume data
    - axis: An integer representing the axis to slice along
    - slice_no: An integer representing the slice number to return

    Returns:
    - slice: A 2D numpy array representing the slice
    """

    # Get the slice
    if axis == 0:
        return volume[slice_no, :, :]
    elif axis == 1:
        return volume[:, slice_no, :]
    return volume[:, :, slice_no]


def set_slice_from_axis(volume, slice, axis, slice_no):
    """
    This function sets a slice in a 3D volume along a given axis.

    Inputs:
    - volume: A 3D numpy array representing the volume data
    - slice: A 2D numpy array representing the slice
    - axis: An integer representing the axis to slice along
    - slice_no: An integer representing the slice number to set

    Returns:
    - volume: A 3D numpy array representing the volume data
    """

    # Set the slice
    if axis == 0:
        volume[slice_no, :, :] = slice
    elif axis == 1:
        volume[:, slice_no, :] = slice
    elif axis == 2:
        volume[:, :, slice_no] = slice

    return volume
