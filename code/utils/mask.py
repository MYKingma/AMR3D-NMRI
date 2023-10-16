# Helper functions for computing brain masks
#
# Author: Maurice Kingma

from skimage import morphology
import numpy as np
from skimage.morphology import remove_small_objects
from skimage.filters import threshold_otsu
from scipy.ndimage import binary_fill_holes
from utils.preprocessing import get_slice_from_axis, set_slice_from_axis
import matplotlib.pyplot as plt


def compute_brain_mask(volume, threshold=None, min_size=100, mask_per_slice=False, axis=0):
    """
    This function computes a brain mask for a given 3D volume.

    Inputs:
    - volume: A 3D numpy array representing the volume
    - threshold: A float representing the threshold value to use for the mask
    - min_size: An integer representing the minimum size of connected components to keep

    Returns:
    - mask: A 2D numpy array representing the brain mask
    """

    if mask_per_slice:
        # Loop over each slice of the volume
        for i in range(volume.shape[axis]):

            # Get the slice from the volume
            slice = get_slice_from_axis(volume, axis, i)

            # Compute threshold value using Otsu's algorithm if not provided
            slice_threshold = threshold_otsu(
                slice) if threshold is None else threshold
            if threshold is None:
                print("Calculated Otsu threshold:", slice_threshold)

            # Apply threshold to the slice
            slice[slice > slice_threshold] = 1
            slice[slice <= slice_threshold] = 0

            # Fill any holes inside the brain
            slice = binary_fill_holes(slice)

            # Remove any small disconnected regions
            slice = remove_small_objects(slice, min_size=min_size)

            # Smooth the mask edges
            slice = morphology.binary_erosion(
                slice, morphology.disk(2))

            # Set the slice back into the volume
            volume = set_slice_from_axis(volume, slice, axis, i)

        # Fill any holes inside the brain in all dimensions, and remove small disconnected regions
        for i in range(volume.shape[0]):
            volume[i, :, :] = binary_fill_holes(volume[i, :, :])
        for i in range(volume.shape[1]):
            volume[:, i, :] = binary_fill_holes(volume[:, i, :])
        for i in range(volume.shape[2]):
            volume[:, :, i] = binary_fill_holes(volume[:, :, i])

        return volume

    else:
        # Compute threshold value using Otsu's algorithm if not provided
        if threshold is None:
            threshold = threshold_otsu(volume)
            print("Calculated Otsu threshold:", threshold)

        # Apply threshold to the volume
        mask = np.zeros_like(volume, dtype=np.int64)
        mask[volume > threshold] = 1

        # Fill any holes inside the brain in all dimensions, and remove small disconnected regions
        for i in range(mask.shape[0]):
            mask[i, :, :] = binary_fill_holes(mask[i, :, :])
            mask[i, :, :] = remove_small_objects(
                mask[i, :, :], min_size=min_size)
        for i in range(mask.shape[1]):
            mask[:, i, :] = binary_fill_holes(mask[:, i, :])
            mask[:, i, :] = remove_small_objects(
                mask[:, i, :], min_size=min_size)
        for i in range(mask.shape[2]):
            mask[:, :, i] = binary_fill_holes(mask[:, :, i])
            mask[:, :, i] = remove_small_objects(
                mask[:, :, i], min_size=min_size)

        # Remove any small disconnected regions
        mask = mask > 0
        mask = remove_small_objects(mask, min_size=min_size)

        # Smooth the mask edges
        for i in range(mask.shape[0]):
            mask[i, :, :] = morphology.binary_erosion(
                mask[i, :, :], morphology.disk(2))

        return mask
