# Script to generate GIFs of the slice-to-volume reconstruction and the baseline reconstruction
#
# Author: Maurice Kingma

import imageio
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from utils.nifti import load_nifti_file
from utils.preprocessing import normalize_volume

svr_output_path = "/path/to/slice-to-volume/recon"
manual_output_path = "/path/to/baseline/recon"

# Open the nifti
svr_output, _, _ = load_nifti_file(svr_output_path)
manual_output, _, _ = load_nifti_file(manual_output_path)

# Replace nan with 0
svr_output = np.nan_to_num(svr_output)
manual_output = np.nan_to_num(manual_output)

# Replace -1 values with 0
svr_output[svr_output < 0] = 0
manual_output[manual_output < 0] = 0

# Normalize
svr_output = normalize_volume(svr_output)
manual_output = normalize_volume(manual_output)


def generate_slice_gif(volume, axis, filepath):
    # Determine the number of slices along the given axis
    num_slices = volume.shape[axis] - 3

    print(num_slices)

    # Loop over all slices along the given axis
    images = []
    for i in range(num_slices):
        # Extract the slice along the given axis
        if axis == 0:
            slice_data = volume[i, :, :]
        elif axis == 1:
            slice_data = volume[:, i, :]
        elif axis == 2:
            slice_data = volume[:, :, i]

        images.append(slice_data)

    pil_images = [Image.fromarray(
        np.uint8(image * 255), mode='L') for image in images]

    # Save the images as a GIF
    imageio.mimsave(filepath, pil_images, format='GIF', duration=40, loop=10)


# Generate the GIFs
generate_slice_gif(
    svr_output, 2, 'output_slice-to-volume.gif')
generate_slice_gif(
    manual_output, 2, 'output_baseline.gif')
