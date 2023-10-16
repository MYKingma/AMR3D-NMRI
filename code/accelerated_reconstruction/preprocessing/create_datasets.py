# Script to evaluate the reconstruction performance of a model
#
# Author: D. Karkalousos
# Optimalized for project by M.Y. Kingma


import os
import argparse
import h5py
import numpy as np
import time
from pathlib import Path
from tqdm import tqdm

from utils.plot import plot_abs_angle_real_imag_from_complex_volume
from utils.preprocessing import readcfl


def reshape_single_shot_array(array):
    if len(array.shape) == 4:
        return array[:, :, :, 0]
    return array


def main(args):
    init_start = time.perf_counter()

    # Define filepaths
    out_dir = Path(args.main_path) / Path(args.out_dir) / Path(args.data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data_path = Path(args.main_path) / Path("proc")
    smap_path = str(data_path / Path('smap') /
                    Path(args.data_dir) / Path('smap'))
    smap_sag_path = str(data_path / Path('smap') /
                        Path(args.data_dir) / Path('smap_sag'))

    # Get k-space path
    if args.single_shot:
        kspace_path = data_path / Path("shot_kspace") / Path(args.data_dir)
    else:
        kspace_path = data_path / Path("kspace") / Path(args.data_dir)

    # Process sensemap
    sensemap = readcfl(smap_path)

    # Normalize sensemap
    if args.normalize:
        sensemap = sensemap / np.max(np.abs(sensemap))

    if args.sensemap_shift_axes:
        axes = tuple(args.sensemap_shift_axes)
        sensemap = np.fft.ifftn(np.fft.ifftshift(np.fft.fftn(
            sensemap, axes=(0, 1)), axes=axes), axes=(0, 1))

    # Process sagittal sensemap
    sensemap_sag = np.array([])
    if os.path.exists(smap_sag_path + ".hdr"):
        sensemap_sag = readcfl(smap_sag_path)

        if args.sensemap_shift_axes:
            axes = tuple(args.sensemap_shift_axes)
            sensemap_sag = np.fft.ifftn(np.fft.ifftshift(np.fft.fftn(
                sensemap_sag, axes=(0, 1)), axes=axes), axes=(0, 1))

        # Normalize sagittal sensemap
        if args.normalize:
            sensemap_sag = sensemap_sag / np.max(np.abs(sensemap_sag))

    # List k-space files
    files = [str(f).split('.cfl')
             for f in list(kspace_path.iterdir()) if ".cfl" in f.name]

    for file in tqdm(files):
        if args.single_shot:
            fname = file[0].split('/')[-1].split('wip_')[-1] + "_single_shot"
        else:
            fname = file[0].split('/')[-1].split('wip_')[-1]

        used_sensemap = sensemap_sag if "sag" in fname else sensemap

        # Process k-space
        kspace = readcfl(file[0])

        if kspace.shape != used_sensemap.shape:
            print("Sensemap and kspace have different shapes")

            if len(kspace.shape) > 4:
                # Temp solution for handling multiple shot kspace volume
                kspace = kspace[:, :, :, :, 1]

            kspace = np.transpose(kspace, (1, 2, 0, 3))

        # If compressed sensing, define mask and acceleration
        acceleration = 1
        mask = np.array([])
        if "cs" in fname:
            mask = np.sum(np.abs(kspace), (-2, -1)) > 0
            acceleration = np.prod(mask.shape) / np.sum(mask)

        # Define imspace
        if args.kspace_shift_axes and args.imspace_shift_axes:
            kspace_axes = list(args.kspace_shift_axes)
            imspace_axes = list(args.imspace_shift_axes)
            imspace = np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(
                kspace, axes=kspace_axes), axes=(0, 1)), axes=imspace_axes)
        elif args.kspace_shift_axes:
            kspace_axes = list(args.kspace_shift_axes)
            imspace = np.fft.ifftn(np.fft.ifftshift(
                kspace, axes=kspace_axes), axes=(0, 1))
        elif args.imspace_shift_axes:
            imspace_axes = list(args.imspace_shift_axes)
            imspace = np.fft.ifftshift(np.fft.ifftn(
                kspace, axes=(0, 1)), axes=imspace_axes)
        else:
            imspace = np.fft.ifftn(kspace, axes=(0, 1))

        # Normalize imspace
        if args.normalize:
            imspace = imspace / np.max(np.abs(imspace))

        # Define target
        target = np.sum(imspace * used_sensemap.conj(), -1)

        # Normalize target
        if args.normalize:
            target = target / np.max(target)

        if args.debug:
            # Print acceleration
            print(f"Acceleration: {acceleration}")

            slice_no = 10
            if "sag" in fname:
                slice_no = 40

            # Reshape is single shot data array (for plot)
            used_sensemap_plot_data = reshape_single_shot_array(used_sensemap)
            imspace_plot_data = reshape_single_shot_array(imspace)
            target_plot_data = reshape_single_shot_array(target)

            # Plot sensemap, imspace, target, and print shapes
            print(
                f"Sensemap shape: {used_sensemap.shape}, min: {np.min(np.abs(used_sensemap))}, max: {np.max(np.abs(used_sensemap))}")
            plot_abs_angle_real_imag_from_complex_volume(
                used_sensemap_plot_data, axis=2, slice_no=slice_no)
            print(
                f"Imspace shape: {imspace.shape}, min: {np.min(np.abs(imspace))}, max: {np.max(np.abs(imspace))}")
            plot_abs_angle_real_imag_from_complex_volume(
                imspace_plot_data, axis=2, slice_no=slice_no)
            print(
                f"Target shape: {target.shape}, min: {np.min(np.abs(target))}, max: {np.max(np.abs(target))}")
            plot_abs_angle_real_imag_from_complex_volume(
                target_plot_data, axis=2, slice_no=slice_no)

        # Save data
        hf = h5py.File(Path(str(out_dir) + "/" + fname), "w")

        hf.create_dataset("kspace", data=np.fft.fftn(
            np.transpose(imspace, (2, 3, 0, 1)), axes=(-2, -1)))
        hf.create_dataset("sensitivity_map", data=np.transpose(
            used_sensemap, (2, 3, 0, 1)))
        hf.create_dataset("target", data=np.transpose(target, (2, 0, 1)))

        if "cs" in fname:
            hf.create_dataset("mask", data=mask)

        hf.close()

    print("Finished! It took", time.perf_counter() - init_start, "s \n")


if __name__ == "__main__":
    # Create argument parser object
    parser = argparse.ArgumentParser(
        description="Preprocess raw data for training and testing.",)

    # Add arguments
    parser.add_argument("--main-path", type=str,
                        help="Path of parent data directories.")
    parser.add_argument("--data-dir", type=str,
                        help="Directory name of the data.")
    parser.add_argument("--out-dir", type=str,
                        help="Directory name of the output.")
    parser.add_argument("--normalize", action="store_true",
                        help="Toggle to turn on normalization.")
    parser.add_argument("--sensemap-shift-axes", nargs="+",
                        type=int, help="Shift sensemap axes.")
    parser.add_argument("--kspace-shift-axes", nargs="+",
                        type=int, help="Shift kspace axes.")
    parser.add_argument("--imspace-shift-axes", nargs="+",
                        type=int, help="Shift imspace axes.")
    parser.add_argument("--single-shot", action="store_true",
                        help="Toggle to turn on single shot mode.")
    parser.add_argument("--debug", action="store_true",
                        help="Toggle to turn on debug mode.")

    # Parse the arguments
    args = parser.parse_args()

    # Print the arguments
    print("Running preprocess script with the following arguments:")
    for arg in vars(args):
        print(f"{arg}:", getattr(args, arg))
    main(args)
