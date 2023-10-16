# Script to evaluate the reconstruction performance of a model
#
# Author: D. Karkalousos
# Optimalized for project by M.Y. Kingma

import numpy as np
from pathlib import Path
import h5py
import argparse

from utils.mask import compute_brain_mask
from utils.metrics import Metrics, METRIC_FUNCS


def main(args):
    # if json file
    if args.targets_path.endswith(".json"):
        import json

        with open(args.targets_path, "r") as f:
            targets = json.load(f)
        targets = [Path(target) for target in targets]
    else:
        targets = list(Path(args.targets_path).iterdir())

    scores = Metrics(METRIC_FUNCS)
    ssims = []
    for target in targets:
        reconstruction = np.array(
            h5py.File(Path(args.reconstructions_path) / str(target).split("/")[-1], "r"))
        if "reconstruction_sense" in h5py.File(target, "r").keys():
            target = np.array(h5py.File(target, "r")[
                "reconstruction_sense"])[()].squeeze()
        elif "reconstruction_rss" in h5py.File(target, "r").keys():
            target = np.array(h5py.File(target, "r")[
                              "reconstruction_rss"])[()].squeeze()
        elif "reconstruction" in h5py.File(target, "r").keys():
            target = np.array(h5py.File(target, "r")[
                              "reconstruction"])[()].squeeze()
        else:
            target = np.array(h5py.File(target, "r")["target"])[()].squeeze()

        if args.eval_per_slice:
            for sl in range(target.shape[0]):

                # Normalise slice
                target_slice = target[sl] / np.max(np.abs(target[sl]))
                reconstruction_slice = reconstruction[sl] / \
                    np.max(np.abs(reconstruction[sl]))

                # Take absolute value
                target_slice = np.abs(target_slice)
                reconstruction_slice = np.abs(reconstruction_slice)

                # Clip values to [0, 1]
                target_slice = np.clip(target_slice, 0, 1)
                reconstruction_slice = np.clip(reconstruction_slice, 0, 1)

                if args.mask:
                    # Compute brain mask
                    mask = compute_brain_mask(
                        target_slice, threshold=args.threshold)

                    # Replace values outside the mask with NaN
                    target_slice = np.where(mask == False, 0, target_slice)
                    reconstruction_slice = np.where(
                        mask == False, 0, reconstruction_slice)

                # Calculate metrics
                scores.push(target_slice, reconstruction_slice)

        else:
            # Normalise
            target = target / np.max(np.abs(target))
            reconstruction = reconstruction / np.max(np.abs(reconstruction))

            # Take absolute value
            target = np.abs(target)
            reconstruction = np.abs(reconstruction)

            # Clip values to [0, 1]
            target = np.clip(target, 0, 1)
            reconstruction = np.clip(reconstruction, 0, 1)

            if args.mask:
                # Compute brain mask
                mask = compute_brain_mask(target, threshold=args.threshold)

                # Replace values outside the mask with NaN
                target = np.where(mask == False, 0, target)
                reconstruction = np.where(mask == False, 0, reconstruction)

            # Calculate metrics
            scores.push(target, reconstruction)

    # Print the scores
    print(scores.__repr__()[:-1])
    print(scores.__csv__())


if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser()

    # Add arguments to the parser
    parser.add_argument("-tp", "--targets-path", type=str,
                        required=True, help="Path to targets directory")
    parser.add_argument("-rp", "--reconstructions-path", type=str,
                        required=True, help="Path to reconstructions directory")
    parser.add_argument("-eps", "--eval-per-slice", action="store_true",
                        default=False, help="Evaluate metrics per slice")
    parser.add_argument("-m", "--mask", action="store_true", default=False,
                        help="Only calculate metrics for masked region")
    parser.add_argument("-t", "--threshold", type=float, default=0.1,
                        help="Threshold value for segmentation and mask creation. If not provided, the threshold will be calculated using the Otsu method. Default: 0.1")

    # Parse the arguments
    args = parser.parse_args()

    # Print the arguments
    print("Running evaluation script with the following arguments:")
    for arg in vars(args):
        print(f"\t{arg}: {getattr(args, arg)}")

    # Run the main function
    main(args)
