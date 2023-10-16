# Script to sequentially reconstruct a volume from a stack of slices, used for evaluation of SVRTK using the HCP dataset
#
# Author: Maurice Kingma

import subprocess
import argparse
import os


def main(args):

    # Loop over directories in data path
    for directory in os.listdir(args.data_path):

        # Step into directory
        os.chdir(os.path.join(args.data_path, directory))

        # Directory name is subject name
        subject = directory

        # Get the files in the directory
        files = os.listdir(os.path.join(args.data_path, directory))

        # Sort the files alphabetically reverse
        files.sort(reverse=True)

        # Define the MIRTK command
        mirtk_command = f"nice -n 10 mirtk reconstruct {subject}_output.nii.gz 2 {files[1]} {files[0]} -resolution {args.resolution} -template {files[1]} -no_registration -iterations {args.iterations}"

        # If mask is true, add mask to command
        if args.mask:
            mirtk_command += f" -mask {files[2]}"

        # If debug is true, add debug flag to command
        if args.debug:
            mirtk_command += " -debug"

        # Run the MIRTK command
        try:
            subprocess.run(mirtk_command, shell=True, check=True)

        except subprocess.CalledProcessError as e:
            print(f"Error processing {subject}: {e}")

        # Debug files are not needed, only keep the output, input and mask files
        for file in os.listdir(os.path.join(args.data_path, directory)):
            if file not in [f"{subject}_output.nii.gz", files[0], files[1], files[2], "init.nii.gz"]:

                # Check if file is a directory
                if not os.path.isdir(os.path.join(args.data_path, directory, file)):

                    # If not, remove the file
                    os.remove(os.path.join(args.data_path, directory, file))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script to sequentially reconstruct a volume from a stack of slices, used for evaluation using the HCP dataset")
    parser.add_argument("-d", "--data-path", type=str, required=True,
                        help="Path to the data directory")
    parser.add_argument("-r", "--resolution", type=float,
                        help="Voxel spacing of the slices")
    parser.add_argument("-i", "--iterations", type=int,
                        help="Number of iterations for the reconstruction")
    parser.add_argument("-m", "--mask", action="store_true",
                        help="Whether to use a brain mask for the reconstruction")
    parser.add_argument("-db", "--debug", action="store_true",
                        help="Run in debug mode")
    args = parser.parse_args()

    # Print the arguments
    print("Running SVR reconstruct with the following arguments:")
    for arg in vars(args):
        print(f"{arg}:", getattr(args, arg))
    main(args)
