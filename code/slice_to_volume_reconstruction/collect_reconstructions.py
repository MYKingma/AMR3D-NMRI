# Script for collecting reconstructions from data directory and moving them to output directory
#
# Author: M.Y. Kingma


import os
import argparse


def main(args):
    # Loop over directories in data directory
    for directory in os.listdir(args.data_path):

        # Check if directory is a directory
        if os.path.isdir(os.path.join(args.data_path, directory)):

            # Get files containing 'output' and 'init' in name
            for file in os.listdir(os.path.join(args.data_path, directory)):
                if 'output' in file:
                    output_file = file

                    # Rename and move file
                    os.rename(os.path.join(args.data_path, directory, output_file),
                              os.path.join(args.output_path, directory + '.nii.gz'))

                if 'init' in file:
                    init_file = file

                    # Rename and move file
                    os.rename(os.path.join(args.data_path, directory, init_file),
                              os.path.join(args.output_path, directory + '_init.nii.gz'))


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Rename and move files in data directory")
    parser.add_argument("-d", "--data_path", type=str,
                        help="path to data directory")
    parser.add_argument("-o", "--output_path", type=str,
                        help="path to output directory")
    args = parser.parse_args()

    # Print arguments
    print("Collecting reconstructions with the following arguments:")
    for arg in vars(args):
        print(arg, getattr(args, arg))

    # Run main function
    main(args)
