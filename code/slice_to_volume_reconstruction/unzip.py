# Script for unzipping the target files from the HCP zipfiles into an output directory
#
# Author: M.Y. Kingma

import argparse
import zipfile
import os


def main(args):
    # List all the zipped files in the directory
    zipped_files = [f for f in os.listdir(
        args.data_path) if f.endswith(".zip")]

    for file_name in zipped_files:
        with zipfile.ZipFile(os.path.join(args.data_path, file_name), "r") as zip_ref:
            with zip_ref.open(f"{file_name.split('_')[0]}/T1w/T1w_acpc_dc.nii.gz") as file_in_zip:
                with open(os.path.join(args.output_path, "T1w_acpc_dc.nii.gz"), "wb") as output_file:
                    output_file.write(file_in_zip.read())

        # Rename the extracted file to match the zip file name
        extracted_file_path = os.path.join(
            args.output_path, "T1w_acpc_dc.nii.gz")
        new_file_path = os.path.join(
            args.output_path, f"{file_name.split('_')[0]}_T1w_acpc_dc.nii.gz")
        os.rename(extracted_file_path, new_file_path)


if __name__ == "__main__":
    # Create argument parser object
    parser = argparse.ArgumentParser(
        description="Downsample HCP data")

    # Add arguments
    parser.add_argument("-d", "--data-path", type=str, required=True,
                        help="Path to directory containing zipped high-resolution NIfTI images")
    parser.add_argument("-o", "--output-path", type=str, required=True,
                        help="Path to output directory for unzipped NIfTI images")

    # Parse the arguments
    args = parser.parse_args()

    # Print the arguments
    print("Running SVR preparation with the following arguments:")
    for arg in vars(args):
        print(f"{arg}:", getattr(args, arg))
    main(args)
