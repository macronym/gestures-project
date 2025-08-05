import os
import random
import argparse
import zipfile

"""
Author: Justin Brown
Date: 2025-08-04

This script creates a randomly sampled subset of an image dataset.
It was originally written for use on the HAGRID lightweight dataset.

Note: If the dataset is a zip file, use the -x flag to ensure it's extracted first.

CLI Usage:
    python create_subsets.py -d [dataset] -o [output_directory] -n [number_of_samples_per_gesture] [-x]

Example: 
    python create_subsets.py -d D:/test_dir.zip -o D:/output_test_dir -n 2 -x
    python create_subsets.py -d ./hagrid/dataset/ -o ./hagrid/subset_dataset/ -n 150
"""

def extract_dataset(zip_path: str) -> str:
    """
    Extracts the contents of a ZIP archive to the same directory.

    Args:
        zip_path (str): The file path to the ZIP archive.
    """
    if not os.path.exists(zip_path):
        print(f"Zip file {zip_path} does not exist.")
        return
    if not zip_path.endswith('.zip'):
        print(f"File {zip_path} is not a zip file.")
        return
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(os.path.dirname(zip_path))
    print(f"Unzipped dataset to {os.path.dirname(zip_path)}")
    

def create_subset_dataset(dataset_path: str, output_dir: str, num_samples: int = 150) -> None:
    """Create a subset of the dataset with a specified number of samples"""

    # Validation checks
    if not os.path.exists(dataset_path):
        print(f"Dataset path {dataset_path} does not exist.")
        return
    if not os.path.isdir(dataset_path):
        print(f"Dataset path {dataset_path} is not a directory.")
        return
    os.makedirs(output_dir, exist_ok=True)

    # Ensures there are enough samples in each gesture directory
    enough_samples = True
    for gesture_dir in os.listdir(dataset_path):
        gesture_path = os.path.join(dataset_path, gesture_dir)
        if not os.path.isdir(gesture_path) or gesture_dir.startswith('.'):
            continue  # Skip files, only process directories
        gesture_samples = os.listdir(gesture_path)
        count = len(gesture_samples)
        if count < num_samples:
            print(f"Warning: Not enough samples for gesture '{gesture_dir}'. Found {count}, expected at least {num_samples}.")
            enough_samples = False
    if not enough_samples:
        print("Not all gestures have enough samples. Exiting subset creation.")
        return

    # Create the randomly sampled subset
    for gesture_dir in os.listdir(dataset_path):
        gesture_path = os.path.join(dataset_path, gesture_dir)
        if not os.path.isdir(gesture_path) or gesture_dir.startswith('.'):
            continue
        gesture_samples = os.listdir(gesture_path)
        selected_samples = random.sample(gesture_samples, num_samples)
        output_gesture_dir = os.path.join(output_dir, gesture_dir)
        os.makedirs(output_gesture_dir, exist_ok=True)
        for sample in selected_samples:
            src_file = os.path.join(gesture_path, sample)
            dst_file = os.path.join(output_gesture_dir, sample)
            if os.path.isfile(src_file):
                with open(src_file, 'rb') as fsrc, open(dst_file, 'wb') as fdst:
                    fdst.write(fsrc.read())

    print(f"Subset dataset created at {output_dir} with {num_samples} samples per gesture.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="create subset of the dataset")
    parser.add_argument("-d", "--dataset", type=str, required=True, help="Path to the dataset")
    parser.add_argument("-o", "--output", type=str, required=True, help="Path to the output directory")
    parser.add_argument("-n", "--num_samples", type=int, default=150, help="Number of samples per gesture")
    parser.add_argument('-x', '--extract', required=False, action='store_true', help="Extract the dataset if it is a zip file")

    args = parser.parse_args()

    # Extract first if necessary
    if args.extract:
        extract_dataset(args.dataset)
        args.dataset = os.path.splitext(args.dataset)[0]

    create_subset_dataset(args.dataset, args.output, args.num_samples)