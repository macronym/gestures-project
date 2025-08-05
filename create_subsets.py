import os
import random
import argparse
import zipfile
import json
import shutil

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

ANNOTATIONS_PATH = 'D:/Hagrid/annotations/'
NUM_SAMPLES = 100
MAIN_DATASET_PATH = None
OUTPUT_NAME = None
OUTPUT_DIR = None

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
    

def create_subset_dataset(dataset_path: str, num_samples: int = 150) -> None:
    """
    Create a subset of the dataset with a specified number of samples.

    Args:
        dataset_path (str): Path to the original dataset directory.
        output_dir (str): Path to the output directory for the subset dataset.
        num_samples (int): Number of samples to include per gesture. Default is 150.
    """

    # Validation checks
    if not os.path.exists(dataset_path):
        print(f"Dataset path {dataset_path} does not exist.")
        return
    if not os.path.isdir(dataset_path):
        print(f"Dataset path {dataset_path} is not a directory.")
        return

    # Overwrite existing subsets
    if os.path.exists(OUTPUT_DIR):
        print(f"Output directory {OUTPUT_DIR} already exists. It will be overwritten.")
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Get all gesture directories
    gesture_dirs = [
        d for d in os.listdir(dataset_path)
        if os.path.isdir(os.path.join(dataset_path, d)) and not d.startswith('.')
    ]

    # Ensures enough samples for each gesture
    not_enough = [
        d for d in gesture_dirs
        if len(os.listdir(os.path.join(dataset_path, d))) < num_samples
    ]
    if not_enough:
        for d in not_enough:
            count = len(os.listdir(os.path.join(dataset_path, d)))
            print(f"Warning: Not enough samples for gesture '{d}'. Found {count}, expected at least {num_samples}.")
        print("Not all gestures have enough samples. Exiting subset creation.")
        return

    # Create subset
    for gesture_dir in gesture_dirs:
        gesture_path = os.path.join(dataset_path, gesture_dir)
        gesture_samples = os.listdir(gesture_path)
        selected_samples = random.sample(gesture_samples, num_samples)
        output_gesture_dir = os.path.join(OUTPUT_DIR, gesture_dir)
        os.makedirs(output_gesture_dir, exist_ok=True)
        for sample in selected_samples:
            src_file = os.path.join(gesture_path, sample)
            dst_file = os.path.join(output_gesture_dir, sample)
            if os.path.isfile(src_file):
                shutil.copy2(src_file, dst_file)

    create_annotations_from_subset(OUTPUT_DIR, ANNOTATIONS_PATH) # Create annotations for the subset dataset

    print(f"Subset dataset created at {OUTPUT_DIR} with {num_samples} samples per gesture.")

def create_annotations_from_subset(subset_dataset_path: str, annotations_path: str) -> None:
    """
    Create an annotations.json just for the subset dataset.
    """
    # Validation checks
    if not os.path.exists(subset_dataset_path):
        print(f"Subset dataset path {subset_dataset_path} does not exist.")
        return
    if not os.path.exists(annotations_path):
        print(f"Annotations path {annotations_path} does not exist.")
        return
    
    # First, collect all annotations from all splits
    all_annotations = {}
    splits = ['train', 'val', 'test']
    
    for split_name in splits:
        split_path = os.path.join(annotations_path, split_name)
        if not os.path.exists(split_path):
            continue
            
        for gesture_json in os.listdir(split_path):
            if not gesture_json.endswith('.json'):
                continue
            gesture_name = os.path.splitext(gesture_json)[0]
            
            # Load the gesture.json file
            with open(os.path.join(split_path, gesture_json), 'r') as f:
                original_gesture_annotations = json.load(f)
            
            # Merge annotations from all splits for this gesture
            if gesture_name not in all_annotations:
                all_annotations[gesture_name] = {}
            all_annotations[gesture_name].update(original_gesture_annotations)
    
    # Create subset annotations based on actual files in subset
    subset_annotations = {}
    for gesture_dir in os.listdir(subset_dataset_path):
        gesture_path = os.path.join(subset_dataset_path, gesture_dir)
        if not os.path.isdir(gesture_path):
            continue
            
        subset_annotations[gesture_dir] = {}
        
        # Get all files in this gesture directory
        gesture_files = os.listdir(gesture_path)
        
        if gesture_dir in all_annotations:
            for file_name in gesture_files:
                # Try matching with and without extension
                file_key = os.path.splitext(file_name)[0]  # Remove extension
                
                if file_key in all_annotations[gesture_dir]:
                    subset_annotations[gesture_dir][file_key] = all_annotations[gesture_dir][file_key]
                elif file_name in all_annotations[gesture_dir]:
                    subset_annotations[gesture_dir][file_name] = all_annotations[gesture_dir][file_name]
                else:
                    print(f"Warning: No annotation found for {file_name} (or {file_key}) in gesture {gesture_dir}")
                    subset_annotations[gesture_dir][file_key] = {}
        else:
            print(f"Warning: No annotations found for gesture '{gesture_dir}' in original dataset")
            # Create empty annotations for all files
            for file_name in gesture_files:
                file_key = os.path.splitext(file_name)[0]
                subset_annotations[gesture_dir][file_key] = {}

    # Write the new annotations.json file
    subset_annotations_path = os.path.join(subset_dataset_path, "annotations.json")
    with open(subset_annotations_path, 'w') as f:
        json.dump(subset_annotations, f, indent=4)

    print(f"Annotations for subset dataset created at {subset_annotations_path}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="create subset of the dataset")
    parser.add_argument("-d", "--dataset_path", type=str, required=True, help="Path to the dataset")
    parser.add_argument("-o", "--output", type=str, required=False, help="Path to the output directory")
    parser.add_argument("-n", "--num_samples", type=int, default=150, help="Number of samples per gesture")
    parser.add_argument('-x', '--extract', required=False, action='store_true', help="Extract the dataset if it is a zip file")

    args = parser.parse_args()

    # Set global variables
    MAIN_DATASET_PATH = args.dataset_path
    base_name = os.path.splitext(os.path.basename(args.dataset_path))[0]
    OUTPUT_DIR = os.path.join(
        os.path.dirname(args.dataset_path),
        f"{base_name}_subset_{args.num_samples}"
    )

    # Extract first if necessary
    if args.extract:
        extract_dataset(args.dataset_path)
        args.dataset_path = os.path.splitext(args.dataset_path)[0]

    create_subset_dataset(args.dataset_path, args.num_samples)