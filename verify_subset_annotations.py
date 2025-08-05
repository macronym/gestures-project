import os
import json
import argparse

"""
Author: Justin Brown
Date: 2025-08-05

Usage:
    python verify_subset_annotations.py -s [subset_dataset_path] -a [annotations_path]
Example:
    python verify_subset_annotations.py -s D:/hagrid/subset_dataset_10/
    python verify_subset_annotations.py -s D:/hagrid/subset_dataset_10/ -a D:/hagrid/subset_dataset_10/annotations.json
"""

def verify_subset_annotations(subset_dataset_path: str, annotations_path: str) -> None:
    """
    Verify that every gesture file has a corresponding annotation in annotations.json.
    """
    if not os.path.exists(subset_dataset_path):
        print(f"Subset dataset path {subset_dataset_path} does not exist.")
        return

    if not os.path.exists(annotations_path):
        print(f"Annotations path {annotations_path} does not exist.")
        return

    # Load the annotations
    with open(annotations_path, 'r') as f:
        annotations = json.load(f)

    # Check each gesture directory in the subset dataset
    for gesture_dir in os.listdir(subset_dataset_path):
        gesture_path = os.path.join(subset_dataset_path, gesture_dir)
        if not os.path.isdir(gesture_path) or gesture_dir.startswith('.'):
            continue

        # Check if the gesture has annotations
        if gesture_dir not in annotations:
            print(f"Warning: No annotations found for gesture '{gesture_dir}' in the subset dataset.")
            continue

        # Use a set for fast lookup
        annotation_keys = set(annotations[gesture_dir].keys())

        # Verify that all files in the gesture directory are accounted for in the annotations
        for sample in os.listdir(gesture_path):
            sample_path = os.path.join(gesture_path, sample)
            if not os.path.isfile(sample_path):
                continue
            sample_key = os.path.splitext(sample)[0]
            if sample_key not in annotation_keys:
                print(f"Warning: No annotation found for '{sample}' in gesture '{gesture_dir}'.")

    print(f"Subset annotations verification completed on {subset_dataset_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify subset annotations.")
    parser.add_argument('-s', '--subset_dataset_path', type=str, required=True, help="Path to the subset dataset.")
    parser.add_argument('-a', '--annotations_path', type=str, required=False, help="Path to the annotations json.")

    args = parser.parse_args()

    if not args.annotations_path:
        args.annotations_path = os.path.join(args.subset_dataset_path, "annotations.json")

    verify_subset_annotations(args.subset_dataset_path, args.annotations_path)