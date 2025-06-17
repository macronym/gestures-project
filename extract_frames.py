import cv2
import os
import argparse
import csv

"""Example usage: 
python extract_frames.py --video point_down.mp4 --output ./output/point_down --label point_down --frame_rate 5
"""

def extract_frames(video_path, output_dir, every_frame=False, frame_rate=1.0):
    """
    Extracts frames from a video file and saves them as images in the specified output directory.
    Parameters:
        video_path (str): Path to the input video file.
        output_dir (str): Directory where extracted frames will be saved.
        every_frame (bool): If True, extracts every frame; if False, extracts frames at the specified frame_rate.
        frame_rate (float): Number of frames per second to extract if every_frame is False.
    Returns:
        int: The number of frames saved.
    """
    # Attempts to load the video file
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    # Get rotation metadata
    rotation = int(capture.get(cv2.CAP_PROP_ORIENTATION_META))
    
    os.makedirs(output_dir, exist_ok=True)
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    interval = 1 if every_frame else int(max(1, fps / frame_rate)) # Increment for frame extraction
    count = 0
    saved = 0
    
    # Loop to extract frames
    while True:
        ret, frame = capture.read()
        if not ret:
            break
        if count % interval == 0:
            # Apply rotation if needed
            if rotation == 90:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            elif rotation == 270:
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            elif rotation == 180:
                frame = cv2.rotate(frame, cv2.ROTATE_180)
            
            # Save the frame as an image file
            fname = f"frame_{saved:05d}.jpg"
            cv2.imwrite(os.path.join(output_dir, fname), frame)
            saved += 1
        count += 1

    capture.release()
    return saved

def generate_labels(output_dir, label):
    """ Generates a CSV file with labels for each extracted frame.
    Parameters:
        output_dir (str): Directory where extracted frames are saved.
        label (str): Label to assign to all extracted frames.
    """
    files = sorted([f for f in os.listdir(output_dir) if f.startswith('frame_')])
    with open(os.path.join(output_dir, 'labels.csv'), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['filename', 'label'])
        for f in files:
            writer.writerow([f, label])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract frames from a video and generate labels.')
    parser.add_argument('--video', required=True, help='Path to input video file (e.g., input.mp4)')
    parser.add_argument('--output', required=True, help='Directory to save extracted frames and labels')
    parser.add_argument('--label', required=False, help='Label to assign to all extracted frames (e.g., "wave")')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--every_frame', action='store_true',
                       help='If set, extract every frame from the video.')
    group.add_argument('--frame_rate', type=float, default=1.0,
                       help='Number of frames per second to extract (default: 1). Ignored if --every_frame is set.')
    args = parser.parse_args()
    count = extract_frames(args.video, args.output, args.every_frame, args.frame_rate)

    if args.label:
        generate_labels(args.output, args.label)
        print(f"Extracted {count} frames to {args.output} with label '{args.label}'")
        print(f"Generated labels.csv in {args.output}")
