import argparse
import glob
import cv2
import numpy as np
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import timeit
import time
from pathlib import Path
import math
from PIL import Image, ImageOps
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

'''
example: python algorithm_detect.py gesture_recognition_custom_dataset
'''

MARGIN = 10
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54)

WRIST = 0
THUMB_CMC = 1
THUMB_MCP = 2
THUMB_IP = 3
THUMB_TIP = 4
INDEX_FINGER_MCP = 5
INDEX_FINGER_PIP = 6
INDEX_DIP = 7
INDEX_TIP = 8
MIDDLE_FINGER_MCP = 9
MIDDLE_FINGER_PIP = 10
MIDDLE_FINGER_DIP = 11
MIDDLE_TIP = 12
RING_FINGER_MCP = 13
RING_FINGER_PIP = 14
RING_FINGER_DIP = 15
RING_TIP = 16
PINKY_MCP = 17
PINKY_PIP = 18
PINKY_DIP = 19
PINKY_TIP = 20

def draw_landmarks_on_image(rgb_image, detection_result):
    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness
    annotated_image = np.copy(rgb_image)

    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        handedness = handedness_list[idx]

        # Draw the hand landmarks.
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
        ])
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            hand_landmarks_proto,
            solutions.hands.HAND_CONNECTIONS,
            solutions.drawing_styles.get_default_hand_landmarks_style(),
            solutions.drawing_styles.get_default_hand_connections_style())

    return annotated_image

# returns a n
def distance(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

def angle_between_points(a, b, c):
    ab = distance(a, b)
    bc = distance(b, c)
    ac = distance(a, c)

    # do NOT divide by zero
    if ab * bc == 0:
        return 0.0

    # law of cosines!
    cos_angle = (ab**2 + bc**2 - ac**2) / (2 * ab * bc)
    # clamp the value to the range [-1, 1] to avoid NaN from acos
    cos_angle = max(-1.0, min(1.0, cos_angle))

    # arccos and convert to degrees
    angle_rad = math.acos(cos_angle)
    return math.degrees(angle_rad)

def is_horns(landmarks):
    # the distance between wrist and thumb tip is larger 
    thumb_extended = distance(landmarks[WRIST], landmarks[THUMB_TIP]) > \
                     distance(landmarks[WRIST], landmarks[THUMB_IP]) * 1.1

    middle_down = distance(landmarks[WRIST], landmarks[MIDDLE_TIP]) < \
        distance(landmarks[WRIST], landmarks[MIDDLE_FINGER_DIP])
    
    ring_down = distance(landmarks[WRIST], landmarks[RING_TIP]) < \
        distance(landmarks[WRIST], landmarks[RING_FINGER_DIP])
    
    index_up = distance(landmarks[WRIST], landmarks[INDEX_TIP]) > \
                     distance(landmarks[WRIST], landmarks[INDEX_DIP]) * 1.1

    pinky_up = distance(landmarks[WRIST], landmarks[PINKY_TIP]) > \
                     distance(landmarks[WRIST], landmarks[PINKY_DIP]) * 1.1

    return thumb_extended and middle_down and ring_down and index_up and pinky_up

# def is_fist(landmarks):

#     MCPs = [INDEX_FINGER_MCP, MIDDLE_FINGER_MCP, RING_FINGER_MCP, PINKY_MCP]
#     PIPs = [INDEX_FINGER_PIP, MIDDLE_FINGER_PIP, RING_FINGER_PIP, PINKY_PIP]
#     TIPS = [INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]

#     lenient_finger = 0
#     for mcp, pip, tip in zip(MCPs, PIPs, TIPS):
#         ang = angle_between_points(landmarks[mcp], landmarks[pip], landmarks[tip])
#         if ang > 150:
#             if ang < 170:
#                 lenient_finger += 1
#                 if lenient_finger > 1:
#                     print(f"Fist detection HERE failed at {mcp}, {pip}, {tip} with angle {ang}")
#                     return False
#                 continue
#             print(f"Fist detection failed at {mcp}, {pip}, {tip} with angle {ang}")
#             return False
    
#     thumb_angle = angle_between_points(landmarks[THUMB_TIP], landmarks[THUMB_CMC], landmarks[INDEX_FINGER_PIP])
#     if thumb_angle > 45:
#         print(f"FIST detection failed at thumb with angle {thumb_angle}")
#         return False
        
#     return True

def is_fist(landmarks):
    MCPs = [THUMB_CMC, INDEX_FINGER_MCP, MIDDLE_FINGER_MCP, RING_FINGER_MCP, PINKY_MCP]
    PIPs = [THUMB_MCP, INDEX_FINGER_PIP, MIDDLE_FINGER_PIP, RING_FINGER_PIP, PINKY_PIP]
    DIPs = [THUMB_IP, INDEX_DIP, MIDDLE_FINGER_DIP, RING_FINGER_DIP, PINKY_DIP]
    TIPS = [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]

    for pip, dip, tip in zip(PIPs[1:], DIPs[1:], TIPS[1:], ):
        if angle_between_points(landmarks[tip], landmarks[dip], landmarks[pip]) > 160:
            print(f"FIST detection failed at {pip}, {dip}, {tip} with angle {angle_between_points(landmarks[tip], landmarks[dip], landmarks[pip])}")
            return False

    thumb_angle = angle_between_points(landmarks[THUMB_TIP], landmarks[THUMB_MCP], landmarks[INDEX_FINGER_PIP])
    if thumb_angle > 55:
        print(f"FIST detection failed at thumb with angle {thumb_angle}")
        return False
          
    return True


def is_ok(landmarks):
    # check if thumb and index are touching
    touching = abs(distance(landmarks[WRIST], landmarks[THUMB_TIP]) - distance(landmarks[WRIST], landmarks[INDEX_TIP])) < 0.1

    index_curled = distance(landmarks[INDEX_TIP], landmarks[MIDDLE_TIP]) > \
                    distance(landmarks[INDEX_TIP], landmarks[THUMB_TIP])
    
    middle_up = distance(landmarks[WRIST], landmarks[MIDDLE_TIP]) > \
                     distance(landmarks[WRIST], landmarks[MIDDLE_FINGER_DIP])

    ring_up = distance(landmarks[WRIST], landmarks[RING_TIP]) > \
                     distance(landmarks[WRIST], landmarks[RING_FINGER_DIP])

    pinky_up = distance(landmarks[WRIST], landmarks[PINKY_TIP]) > \
                     distance(landmarks[WRIST], landmarks[PINKY_DIP])
    
    return touching and middle_up and ring_up and pinky_up and index_curled

def is_palm(landmarks):

    MCPs = [INDEX_FINGER_MCP, MIDDLE_FINGER_MCP, RING_FINGER_MCP, PINKY_MCP]
    PIPs = [INDEX_FINGER_PIP, MIDDLE_FINGER_PIP, RING_FINGER_PIP, PINKY_PIP]
    TIPS = [INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]

    for mcp, pip, tip in zip(MCPs, PIPs, TIPS):
        ang = angle_between_points(landmarks[mcp], landmarks[pip], landmarks[tip])
        if ang < 150:
            # print(f"Palm detection failed at {mcp}, {pip}, {tip} with angle {ang}")
            return False

    thumb_angle = angle_between_points(landmarks[WRIST], landmarks[THUMB_MCP], landmarks[THUMB_TIP])
    if thumb_angle < 30:
        # print(f"Palm detection failed at thumb with angle {thumb_angle}")
        return False

    thumb_pos = angle_between_points(landmarks[THUMB_TIP], landmarks[INDEX_TIP], landmarks[MIDDLE_TIP])
    if thumb_pos < 60:
        return False
    
    return True

def is_pointup(landmarks):
    index_up = distance(landmarks[WRIST], landmarks[INDEX_TIP]) > \
               distance(landmarks[WRIST], landmarks[INDEX_DIP]) * 1.1

    other_fingers_curl = all(
        distance(landmarks[WRIST], landmarks[finger_tip]) < 
        distance(landmarks[WRIST], landmarks[finger_dip]) * 1.1
        for finger_tip, finger_dip in zip(
            [MIDDLE_TIP, RING_TIP, PINKY_TIP],
            [MIDDLE_FINGER_DIP, RING_FINGER_DIP, PINKY_DIP]
        )
    )

    # UNCOMMENT AFTER ALTERING THE DATASET
    is_upward = landmarks[INDEX_TIP][1] < landmarks[WRIST][1]

    return index_up and other_fingers_curl and is_upward

def is_thumbs_down(landmarks): 

    MCPs = [INDEX_FINGER_MCP, MIDDLE_FINGER_MCP, RING_FINGER_MCP, PINKY_MCP]
    PIPs = [INDEX_FINGER_PIP, MIDDLE_FINGER_PIP, RING_FINGER_PIP, PINKY_PIP]
    TIPS = [INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]

    lenient_finger = 0
    for mcp, pip, tip in zip(MCPs, PIPs, TIPS):
        ang = angle_between_points(landmarks[mcp], landmarks[pip], landmarks[tip])
        if ang > 150:
            if ang < 170:
                lenient_finger += 1
                if lenient_finger > 1:
                    return False
                continue
            print(f"Fist detection failed at {mcp}, {pip}, {tip} with angle {ang}")
            return False

    downward = landmarks[THUMB_TIP][1] > landmarks[WRIST][1] and distance(landmarks[WRIST], landmarks[THUMB_TIP]) > \
                        distance(landmarks[WRIST], landmarks[THUMB_IP]) * 1.1   

    if downward:

        thumb_angle = angle_between_points(landmarks[WRIST], landmarks[THUMB_MCP], landmarks[THUMB_TIP])
        if thumb_angle < 30:
            print(f"thumbd detection failed at thumb with angle {thumb_angle}")
            return False
        
        return True   
    
    return False

def is_thumbs_up(landmarks): 

    MCPs = [INDEX_FINGER_MCP, MIDDLE_FINGER_MCP, RING_FINGER_MCP, PINKY_MCP]
    PIPs = [INDEX_FINGER_PIP, MIDDLE_FINGER_PIP, RING_FINGER_PIP, PINKY_PIP]
    TIPS = [INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]

    lenient_finger = 0
    for mcp, pip, tip in zip(MCPs, PIPs, TIPS):
        ang = angle_between_points(landmarks[mcp], landmarks[pip], landmarks[tip])
        if ang > 150:
            if ang < 170:
                if mcp == INDEX_FINGER_MCP:
                    return False
                lenient_finger += 1
                if lenient_finger > 1:
                    return False
                continue
            print(f"Fist detection failed at {mcp}, {pip}, {tip} with angle {ang}")
            return False

    upward = landmarks[THUMB_TIP][1] < landmarks[WRIST][1] and distance(landmarks[WRIST], landmarks[THUMB_TIP]) > \
                        distance(landmarks[WRIST], landmarks[THUMB_IP]) * 1.1   
    if upward:

        thumb_angle = angle_between_points(landmarks[WRIST], landmarks[THUMB_MCP], landmarks[THUMB_TIP])
        if thumb_angle < 40:
            print(f"THUMBSUP detection failed at thumb with angle {thumb_angle}")
            return False

        return True   
    
    return False

def load_image_correct_orientation(image_path):
    """Loads an image with corrected orientation based on EXIF data."""
    with Image.open(image_path) as img:
        img = ImageOps.exif_transpose(img)
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def main():
    parser = argparse.ArgumentParser(description="Detect gestures in images")
    parser.add_argument("folder", type=str, help="Path to the folder containing input images")
    parser.add_argument("--show", action="store_true", help="Show annotated images")

    args = parser.parse_args()

    base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
    options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
    detector = vision.HandLandmarker.create_from_options(options)

    parent = Path(args.folder)

    start_time = time.time()

    stats = {}
    y_true = []
    y_pred = []

    for child in parent.iterdir():
        if child.is_dir():
            childname = child.name
            # if childname not in ["ok"]:
            #     continue
            stats[childname] = [0, 0, 0, 0]
            match childname:
                case "fist":
                    stats[childname][3] = 17
                case "none":
                    stats[childname][3] = 2
                case "ok":
                    stats[childname][3] = 10
                case "open_palm":
                    stats[childname][3] = 2
                case "point_up":
                    stats[childname][3] = 6
                case "thumbs_down":
                    stats[childname][3] = 1
                case "thumbs_up":
                    stats[childname][3] = 43


            print(f"\nFolder: {childname}\n")
            for jpg in child.glob('*.jpg'):
                image_path = str(jpg)
                y_true.append(childname)

                image_start = time.time()

                # img = cv2.imread(image_path)
                img = load_image_correct_orientation(image_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                if img is None:
                    print(f"Skipping unreadable image: {image_path}")
                    continue

                # image = mp.Image.create_from_file(image_path)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)

                detection_result = detector.detect(image)

                display_status = "none"

                for i, hand_landmarks in enumerate(detection_result.hand_landmarks):
                    points = [[lm.x, lm.y, lm.z] for lm in hand_landmarks]
                    # if is_horns(points):
                    #     display_status = "horns"
                    if is_ok(points):
                        display_status = "ok"
                    if is_fist(points):
                        display_status = "fist"
                    if is_palm(points):
                        display_status = "open_palm"
                    if is_pointup(points):
                        display_status = "point_up"
                    if is_thumbs_down(points):
                        display_status = "thumbs_down"
                    if is_thumbs_up(points):
                        display_status = "thumbs_up"
                
                image_end = time.time()

                y_pred.append(display_status)
                execution_time = image_end - image_start

                if display_status == childname:
                    stats[childname][0] += 1
                stats[childname][1] += 1
                stats[childname][2] += execution_time

                print(f"{image_path}: {display_status}\t\t{execution_time:12.6f}")

                # annotate image
                if args.show and display_status != childname:
                    annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)

                    # add text label
                    if detection_result.hand_landmarks:
                        height, width, _ = annotated_image.shape
                        x_coordinates = [lm.x for lm in detection_result.hand_landmarks[0]]
                        y_coordinates = [lm.y for lm in detection_result.hand_landmarks[0]]
                        text_x = int(min(x_coordinates) * width)
                        text_y = int(min(y_coordinates) * height) - 10
                        cv2.putText(annotated_image, display_status, (text_x, text_y),
                                    cv2.FONT_HERSHEY_DUPLEX, FONT_SIZE, HANDEDNESS_TEXT_COLOR,
                                    FONT_THICKNESS, cv2.LINE_AA)
                    resized_image = cv2.resize(annotated_image, None, fx=0.3, fy=0.3, interpolation=cv2.INTER_AREA)
                    cv2.imshow(f"{childname}", cv2.cvtColor(resized_image, cv2.COLOR_RGB2BGR))
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()

    end_time = time.time()
    total_time = end_time - start_time

    total_images = 0
    total_correct = 0
    total_adjusted_correct = 0
    total_misreads = 0
    for gesture_name, (correct, total, timer, misread) in stats.items():
        total_images += total
        total_correct += correct
        accuracy = correct / total * 100

        adjusted_correct = correct + misread
        adjusted_accuracy = adjusted_correct / total * 100
        total_adjusted_correct += adjusted_correct
        total_misreads += misread
        print(f"{gesture_name:<15} | Accuracy:           {accuracy:6.2f}%")
        print(f"{'':<15} | Corrected Accuracy: {adjusted_accuracy:6.2f}% ({misread} misreads)")
        print(f"{'':<15} | Average Time:       {timer/total_images:.6f}s")


    print(f"total accuracy: {total_correct / total_images * 100:.2f}%")
    print(f"total adjusted accuracy: {total_adjusted_correct / total_images * 100:.2f}% ({total_misreads} total misreads)")
    print(f"total execution time: {total_time}")
    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    gesture_labels = list(stats.keys())
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=gesture_labels)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.show()

# Entry point
if __name__ == "__main__":
    main()
