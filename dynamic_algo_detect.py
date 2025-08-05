import cv2
import numpy as np
import time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
from mediapipe import solutions
import math

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

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

class HandDetector:
    def __init__(self, model_path="hand_landmarker.task", max_num_hands=2, detection_confidence=0.5):
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options= base_options,
            num_hands= max_num_hands,
            min_hand_detection_confidence= detection_confidence,
            min_hand_presence_confidence= detection_confidence,
            min_tracking_confidence= detection_confidence,
            running_mode= vision.RunningMode.LIVE_STREAM,
            result_callback= self._on_result,
        )

        self.landmarker = vision.HandLandmarker.create_from_options(options)
        self.results = None

    def _on_result(self, result, output_image, timestamp_ms):
        self.results = result

    def findHands(self, img, timestamp, draw = True):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
        self.landmarker.detect_async(mp_image, timestamp)

        if self.results and draw:
            for hand_landmarks in self.results.hand_landmarks:
                landmark_proto = landmark_pb2.NormalizedLandmarkList()
                
                for lm in hand_landmarks:
                    new_lm = landmark_pb2.NormalizedLandmark(
                        x=lm.x,
                        y=lm.y,
                        z=lm.z
                    )
                    landmark_proto.landmark.append(new_lm)

                mp_drawing.draw_landmarks(
                    img,
                    landmark_proto,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
                )
        return img

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
            # print(f"FIST detection failed at {pip}, {dip}, {tip} with angle {angle_between_points(landmarks[tip], landmarks[dip], landmarks[pip])}")
            return False

    thumb_angle = angle_between_points(landmarks[THUMB_TIP], landmarks[THUMB_MCP], landmarks[INDEX_FINGER_PIP])
    if thumb_angle > 55:
        # print(f"FIST detection failed at thumb with angle {thumb_angle}")
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
            # print(f"Fist detection failed at {mcp}, {pip}, {tip} with angle {ang}")
            return False

    downward = landmarks[THUMB_TIP][1] > landmarks[WRIST][1] and distance(landmarks[WRIST], landmarks[THUMB_TIP]) > \
                        distance(landmarks[WRIST], landmarks[THUMB_IP]) * 1.1   

    if downward:

        thumb_angle = angle_between_points(landmarks[WRIST], landmarks[THUMB_MCP], landmarks[THUMB_TIP])
        if thumb_angle < 30:
            # print(f"thumbd detection failed at thumb with angle {thumb_angle}")
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
            # print(f"thumbsup detection failed at {mcp}, {pip}, {tip} with angle {ang}")
            return False

    upward = landmarks[THUMB_TIP][1] < landmarks[WRIST][1] and distance(landmarks[WRIST], landmarks[THUMB_TIP]) > \
                        distance(landmarks[WRIST], landmarks[THUMB_IP]) * 1.1   
    if upward:

        thumb_angle = angle_between_points(landmarks[WRIST], landmarks[THUMB_MCP], landmarks[THUMB_TIP])
        if thumb_angle < 40:
            #print(f"THUMBSUP detection failed at thumb with angle {thumb_angle}")
            return False

        return True   
    
    return False


def main():
    cap = cv2.VideoCapture(0)
    detector = HandDetector()
    prev_time = 0
    timestamp = 0
    display_status = "none"

    open_palm_start_time = None
    swipe_start_x = None
    SWIPE_THRESHOLD = 0.1

    while cap.isOpened():
        success, img = cap.read()
        if not success:
            continue

        timestamp += int(1000 / 30)  # ms between frames assuming 30 FPS

        # convert BGR to RGB for processing
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        detector.findHands(rgb_img, timestamp)
        img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)

        if detector.results and detector.results.hand_landmarks:
            for hand_landmarks in detector.results.hand_landmarks:
                points = [[lm.x, lm.y, lm.z] for lm in hand_landmarks]

                if is_ok(points):
                    display_status = "ok"
                elif is_fist(points):
                    display_status = "fist"
                elif is_palm(points):
                    display_status = "open_palm"
                elif is_pointup(points):
                    display_status = "point_up"
                elif is_thumbs_down(points):
                    display_status = "thumbs_down"
                elif is_thumbs_up(points):
                    display_status = "thumbs_up"
                else:
                    display_status = "none"
        
                if display_status == "open_palm":
                    if open_palm_start_time is None:
                        open_palm_start_time = time.time()
                        swipe_start_x = points[0][0] # wrist x coordinate
                    if time.time() - open_palm_start_time > 0.3:
                        current_x = points[0][0] 
                        dx = swipe_start_x - current_x

                        if dx > SWIPE_THRESHOLD:
                            print("Swipe Right detected!")
                            open_palm_start_time = None
                            swipe_start_x = None
                        elif dx < -SWIPE_THRESHOLD:
                            print("Swipe Left detected!")
                            open_palm_start_time = None
                            swipe_start_x = None
                else:
                    open_palm_start_time = None
                    swipe_start_x = None


        cv2.putText(img, f"Gesture: {display_status}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)

        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time

        cv2.putText(img, f'FPS: {int(fps)}', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        cv2.imshow("Hand Landmarker", img)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
