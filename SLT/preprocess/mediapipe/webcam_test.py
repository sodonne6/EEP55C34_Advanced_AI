import time
from pathlib import Path
import cv2
import mediapipe as mp

HERE = Path(__file__).resolve().parent

# If you used "lite", change this filename accordingly.
POSE_MODEL = str(HERE / "models" / "pose_landmarker_full.task")
HAND_MODEL = str(HERE / "models" / "hand_landmarker.task")

BaseOptions = mp.tasks.BaseOptions
RunningMode = mp.tasks.vision.RunningMode

PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions

HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions

# Minimal, hardcoded connections (enough for a good visual sanity check)
HAND_EDGES = [
    (0,1),(1,2),(2,3),(3,4),          # thumb
    (0,5),(5,6),(6,7),(7,8),          # index
    (0,9),(9,10),(10,11),(11,12),     # middle
    (0,13),(13,14),(14,15),(15,16),   # ring
    (0,17),(17,18),(18,19),(19,20),   # pinky
    (5,9),(9,13),(13,17),(5,17)       # palm-ish
]

POSE_EDGES = [
    (11,12),          # shoulders
    (11,13),(13,15),  # left arm
    (12,14),(14,16),  # right arm
    (23,24),          # hips
    (11,23),(12,24)   # torso sides
]

def draw_points(img, lms, color=(255,255,255), r=2):
    h, w = img.shape[:2]
    for lm in lms:
        x = int(lm.x * w)
        y = int(lm.y * h)
        cv2.circle(img, (x, y), r, color, -1)

def draw_edges(img, lms, edges, color=(255,255,255), t=2):
    h, w = img.shape[:2]
    pts = [(int(lm.x*w), int(lm.y*h)) for lm in lms]
    for a, b in edges:
        if a < len(pts) and b < len(pts):
            cv2.line(img, pts[a], pts[b], color, t)

def main(cam_index=0):
    pose_options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=POSE_MODEL),
        running_mode=RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.3,
        min_pose_presence_confidence=0.3,
        min_tracking_confidence=0.3,
    )

    hand_options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=HAND_MODEL),
        running_mode=RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.3,
        min_hand_presence_confidence=0.3,
        min_tracking_confidence=0.3,
    )

    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam index {cam_index}. Try 1.")

    start = time.time()

    with PoseLandmarker.create_from_options(pose_options) as pose_lm, \
         HandLandmarker.create_from_options(hand_options) as hand_lm:

        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            ts_ms = int((time.time() - start) * 1000)

            pose_res = pose_lm.detect_for_video(mp_image, ts_ms)
            hand_res = hand_lm.detect_for_video(mp_image, ts_ms)

            out = frame_bgr.copy()

            # Pose (draw points + a small upper-body skeleton)
            if pose_res.pose_landmarks:
                lms = pose_res.pose_landmarks[0]
                draw_points(out, lms, r=2)
                draw_edges(out, lms, POSE_EDGES, t=2)

            # Hands (draw points + skeleton)
            if hand_res.hand_landmarks:
                for lms in hand_res.hand_landmarks:
                    draw_points(out, lms, r=2)
                    draw_edges(out, lms, HAND_EDGES, t=2)

            cv2.imshow("MediaPipe Tasks: Pose+Hands (q to quit)", out)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main(0)
