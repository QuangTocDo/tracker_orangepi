import cv2
import numpy as np
import time
from ultralytics import YOLO
from typing import Optional, List, Tuple, Dict
from .logging_python_orangepi import get_logger

logger = get_logger(__name__)

# Các kết nối keypoints
MEDIAPIPE_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10),
    (11, 12), (11, 23), (12, 24), (23, 24),
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
    (24, 26), (26, 28), (28, 30), (28, 32), (30, 32)
]

import mediapipe as mp
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode
from mediapipe.framework.formats import landmark_pb2

class HumanDetection:
    def __init__(self,
                 person_model='python/models/yolo11n.pt',
                 pose_model='python/face_processing/models/pose_landmarker.task'):
        logger.info('Init Human Detection with YOLO + MediaPipe Pose Hybrid Approach')
        
        # YOLO detector
        self.person_detector = YOLO(person_model)
        logger.info(f"Person detector initialized successfully with model: {person_model}")

        # MediaPipe Pose Landmarker
        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=pose_model),
            running_mode=RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            output_segmentation_masks=False
        )
        self.landmarker = PoseLandmarker.create_from_options(options)
        logger.info(f"Pose Landmarker initialized successfully with model: {pose_model}")

        self.fps_avg = 0.0
        self.call_count = 0
        self.last_results = None
        self.classes = [0]

    # ========== Pose detection from bbox ==========
    def detect_pose_from_bbox(self, full_frame: np.ndarray, bbox: tuple):
        x1, y1, x2, y2 = map(int, bbox)
        padding = 10
        person_crop = full_frame[max(0, y1-padding):y2+padding, max(0, x1-padding):x2+padding]
        if person_crop.size == 0:
            return None, None

        person_crop_rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
        detection_result = self.landmarker.detect(person_crop_rgb)

        if detection_result.pose_landmarks:
            pose_landmarks = detection_result.pose_landmarks.landmark
            crop_h, crop_w, _ = person_crop_rgb.shape
            keypoints = np.zeros((33, 3))
            z_coords = np.zeros(33)
            for i, lm in enumerate(pose_landmarks):
                global_x = lm.x * crop_w + (x1 - padding)
                global_y = lm.y * crop_h + (y1 - padding)
                keypoints[i] = [global_x, global_y, lm.visibility]
                z_coords[i] = lm.z
            return keypoints, z_coords
        return None, None

    # ========== Run YOLO + Pose ==========
    def run_detection(self, source: np.ndarray):
        start_time = time.time()
        image_rgb = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        yolo_results = self.person_detector.predict(source=image_rgb, verbose=False, classes=self.classes, conf=0.5)

        all_keypoints, all_z, boxes_data = [], [], []

        for box in yolo_results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            conf = float(box.conf[0])
            boxes_data.append((x1, y1, x2, y2, conf))

            keypoints, z_coords = self.detect_pose_from_bbox(source, (x1, y1, x2, y2))
            if keypoints is not None:
                all_keypoints.append(keypoints)
                all_z.append(z_coords)

        # Update FPS
        duration = time.time() - start_time
        fps_current = 1 / duration if duration > 0 else 0
        self.fps_avg = (self.fps_avg * self.call_count + fps_current) / (self.call_count + 1)
        self.call_count += 1
        logger.info(f"FPS Human detection (YOLO+MediaPipe): {self.fps_avg:.2f}")

        if not all_keypoints:
            self.last_results = None
            return np.array([]), [], np.array([])

        self.last_results = (np.array(all_keypoints), boxes_data)
        return np.array(all_keypoints), boxes_data, np.array(all_z)

    # ========== Draw results ==========
    def draw_results(self, image: np.ndarray, min_conf: float = 0.5):
        if self.last_results is None:
            return image
        annotated_image = image.copy()
        keypoints_data, boxes_data = self.last_results
        for i, (kpts, box) in enumerate(zip(keypoints_data, boxes_data)):
            x1, y1, x2, y2, conf = box
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated_image, f"Person {i+1} ({conf:.2f})", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            points = {j: (int(x), int(y)) for j, (x, y, vis) in enumerate(kpts) if vis > min_conf}
            for start_idx, end_idx in MEDIAPIPE_EDGES:
                if start_idx in points and end_idx in points:
                    cv2.line(annotated_image, points[start_idx], points[end_idx], (255, 255, 0), 1)
            for pt in points.values():
                cv2.circle(annotated_image, pt, 3, (0, 0, 255), -1)
        return annotated_image

    # =================== STATIC METHODS ===================
    @staticmethod
    def _get_limb_coords(side, limb_type, keypoints, z_coords):
        if limb_type == 'arm':
            indices = {
                'main': getattr(landmark_pb2.PoseLandmark, f'{side}_WRIST').value,
                'p1': getattr(landmark_pb2.PoseLandmark, f'{side}_SHOULDER').value,
                'p2': getattr(landmark_pb2.PoseLandmark, f'{side}_ELBOW').value,
                'p3': getattr(landmark_pb2.PoseLandmark, f'{side}_WRIST').value
            }
            labels = ['shoulder', 'elbow', 'wrist']
        elif limb_type == 'leg':
            indices = {
                'main': getattr(landmark_pb2.PoseLandmark, f'{side}_ANKLE').value,
                'p1': getattr(landmark_pb2.PoseLandmark, f'{side}_HIP').value,
                'p2': getattr(landmark_pb2.PoseLandmark, f'{side}_KNEE').value,
                'p3': getattr(landmark_pb2.PoseLandmark, f'{side}_ANKLE').value
            }
            labels = ['hip', 'knee', 'ankle']
        else:
            return None
        coords = {}
        for i, label in enumerate(labels):
            idx = indices[f'p{i+1}']
            x, y, vis = keypoints[idx]
            z = z_coords[idx]
            coords[label] = {'x': x, 'y': y, 'z': z, 'visibility': vis}
        return coords

    @staticmethod
    def select_best_arm(keypoints: np.ndarray, z_coords: np.ndarray, visibility_threshold: float = 0.9):
        left_vis = keypoints[landmark_pb2.PoseLandmark.LEFT_WRIST.value][2]
        right_vis = keypoints[landmark_pb2.PoseLandmark.RIGHT_WRIST.value][2]
        left_z = z_coords[landmark_pb2.PoseLandmark.LEFT_WRIST.value]
        right_z = z_coords[landmark_pb2.PoseLandmark.RIGHT_WRIST.value]

        best_side = None
        if left_vis > visibility_threshold and right_vis > visibility_threshold:
            best_side = 'LEFT' if left_z < right_z else 'RIGHT'
        elif left_vis > visibility_threshold:
            best_side = 'LEFT'
        elif right_vis > visibility_threshold:
            best_side = 'RIGHT'

        if best_side:
            coords = HumanDetection._get_limb_coords(best_side, 'arm', keypoints, z_coords)
            return best_side, coords
        return None, None

    @staticmethod
    def select_best_leg(keypoints: np.ndarray, z_coords: np.ndarray, visibility_threshold: float = 0.8):
        left_vis = keypoints[landmark_pb2.PoseLandmark.LEFT_ANKLE.value][2]
        right_vis = keypoints[landmark_pb2.PoseLandmark.RIGHT_ANKLE.value][2]
        left_z = z_coords[landmark_pb2.PoseLandmark.LEFT_ANKLE.value]
        right_z = z_coords[landmark_pb2.PoseLandmark.RIGHT_ANKLE.value]

        best_side = None
        if left_vis > visibility_threshold and right_vis > visibility_threshold:
            best_side = 'LEFT' if left_z < right_z else 'RIGHT'
        elif left_vis > visibility_threshold:
            best_side = 'LEFT'
        elif right_vis > visibility_threshold:
            best_side = 'RIGHT'

        if best_side:
            coords = HumanDetection._get_limb_coords(best_side, 'leg', keypoints, z_coords)
            return best_side, coords
        return None, None
