import cv2
import numpy as np
import mediapipe as mp
import time
import os
from .dlib_aligner import FaceAligner

# --- Thư viện mới của MediaPipe Tasks ---
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
# XÓA BỎ IMPORT CŨ, chúng ta không cần nó nữa
# from mediapipe.framework.formats import detection_pb2

# THÊM IMPORT MỚI: namedtuple để tạo một đối tượng đơn giản
from collections import namedtuple

# TẠO LỚP ĐƠN GIẢN ĐỂ LƯU BBOX, thay thế cho đối tượng phức tạp của MediaPipe
# Nó có các thuộc tính y hệt như code cũ cần (.xmin, .ymin, .width, .height)
RelativeBoundingBox = namedtuple("RelativeBoundingBox", ["xmin", "ymin", "width", "height"])


class FaceDetection:
    """ 
    Lớp phát hiện khuôn mặt sử dụng API MediaPipe Tasks mới.
    """
    def __init__(self,
                 model_name: str = "blaze_face_short_range.tflite",
                 min_detection_confidence: float = 0.4):
        
        model_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "models",
            model_name
        )

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Không tìm thấy file model tại: {model_path}. Hãy chắc chắn bạn đã tải và đặt nó vào thư mục 'models'.")

        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceDetectorOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            min_detection_confidence=min_detection_confidence
        )
        self.detector = vision.FaceDetector.create_from_options(options)
        self.face_aligner = FaceAligner()
        self.fps_avg = 0.0
        self.call_count = 0
        print(f"FaceDetection (MediaPipe Tasks) đã khởi tạo thành công với model: {model_name}")

    def detect(self, image: np.ndarray):
        start_time = time.time()
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        detection_result = self.detector.detect(mp_image)

        detections_info = []
        raw_detections = []
        
        if detection_result.detections:
            img_h, img_w = image.shape[:2]
            for det in detection_result.detections:
                score = det.categories[0].score
                bbox_abs = det.bounding_box
                
                # SỬA LỖI: Dùng đối tượng RelativeBoundingBox đơn giản mà chúng ta đã tạo
                bbox_rel = RelativeBoundingBox(
                    xmin=bbox_abs.origin_x / img_w,
                    ymin=bbox_abs.origin_y / img_h,
                    width=bbox_abs.width / img_w,
                    height=bbox_abs.height / img_h,
                )

                keypoints_rel = [(kp.x, kp.y) for kp in det.keypoints]
                info = {
                    'confidence': score,
                    'bbox': bbox_rel,
                    'keypoints': keypoints_rel
                }
                detections_info.append(info)
                raw_detections.append(det)

        end_time = time.time()
        duration = end_time - start_time
        self.call_count += 1
        
        return detections_info, raw_detections

    # ... các hàm còn lại (close, detect_and_align) giữ nguyên không thay đổi ...
    def close(self):
        if self.detector:
            self.detector.close()
            print("Tài nguyên của FaceDetector đã được giải phóng.")

    def detect_and_align(self, image, margin: float = 0.3, padding: float = 0.2):
        infos, _ = self.detect(image)
        if not infos:
            return None
        bbox = infos[0]['bbox']
        h, w = image.shape[:2]
        x1 = max(0, int((bbox.xmin * w) - bbox.width * w * margin))
        y1 = max(0, int((bbox.ymin * h) - bbox.height * h * margin))
        x2 = min(w, int(x1 + bbox.width * w * (1 + 2 * margin)))
        y2 = min(h, int(y1 + bbox.height * h * (1 + 2 * margin)))
        roi = image[y1:y2, x1:x2]
        if roi.size == 0:
            return None
        return self.face_aligner.aligning(roi, padding=padding)