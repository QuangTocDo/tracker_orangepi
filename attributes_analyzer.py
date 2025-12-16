# attributes_analyzer.py
import cv2
import asyncio
import numpy as np
import os
import time
from concurrent.futures import ThreadPoolExecutor

import config
from utils.logging_python_orangepi import get_logger
from utils.mediapipe_pose import HumanDetection
from utils.gender_hybrid import GenderClassification
from utils.pose_color_new1 import PoseColorAnalyzer
from utils.clothing_classifier_by_color_new import ClothingClassifier
from utils.cut_body_part import extract_head_from_frame 

# --- IMPORT MODULE AGE/RACE (CHỈ DÙNG ONNX) ---
try:
    from utils.age_race_onnx import AgeRaceEstimatorONNX
except ImportError:
    AgeRaceEstimatorONNX = None

logger = get_logger(__name__)

class AttributesAnalyzer:
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=os.cpu_count())
        self.human_detector = None
        self.gender_classifier = None
        self.clothing_classifier = None
        self.pose_color_analyzer = None
        self.age_race_estimator = None 
        self.models_loaded = False

    async def load_models(self):
        print("--- LOADING ATTRIBUTE MODELS (ONNX ONLY) ---")
        loop = asyncio.get_event_loop()

        # 1. Load các model cơ bản (Pose, Gender, Clothing)
        self.human_detector = await loop.run_in_executor(
            self.executor, lambda: HumanDetection(config.PERSON_MODEL_PATH, config.POSE_MODEL_PATH)
        )
        self.gender_classifier = await loop.run_in_executor(
            self.executor, lambda: GenderClassification(config.GENDER_FACE_MODEL_PATH, config.GENDER_POSE_MODEL_PATH)
        )
        self.clothing_classifier = await loop.run_in_executor(
            self.executor, lambda: ClothingClassifier(config.SKIN_CSV_PATH)
        )
        self.pose_color_analyzer = PoseColorAnalyzer()
        
        # 2. Load Age/Race (Chỉ load ONNX)
        if AgeRaceEstimatorONNX and os.path.exists(config.AGE_RACE_MODEL_ONNX_PATH):
             print(f"--> Loading ONNX Age/Race: {config.AGE_RACE_MODEL_ONNX_PATH}")
             self.age_race_estimator = await loop.run_in_executor(
                 self.executor, lambda: AgeRaceEstimatorONNX(config.AGE_RACE_MODEL_ONNX_PATH)
             )
        else:
            print("❌ KHÔNG LOAD ĐƯỢC MODEL AGE/RACE ONNX! (Kiểm tra lại config.py và file model)")

        self.models_loaded = True
        print("✅ Attribute Models Loaded.")

    def safe_crop(self, image, x1, y1, x2, y2):
        """
        Hàm cắt ảnh an toàn: Tự động kẹp tọa độ vào trong khung hình.
        Sửa lỗi 'Invalid bounding box' khi detect trên Orange Pi.
        """
        if image is None or image.size == 0: return None
        h, w = image.shape[:2]
        
        # Kẹp tọa độ (Clamping) - Đảm bảo không bao giờ tràn viền
        x1 = max(0, min(x1, w))
        y1 = max(0, min(y1, h))
        x2 = max(0, min(x2, w))
        y2 = max(0, min(y2, h))
        
        # Kiểm tra tính hợp lệ: Nếu diện tích bằng 0 hoặc âm
        if x2 <= x1 or y2 <= y1:
            return None
            
        return image[y1:y2, x1:x2]

    async def analyze_person_by_bbox(self, frame, bbox, person_id):
        if not self.models_loaded: return None

        # 1. Cắt người từ khung hình (Dùng safe_crop thay vì cắt trực tiếp)
        bx1, by1, bx2, by2 = map(int, bbox)
        person_crop = self.safe_crop(frame, bx1, by1, bx2, by2)
        
        if person_crop is None: return None

        # Detect Pose
        keypoints, kpts_z = self.human_detector.detect_pose_from_bbox(frame, bbox)
        
        loop = asyncio.get_event_loop()

        # ====================================================
        # CHIẾN THUẬT CẮT MẶT (SỬ DỤNG SAFE CROP)
        # ====================================================
        face_img = None
        method = "None"

        # CÁCH 1: Keypoints (Ưu tiên)
        if keypoints is not None:
            head_bbox = extract_head_from_frame(frame, keypoints, scale=1.2)
            if head_bbox:
                hx1, hy1, hx2, hy2 = head_bbox
                # Tính toán tọa độ ép vuông
                fw, fh = hx2 - hx1, hy2 - hy1
                side = max(fw, fh)
                cx, cy = hx1 + fw//2, hy1 + fh//2
                
                new_hx1 = int(cx - side//2)
                new_hy1 = int(cy - side//2)
                new_hx2 = int(new_hx1 + side)
                new_hy2 = int(new_hy1 + side)

                # Cắt an toàn
                face_img = self.safe_crop(frame, new_hx1, new_hy1, new_hx2, new_hy2)
                if face_img is not None:
                    method = "Keypoints"

        # CÁCH 2: Fallback (Cắt 20% đầu người nếu không có Keypoints)
        if face_img is None:
            h, w = person_crop.shape[:2]
            crop_h = int(h * 0.20)
            if crop_h > 5: 
                crop_w = min(crop_h, w) # Cố gắng lấy vuông
                center_x = w // 2
                start_x = center_x - crop_w // 2
                
                # Cắt an toàn từ person_crop
                face_img = self.safe_crop(person_crop, start_x, 0, start_x + crop_w, crop_h)
                method = "Fallback_20%"
        
        # ====================================================

        # Task Gender
        img_for_gender = face_img if (face_img is not None) else person_crop
        gender_task = loop.run_in_executor(
            self.executor, self.gender_classifier.predict, img_for_gender.copy(), keypoints
        )
        
        # Task Clothing
        clothing_task = self.pose_color_analyzer.process_and_classify(
            image=frame, keypoints=keypoints, kpts_z=kpts_z, classifier=self.clothing_classifier
        )

        # Task Age & Race (Chạy ONNX)
        async def run_age_race():
            if self.age_race_estimator is None: return None
            if face_img is None: return None 
            
            try:
                res = self.age_race_estimator.predict(face_img)
                if res:
                    print(f"🎯 [ID:{person_id}] Age/Race [{method}]: {res}")
                return res
            except Exception as e:
                print(f"❌ [ID:{person_id}] Age Predict Error: {e}")
                return None

        age_race_task = run_age_race()

        # Chạy song song 3 task
        gender_res, clothing_res, age_race_res = await asyncio.gather(gender_task, clothing_task, age_race_task)

        return {
            'id': person_id,
            'status': 'success',
            'timestamp': time.time(),
            'gender_analysis': gender_res,
            'clothing_analysis': clothing_res,
            'age_race_analysis': age_race_res
        }