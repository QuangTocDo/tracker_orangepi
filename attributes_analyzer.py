import cv2
import asyncio
import numpy as np
import os
import time
from concurrent.futures import ThreadPoolExecutor
from glob import glob

# Import tất cả cấu hình từ file config.py
import config

# Import các module xử lý
from utils.logging_python_orangepi import setup_logging, get_logger
from utils.mediapipe_pose import HumanDetection
from utils.gender_hybrid import GenderClassification
from utils.pose_color_new1 import PoseColorAnalyzer
from utils.clothing_classifier_by_color_new import ClothingClassifier

# --- Thiết lập logging ---
# setup_logging() # Bỏ dòng này để tránh xung đột với logging của main_track
logger = get_logger(__name__)

class AttributesAnalyzer:
    """
    Lớp chuyên phân tích các thuộc tính của con người như giới tính, quần áo.
    Đã được tái cấu trúc để có thể import như một module.
    """
    def __init__(self):
        """Hàm khởi tạo, chuẩn bị môi trường."""
        self.executor = ThreadPoolExecutor(max_workers=os.cpu_count())
        self.human_detector = None
        self.gender_classifier = None
        self.clothing_classifier = None
        self.pose_color_analyzer = None
        self.models_loaded = False

    def _check_models(self):
        """Kiểm tra sự tồn tại của các file model cần thiết."""
        logger.info("Kiểm tra các file model cho phân tích thuộc tính...")
        missing_files = []
        # Giả sử config được import đúng cách và có các đường dẫn này
        required_paths = {
            "Person Model": config.PERSON_MODEL_PATH,
            "Pose Model": config.POSE_MODEL_PATH,
            "Gender Face Model": config.GENDER_FACE_MODEL_PATH,
            "Gender Pose Model": config.GENDER_POSE_MODEL_PATH,
            "Skin CSV": config.SKIN_CSV_PATH,
        }
        for name, path in required_paths.items():
            if not os.path.exists(path):
                missing_files.append((name, path))

        if missing_files:
            logger.error("="*50)
            logger.error("LỖI: Một hoặc nhiều file model cho AttributesAnalyzer không được tìm thấy!")
            for name, path in missing_files:
                logger.error(f" - {name}: '{path}'")
            logger.error("="*50)
            return False
        logger.info("Tất cả các file model thuộc tính đã được tìm thấy.")
        return True

    async def load_models(self):
        """Tải các model xử lý một cách bất đồng bộ. Phải được gọi sau khi khởi tạo."""
        if not self._check_models():
            raise RuntimeError("Không thể tải models do thiếu file.")

        logger.info("Bắt đầu tải các model phân tích thuộc tính...")
        loop = asyncio.get_event_loop()

        self.human_detector = await loop.run_in_executor(
            self.executor, lambda: HumanDetection(person_model=config.PERSON_MODEL_PATH, pose_model=config.POSE_MODEL_PATH)
        )
        self.gender_classifier = await loop.run_in_executor(
            self.executor, lambda: GenderClassification(
                gender_face_model_path=config.GENDER_FACE_MODEL_PATH,
                gender_pose_model_path=config.GENDER_POSE_MODEL_PATH
            )
        )
        self.clothing_classifier = await loop.run_in_executor(
            self.executor, lambda: ClothingClassifier(skin_csv_path=config.SKIN_CSV_PATH)
        )
        self.pose_color_analyzer = PoseColorAnalyzer()
        self.models_loaded = True
        logger.info("Tải model phân tích thuộc tính hoàn tất.")

    async def analyze_person_by_bbox(self, frame, bbox, person_id):
        """
        Phân tích một người duy nhất dựa trên bounding box và ID được cung cấp.
        Hàm này sẽ tự thực hiện việc ước tính tư thế và sau đó chạy các phân tích khác.
        """
        if not self.models_loaded:
            logger.error("Models chưa được tải! Vui lòng gọi hàm load_models() trước.")
            return {'id': person_id, 'status': 'error', 'reason': 'Models not loaded'}

        logger.info(f"[Thuộc tính] Bắt đầu phân tích cho ID: {person_id} tại bbox: {bbox}")
        
        # Bước 1: Lấy keypoint từ bbox
        keypoints, kpts_z = self.human_detector.detect_pose_from_bbox(frame, bbox)

        if keypoints is None or kpts_z is None:
            logger.warning(f"[Thuộc tính] Không thể ước tính tư thế cho ID {person_id}. Bỏ qua.")
            return {'id': person_id, 'status': 'error', 'reason': 'Pose estimation failed'}

        try:
            x1, y1, x2, y2 = map(int, bbox)
            person_crop = frame[max(0, y1):y2, max(0, x1):x2]
            if person_crop.size == 0:
                return {'id': person_id, 'status': 'error', 'reason': 'Empty crop from bbox'}
            
            loop = asyncio.get_event_loop()
            
            gender_task = loop.run_in_executor(
                self.executor, self.gender_classifier.predict, person_crop.copy(), keypoints
            )
            clothing_task = self.pose_color_analyzer.process_and_classify(
                image=frame, keypoints=keypoints, kpts_z=kpts_z, classifier=self.clothing_classifier
            )
            
            gender_result, clothing_result = await asyncio.gather(gender_task, clothing_task)

            final_result = {
                'id': person_id,
                'status': 'success',
                'timestamp': time.time(),
                'gender_analysis': gender_result,
                'clothing_analysis': clothing_result
            }
            logger.info(f"[Thuộc tính] Phân tích hoàn tất cho ID: {person_id}")
            return final_result

        except Exception as e:
            logger.error(f"[Thuộc tính] Lỗi không mong muốn khi phân tích ID {person_id}: {e}", exc_info=True)
            return {'id': person_id, 'status': 'error', 'reason': str(e)}