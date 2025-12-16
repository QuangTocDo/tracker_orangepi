import cv2
import numpy as np
import os
from .logging_python_orangepi import get_logger

logger = get_logger(__name__)

# Cố gắng import thư viện RKNN Lite cho Orange Pi
try:
    from rknnlite.api import RKNNLite as RKNN
except ImportError:
    # Fallback nếu chạy trên PC (để test logic, dù không load được model)
    try:
        from rknn.api import RKNN
    except ImportError:
        RKNN = None

class AgeRaceEstimator:
    def __init__(self, model_path):
        if RKNN is None:
            logger.error("Chưa cài đặt thư viện rknn-toolkit-lite2 hoặc rknn-toolkit2!")
            self.rknn = None
            return

        if not os.path.exists(model_path):
            logger.error(f"Không tìm thấy model Age/Race tại: {model_path}")
            self.rknn = None
            return

        self.rknn = RKNN()
        
        # Load RKNN Model
        logger.info(f"Đang tải model Age/Race: {model_path}...")
        ret = self.rknn.load_rknn(model_path)
        if ret != 0:
            logger.error("Tải model RKNN thất bại!")
            self.rknn = None
            return

        # Init Runtime (Chạy trên NPU RK3588 - Core 0)
        ret = self.rknn.init_runtime(core_mask=RKNN.NPU_CORE_0) 
        if ret != 0:
            logger.error("Khởi tạo RKNN runtime thất bại!")
            self.rknn = None
            return
        
        logger.info("✅ Khởi tạo Age/Race Estimator thành công trên NPU.")

        # --- ĐỊNH NGHĨA NHÃN THEO YÊU CẦU CỦA BẠN ---
        
        # 7 Nhóm tuổi
        self.age_labels = [
            "0-10 (Child)", 
            "11-19 (Teen)", 
            "20-30 (Youth)", 
            "31-40 (Early Mid)", 
            "41-50 (Middle)", 
            "50-69 (Elderly)", 
            "70+ (Old)"
        ]

        # 5 Nhóm chủng tộc
        self.race_labels = [
            "Asian", 
            "White", 
            "Black", 
            "Indian", 
            "Others"
        ]

    def preprocess(self, face_img):
        """
        Chuẩn bị ảnh mặt. Kích thước phổ biến là 224x224.
        Nếu model bạn train size khác (ví dụ 112x112), hãy sửa số ở đây.
        """
        img = cv2.resize(face_img, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.expand_dims(img, axis=0) # Shape: (1, 224, 224, 3)
        return img

    def predict(self, face_img):
        if self.rknn is None or face_img is None or face_img.size == 0:
            return None

        try:
            input_data = self.preprocess(face_img)
            
            # Chạy suy luận (Inference)
            # outputs sẽ là một list chứa 2 mảng numpy: [tensor1, tensor2]
            outputs = self.rknn.inference(inputs=[input_data])
            
            if len(outputs) < 2:
                logger.warning("Model output không đủ 2 nhánh (Age & Race).")
                return None

            # --- POST-PROCESSING TỰ ĐỘNG ---
            # Model có 2 output: 1 cái shape (1, 7) cho Age, 1 cái shape (1, 5) cho Race.
            # Ta kiểm tra shape để gán đúng biến, không lo bị ngược thứ tự.
            
            out1 = outputs[0][0] # Bỏ batch dimension -> shape (N,)
            out2 = outputs[1][0]
            
            age_logits = None
            race_logits = None

            # Logic kiểm tra shape
            if len(out1) == 7 and len(out2) == 5:
                age_logits = out1
                race_logits = out2
            elif len(out1) == 5 and len(out2) == 7:
                race_logits = out1
                age_logits = out2
            else:
                logger.error(f"Shape output không khớp (Kỳ vọng 5 & 7). Nhận được: {len(out1)} và {len(out2)}")
                return None

            # --- XỬ LÝ KẾT QUẢ ---
            
            # 1. Tính Age (Argmax)
            age_idx = np.argmax(age_logits)
            age_label = self.age_labels[age_idx]
            # Tính confidence cho Age (Softmax)
            age_probs = self.softmax(age_logits)
            age_conf = float(age_probs[age_idx])

            # 2. Tính Race (Argmax)
            race_idx = np.argmax(race_logits)
            race_label = self.race_labels[race_idx]
            # Tính confidence cho Race (Softmax)
            race_probs = self.softmax(race_logits)
            race_conf = float(race_probs[race_idx])

            return {
                "age": age_label,
                "race": race_label,
                "age_conf": age_conf,
                "race_conf": race_conf
            }

        except Exception as e:
            logger.error(f"Lỗi khi dự đoán Age/Race: {e}")
            return None

    def softmax(self, x):
        """Hàm phụ trợ tính softmax để lấy % độ tin cậy"""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    def release(self):
        if self.rknn:
            self.rknn.release()