# utils/age_race_onnx.py
import cv2
import numpy as np
import os
import onnxruntime as ort
from .logging_python_orangepi import get_logger

logger = get_logger(__name__)

class AgeRaceEstimatorONNX:
    def __init__(self, model_path):
        if not os.path.exists(model_path):
            logger.error(f"❌ [ONNX] Không tìm thấy file model: {model_path}")
            self.session = None
            return

        logger.info(f"🔄 [ONNX] Đang tải model Age/Race: {model_path}...")
        try:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            self.session = ort.InferenceSession(model_path, providers=providers)
            
            self.input_name = self.session.get_inputs()[0].name
            input_shape = self.session.get_inputs()[0].shape
            self.input_height = input_shape[2] if len(input_shape) == 4 else 224
            self.input_width = input_shape[3] if len(input_shape) == 4 else 224
            
            logger.info(f"✅ [ONNX] Khởi tạo thành công! Input: {self.input_width}x{self.input_height}")
        except Exception as e:
            logger.error(f"❌ [ONNX] Lỗi khởi tạo session: {e}")
            self.session = None

        self.age_labels = ["0-10", "11-19", "20-30", "31-40", "41-50", "50-69", "70+"]
        self.race_labels = ["Asian", "White", "Black", "Indian", "Others"]

    def preprocess(self, face_img):
        img = cv2.resize(face_img, (self.input_width, self.input_height))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.transpose(2, 0, 1) # HWC -> CHW
        img = np.expand_dims(img, axis=0)
        img = img.astype(np.float32) / 255.0
        return img

    def predict(self, face_img):
        if self.session is None or face_img is None or face_img.size == 0:
            return None

        try:
            input_data = self.preprocess(face_img)
            outputs = self.session.run(None, {self.input_name: input_data})
            
            out1 = outputs[0][0]
            out2 = outputs[1][0]
            
            # Tự động nhận diện output nào là Age (7 lớp), nào là Race (5 lớp)
            age_logits, race_logits = None, None
            if len(out1) == 7 and len(out2) == 5:
                age_logits, race_logits = out1, out2
            elif len(out1) == 5 and len(out2) == 7:
                race_logits, age_logits = out1, out2
            else:
                return None

            age_idx = np.argmax(age_logits)
            race_idx = np.argmax(race_logits)
            
            return {
                "age": self.age_labels[age_idx],
                "race": self.race_labels[race_idx]
            }
        except Exception as e:
            logger.error(f"Lỗi predict ONNX: {e}")
            return None