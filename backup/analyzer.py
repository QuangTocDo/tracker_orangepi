import torch
import torchreid
import numpy as np
import cv2
from PIL import Image
from deepface import DeepFace
from torchvision import transforms
import os 
from typing import Optional, Tuple
# --- Cấu hình ---
# Đường dẫn tương đối đến thư mục models
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models')

# Cấu hình OSNet (Re-ID)
OSNET_MODEL_PATH = os.path.join(MODELS_DIR, "osnet_ain_x1_0_msmt17_256x128_amsgrad_ep50_lr0_0015_coslr_b64_fb10.pth")
OSNET_INPUT_SIZE = (128, 256) # (width, height)

# Cấu hình Face Recognition
DEEPFACE_MODEL_NAME = 'Dlib'
DEEPFACE_MODEL_NAME = 'Facenet'

MOBILEFACENET_MODEL_PATH = os.path.join(MODELS_DIR, "mobilefacenet.pt") # File .pt bạn cung cấp
MOBILEFACENET_INPUT_SIZE = (112, 112) # (width, height)


class Analyzer:
    """
    Class chứa các model AI để trích xuất vector đặc trưng.
    - Hỗ trợ OSNet cho Re-ID.
    - Hỗ trợ DeepFace hoặc MobileFaceNetV2 cho nhận diện khuôn mặt.
    """
    def __init__(self, face_model_name: str = 'deepface'):
        """
        Khởi tạo và tải các model cần thiết.

        Args:
            face_model_name (str): Tên model nhận diện khuôn mặt cần sử dụng.
                                   Hỗ trợ: 'deepface' (mặc định) hoặc 'mobilefacenet'.
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        #self.device = 'cpu'

        print(f"✅ Analyzer đang sử dụng thiết bị: {self.device}")

        # --- Kiểm tra và lưu lựa chọn model face ---
        if face_model_name.lower() not in ['deepface', 'mobilefacenet']:
            raise ValueError(f"Model khuôn mặt '{face_model_name}' không được hỗ trợ. Vui lòng chọn 'deepface' hoặc 'mobilefacenet'.")
        self.face_model_name = face_model_name.lower()
        print(f"✅ Model nhận diện khuôn mặt được chọn: {self.face_model_name}")

        try:
            # 1. Tải model Re-ID (OSNet)
            self._load_osnet_model()

            # 2. Tải model Face Recognition được chọn
            if self.face_model_name == 'mobilefacenet':
                self._load_mobilefacenet_model()
            else: # 'deepface'
                self._warmup_deepface_model()

        except Exception as e:
            print(f"❌ Lỗi nghiêm trọng khi khởi tạo Analyzer: {e}")
            raise

    def _load_osnet_model(self):
        """Tải kiến trúc và trọng số của model OSNet."""
        print("Đang tải model OSNet...")
        self.osnet_model = torchreid.models.build_model(
            name='osnet_ain_x1_0', num_classes=1000, loss='softmax', pretrained=False
        )
        torchreid.utils.load_pretrained_weights(self.osnet_model, OSNET_MODEL_PATH)
        self.osnet_model.to(self.device)
        self.osnet_model.eval()

        _, self.osnet_transform = torchreid.reid.data.transforms.build_transforms(
            height=OSNET_INPUT_SIZE[1], width=OSNET_INPUT_SIZE[0], is_train=False
        )
        print("✅ Tải model OSNet và transform thành công.")

    def _load_mobilefacenet_model(self):
        """Tải model MobileFaceNetV2 từ file .pt và tạo transform."""
        print("Đang tải model MobileFaceNetV2...")
        if not os.path.exists(MOBILEFACENET_MODEL_PATH):
            raise FileNotFoundError(f"Không tìm thấy file model MobileFaceNet: {MOBILEFACENET_MODEL_PATH}")

        self.face_model = torch.jit.load(MOBILEFACENET_MODEL_PATH)
        self.face_model.to(self.device)
        self.face_model.eval()

        self.face_transform = transforms.Compose([
            transforms.Resize(MOBILEFACENET_INPUT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        print("✅ Tải model MobileFaceNetV2 và transform thành công.")

    def _warmup_deepface_model(self):
        """Khởi động DeepFace để tải model về cache."""
        print("Đang khởi động model DeepFace...")
        _ = DeepFace.represent(
            np.zeros((100, 100, 3), dtype=np.uint8),
            model_name=DEEPFACE_MODEL_NAME,
            enforce_detection=False
        )
        print(f"✅ Khởi động model DeepFace ({DEEPFACE_MODEL_NAME}) thành công.")

    def extract_reid_feature(self, person_crop: np.ndarray) -> Optional[list]:
        """Trích xuất vector đặc trưng Re-ID từ ảnh crop của một người."""
        if person_crop is None or person_crop.size == 0:
            return None
        try:
            rgb_crop = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_crop)
            transformed_image = self.osnet_transform(pil_image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                embedding = self.osnet_model(transformed_image)
            embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
            return embedding.cpu().numpy().flatten().tolist()
        except Exception as e:
            print(f"Lỗi khi trích xuất đặc trưng Re-ID: {e}")
            return None

    def extract_face_feature(self, face_crop: np.ndarray) -> Tuple[Optional[list],float]:
        """
        Trích xuất vector đặc trưng khuôn mặt bằng model đã được chọn khi khởi tạo.
        Luôn trả về một tuple (vector, confidence).
        """
        if face_crop is None or face_crop.size == 0:
            return None, 0.0

        # --- Logic cho MobileFaceNetV2 ---
        if self.face_model_name == 'mobilefacenet':
            try:
                # Chuyển đổi BGR (OpenCV) -> RGB (PIL) và áp dụng transform
                rgb_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_crop)
                transformed_image = self.face_transform(pil_image).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    embedding = self.face_model(transformed_image)

                # Chuẩn hóa vector và trả về
                embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
                # Model này không cung cấp confidence, ta có thể trả về 1.0 mặc định
                return embedding.cpu().numpy().flatten().tolist(), 1.0 
            except Exception as e:
                # print(f"Lỗi khi trích xuất đặc trưng MobileFaceNet: {e}")
                return None, 0.0

        # --- Logic cho DeepFace ---
        elif self.face_model_name == 'deepface':
            try:
                # DeepFace.represent có thể trả về list các embedding
                # và có thể có thông tin về vùng mặt và confidence
                result = DeepFace.represent(
                    img_path=face_crop,
                    model_name=DEEPFACE_MODEL_NAME,
                    enforce_detection=True, # Bật enforce_detection để có confidence
                    detector_backend='ssd' # Chọn một backend cụ thể
                )
                # Lấy kết quả của khuôn mặt đầu tiên tìm thấy
                first_result = result[0]
                embedding = first_result['embedding']
                confidence = first_result['face_confidence']
                print("-------------------------------------------------------------")
                print(first_result['face_confidence'])
                return embedding, confidence
            except Exception:
                # Trả về None nếu DeepFace không tìm thấy khuôn mặt nào
                return None, 0.0

    # --- SỬA LỖI: Đã xóa hàm extract_face_feature1 không sử dụng ---
