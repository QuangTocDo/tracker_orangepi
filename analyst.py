import torch
import torchreid
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
import os
from typing import Optional, List, Tuple

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, '..', 'models')

OSNET_MODEL_PATH = os.path.join(
    MODELS_DIR,
    "osnet_ain_x1_0_msmt17_256x128_amsgrad_ep50_lr0_0015_coslr_b64_fb10.pth"
)

MOBILEFACENET_MODEL_PATH = os.path.join(MODELS_DIR, "mobilefacenet.pt")

OSNET_INPUT_SIZE = (128, 256)
MOBILEFACENET_INPUT_SIZE = (112, 112)


class Analyzer:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"✅ Analyzer device: {self.device}")

        self._load_osnet()
        self._load_mobilefacenet()

    # ---------------- OSNET ----------------
    def _load_osnet(self):
        print("🔄 Loading OSNet...")
        self.osnet_model = torchreid.models.build_model(
            name='osnet_ain_x1_0',
            num_classes=1000,
            loss='softmax',
            pretrained=False
        )

        torchreid.utils.load_pretrained_weights(
            self.osnet_model, OSNET_MODEL_PATH
        )

        self.osnet_model.to(self.device).eval()

        _, self.osnet_transform = torchreid.data.transforms.build_transforms(
            height=OSNET_INPUT_SIZE[1],
            width=OSNET_INPUT_SIZE[0],
            is_train=False
        )

        print("✅ OSNet loaded")

    # ---------------- FACE ----------------
    def _load_mobilefacenet(self):
        print("🔄 Loading MobileFaceNet...")
        self.face_model = torch.jit.load(MOBILEFACENET_MODEL_PATH)
        self.face_model.to(self.device).eval()

        self.face_transform = transforms.Compose([
            transforms.Resize(MOBILEFACENET_INPUT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5]
            )
        ])

        print("✅ MobileFaceNet loaded")

    # ---------------- REID ----------------
    def extract_reid_feature(
        self, person_crop: np.ndarray
    ) -> Optional[List[float]]:

        if person_crop is None or person_crop.size == 0:
            return None

        rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        img = self.osnet_transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            feat = self.osnet_model(img)
            feat = torch.nn.functional.normalize(feat, p=2, dim=1)

        return feat.cpu().numpy().flatten().tolist()

    # ---------------- FACE ----------------
    def extract_face_feature(
        self, face_crop: np.ndarray
    ) -> Tuple[Optional[List[float]], float]:

        if face_crop is None or face_crop.size == 0:
            return None, 0.0

        rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        img = self.face_transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            feat = self.face_model(img)
            feat = torch.nn.functional.normalize(feat, p=2, dim=1)

        return feat.cpu().numpy().flatten().tolist(), 1.0
