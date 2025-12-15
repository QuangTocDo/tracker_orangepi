import cv2
import numpy as np
import csv
from typing import Optional, List, Dict, Any

from .logging_python_orangepi import get_logger
logger = get_logger(__name__)

class ClothingClassifier:
    """
    [PHIÊN BẢN CẬP NHẬT - LOGIC PHÂN LOẠI MỚI THEO LUỒNG]
    - Áp dụng cây quyết định chặt chẽ để phân loại áo.
    - Ưu tiên sử dụng màu da mặt làm "chứng cứ vàng".
    - Xử lý các trường hợp phức tạp như áo dài tay hai màu hoặc tay trần.
    - Cung cấp lý do phân loại chi tiết cho từng quyết định.
    """
    def __init__(
        self,
        skin_csv_path: str,
        sleeve_color_similarity_threshold: float = 45.0,
        pants_color_similarity_threshold: float = 90.0,
        skin_similarity_threshold: float = 35.0
    ):
        self.sleeve_threshold = sleeve_color_similarity_threshold
        self.pants_threshold = pants_color_similarity_threshold
        self.skin_threshold = skin_similarity_threshold # Ngưỡng chặt hơn khi so sánh với da
        self.skin_tone_palette = self._load_skin_tone_palette(skin_csv_path)

    def _load_skin_tone_palette(self, csv_path: str) -> List[Dict]:
        # (Không thay đổi)
        palette = []
        try:
            with open(csv_path, mode='r', encoding='utf-8') as infile:
                reader = csv.DictReader(infile)
                for row in reader:
                    try:
                        bgr_color = (int(row['b']), int(row['g']), int(row['r']))
                        palette.append({'id': int(row['id']), 'bgr': bgr_color})
                    except (ValueError, KeyError):
                        continue
        except Exception as e:
            logger.error(f"Không thể tải skin palette từ {csv_path}: {e}")
        if not palette: # Fallback
            return [{'id': 1, 'bgr': (145, 169, 210)}]
        return palette

    def _are_colors_similar(self, c1_bgr: List[int], c2_bgr: List[int], threshold: float) -> bool:
        if c1_bgr is None or c2_bgr is None:
            return False
        try:
            lab1 = cv2.cvtColor(np.uint8([[c1_bgr]]), cv2.COLOR_BGR2LAB)[0][0]
            lab2 = cv2.cvtColor(np.uint8([[c2_bgr]]), cv2.COLOR_BGR2LAB)[0][0]
            dist = np.linalg.norm(lab1.astype(float) - lab2.astype(float))
            return dist < threshold
        except cv2.error:
            return False

    def _find_closest_skin_tone(self, bgr_color: List[int]) -> tuple:
        # (Không thay đổi)
        if bgr_color is None:
            return None, None
        lab_det = cv2.cvtColor(np.uint8([[bgr_color]]), cv2.COLOR_BGR2LAB)[0][0]
        best, best_dist = None, float('inf')
        for tone in self.skin_tone_palette:
            lab_pal = cv2.cvtColor(np.uint8([[tone['bgr']]]), cv2.COLOR_BGR2LAB)[0][0]
            d = np.linalg.norm(lab_det.astype(float) - lab_pal.astype(float))
            if d < best_dist:
                best_dist, best = d, tone
        return best['id'], best['bgr']

    def _classify_pants_type(self, regional_data: Dict) -> Dict[str, Any]:
        # (Không thay đổi)
        thigh_colors = regional_data.get("thigh_colors")
        shin_colors = regional_data.get("shin_colors")

        if not thigh_colors or not shin_colors:
            return {"pants_type": "CHUA XAC DINH", "reason": "Thiếu dữ liệu màu ở chân."}

        main_thigh = thigh_colors[0]['bgr']
        main_shin = shin_colors[0]['bgr']

        if self._are_colors_similar(main_thigh, main_shin, self.pants_threshold):
            return {"pants_type": "QUAN DAI", "reason": "Màu đùi và ống quần giống nhau."}
        else:
            return {"pants_type": "QUAN NGAN", "reason": "Màu đùi và ống quần khác nhau."}

    def _classify_sleeve_type(self, regional_data: Dict, face_skin_bgr: Optional[List[int]]) -> Dict[str, Any]:
        """Thực hiện logic phân loại áo theo cây quyết định mới."""
        torso_colors = regional_data.get("torso_colors")
        brachium_colors = regional_data.get("brachium_colors")
        forearm_colors = regional_data.get("forearm_colors")

        main_torso = torso_colors[0]['bgr'] if torso_colors else None
        main_brachium = brachium_colors[0]['bgr'] if brachium_colors else None
        main_forearm = forearm_colors[0]['bgr'] if forearm_colors else None

        if not all([main_torso, main_brachium, main_forearm]):
            return {"sleeve_type": "CHUA XAC DINH", "reason": "Thiếu dữ liệu màu ở thân hoặc tay."}

        # BƯỚC 3.1: So sánh Cẳng tay và Thân áo
        if self._are_colors_similar(main_forearm, main_torso, self.sleeve_threshold):
            return {"sleeve_type": "AO DAI TAY", "reason": "Màu cẳng tay trùng khớp với màu thân áo."}

        # BƯỚC 3.2: Phân nhánh khi Cẳng tay khác Thân áo
        is_arm_uniform_color = self._are_colors_similar(main_forearm, main_brachium, self.sleeve_threshold)

        if not is_arm_uniform_color:
            # TRƯỜNG HỢP A: Cẳng tay và Cánh tay trên KHÁC MÀU
            if face_skin_bgr and self._are_colors_similar(main_forearm, face_skin_bgr, self.skin_threshold):
                return {"sleeve_type": "AO NGAN TAY", "reason": "Cẳng tay khác màu cánh tay trên và khớp với màu da mặt."}
            else:
                return {"sleeve_type": "AO DAI TAY", "reason": "Áo dài tay có hai màu khác nhau."}
        else:
            # TRƯỜNG HỢP B: Cẳng tay và Cánh tay trên GIỐNG MÀU (toàn bộ tay đồng màu)
            sleeve_uniform_color = main_forearm
            if face_skin_bgr and self._are_colors_similar(sleeve_uniform_color, face_skin_bgr, self.skin_threshold):
                return {"sleeve_type": "AO NGAN TAY", "reason": "Toàn bộ cánh tay đồng màu và khớp với màu da mặt."}
            else:
                return {"sleeve_type": "AO DAI TAY", "reason": "Áo dài tay đồng màu, khác màu da."}
    
    def classify(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Hàm phân loại chính, điều phối toàn bộ logic mới.
        """
        regional_data = analysis_data.get("regional_analysis", {})
        # Lấy "chứng cứ vàng" về màu da từ dữ liệu bên ngoài (nếu có)
        face_skin_bgr = analysis_data.get("face_skin_bgr")

        # Phân loại quần (logic không đổi)
        pants_result = self._classify_pants_type(regional_data)
        
        # Phân loại áo (logic mới)
        sleeve_result = self._classify_sleeve_type(regional_data, face_skin_bgr)

        # Xử lý màu da cuối cùng
        final_skin_tone_bgr = face_skin_bgr
        
        # Logic fallback: Nếu là áo ngắn tay và không có da mặt, màu da là màu cẳng tay
        if sleeve_result.get("sleeve_type") == "AO NGAN TAY" and face_skin_bgr is None:
            forearm_colors = regional_data.get("forearm_colors")
            if forearm_colors:
                final_skin_tone_bgr = forearm_colors[0].get("bgr")
                # Cập nhật lý do để rõ ràng hơn
                sleeve_result["reason"] += " (Màu da được suy ra từ cẳng tay)"
        
        skin_id, skin_bgr_from_palette = self._find_closest_skin_tone(final_skin_tone_bgr)
        
        # GIAI ĐOẠN 4: TỔNG HỢP KẾT QUẢ
        return {
            "sleeve_type": sleeve_result.get("sleeve_type", "CHUA XAC DINH"),
            "pants_type": pants_result.get("pants_type", "CHUA XAC DINH"),
            "skin_tone_bgr": skin_bgr_from_palette, # Trả về màu da từ palette chuẩn
            "reason": sleeve_result.get("reason", "Không có đủ thông tin."),
            "raw_colors": regional_data
        }
