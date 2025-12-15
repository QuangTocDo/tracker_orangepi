import asyncio
import cv2
import numpy as np
from sklearn.cluster import KMeans
from typing import Optional, List, Tuple, Dict, Any

# Giả định các import này đúng với cấu trúc dự án của bạn
from .clothing_classifier_by_color_new import ClothingClassifier
from .logging_python_orangepi import get_logger
from .mediapipe_pose import HumanDetection
from .mdp_face_detection_task import FaceDetection # <<< IMPORT MỚI
from .cut_body_part import extract_head_from_frame # <<< IMPORT MỚI
from shapely.geometry import Point, Polygon, LineString

logger = get_logger(__name__)

class PoseColorAnalyzer:
    """
    [PHIÊN BẢN TÁI CẤU TRÚC VÀ CẬP NHẬT LOGIC]
    - Thêm bước trích xuất màu da mặt làm ưu tiên.
    - Thêm logic phân tích ngữ cảnh tư thế: thay đổi cách trích xuất màu thân áo
      dựa trên vị trí của cẳng tay.
    - Sử dụng các hàm tiện ích từ HumanDetection để xác định vùng cơ thể.
    """

    def __init__(self, line_thickness: int = 30, k_clusters: int = 3):
        self.face_detector = FaceDetection(min_detection_confidence=0.5)  # <<< NHẬN FACE DETECTOR
        self.line_thickness = line_thickness
        self.k_clusters = k_clusters
        self.MIN_PIXELS_FOR_COLOR = 50
        self.MIN_PERCENTAGE = 7.0
        self.MERGE_THRESHOLD = 50.0
        self.MONOCHROMATIC_THRESHOLD = 80.0
        self.fps = 0.0
        self.frame_count = 0
        self.start_time = asyncio.get_event_loop().time() if asyncio.get_event_loop().is_running() else 0.0

    async def _get_face_skin_color(self, image: np.ndarray, keypoints: np.ndarray) -> Optional[List[int]]:
        """
        Phát hiện khuôn mặt và trích xuất màu da chủ đạo.
        Đây là "chứng cứ vàng" (ground truth) cho màu da.
        """
        if self.face_detector is None:
            return None
        
        try:
            # Sử dụng keypoints để cắt vùng đầu, tăng tốc độ và độ chính xác
            head_bbox = extract_head_from_frame(image, keypoints, scale=1.2)
            if head_bbox is None:
                return None

            x1, y1, x2, y2 = head_bbox
            head_roi = image[y1:y2, x1:x2]

            if head_roi.size == 0:
                return None
            
            # Chạy face detection trên vùng đầu đã cắt
            infos, _ = await asyncio.to_thread(self.face_detector.detect, head_roi)
            
            if not infos:
                return None
            
            # Lấy khuôn mặt có confidence cao nhất
            best_face = max(infos, key=lambda x: x.get('confidence', 0.0))
            bbox_obj = best_face.get('bbox')
            if not bbox_obj:
                return None

            # Cắt chính xác vùng mặt từ ROI đầu
            h_roi, w_roi = head_roi.shape[:2]
            fx1 = int(bbox_obj.xmin * w_roi)
            fy1 = int(bbox_obj.ymin * h_roi)
            fx2 = int((bbox_obj.xmin + bbox_obj.width) * w_roi)
            fy2 = int((bbox_obj.ymin + bbox_obj.height) * h_roi)
            
            face_pixels_roi = head_roi[fy1:fy2, fx1:fx2]

            if face_pixels_roi.size < self.MIN_PIXELS_FOR_COLOR:
                return None

            # Phân tích màu da mặt
            face_color_analysis = self.analyze_colors_simple(face_pixels_roi)
            if face_color_analysis:
                logger.info(f"Đã phát hiện màu da mặt: {face_color_analysis[0]['bgr']}")
                return face_color_analysis[0]['bgr']

        except Exception as e:
            logger.error(f"Lỗi khi trích xuất màu da mặt: {e}", exc_info=True)
        
        return None

    def _is_valid_point(self, point) -> bool:
        """Kiểm tra điểm keypoint có hợp lệ không."""
        return point is not None and len(point) >= 2 and point[0] != 0 and point[1] != 0

    def _is_forearm_inside_torso(self, arm_coords: Dict, torso_poly: Polygon) -> bool:
        """
        Kiểm tra xem đường thẳng từ khuỷu tay đến cổ tay có nằm trong đa giác thân không.
        """
        if not arm_coords or torso_poly is None or torso_poly.is_empty:
            return False
        try:
            p_elbow = Point(arm_coords['elbow']['x'], arm_coords['elbow']['y'])
            p_wrist = Point(arm_coords['wrist']['x'], arm_coords['wrist']['y'])
            forearm_line = LineString([p_elbow, p_wrist])
            # Trả về True nếu đường thẳng cẳng tay nằm hoàn toàn bên trong vùng thân
            return forearm_line.within(torso_poly)
        except Exception as e:
            logger.debug(f"Lỗi khi kiểm tra vị trí cẳng tay: {e}")
            return False

    def _get_pixels_from_polygon(self, image: np.ndarray, points: List[Tuple[int, int]]) -> Optional[np.ndarray]:
        """Trích xuất pixels từ vùng đa giác (dùng cho torso)."""
        if len(points) < 3:
            return None
        try:
            h, w = image.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillConvexPoly(mask, np.array(points, dtype=np.int32), 255)
            pixels = image[mask == 255]
            return pixels if len(pixels) >= self.MIN_PIXELS_FOR_COLOR else None
        except Exception as e:
            logger.debug(f"Lỗi trích xuất pixels từ polygon: {e}")
            return None

    def _get_pixels_from_line(self, image: np.ndarray, start_point: tuple, end_point: tuple) -> Optional[np.ndarray]:
        """Trích xuất pixels từ một đường thẳng có độ dày (dùng cho tay, chân)."""
        try:
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.line(mask, start_point, end_point, 255, self.line_thickness)
            pixels = image[mask == 255]
            return pixels if len(pixels) >= 10 else None
        except Exception:
            return None

    def _calculate_color_distance_lab(self, color1_bgr: np.ndarray, color2_bgr: np.ndarray) -> float:
        """Tính khoảng cách màu trong không gian LAB để so sánh màu sắc chính xác hơn."""
        try:
            color1_lab = cv2.cvtColor(np.uint8([[color1_bgr]]), cv2.COLOR_BGR2LAB)[0][0]
            color2_lab = cv2.cvtColor(np.uint8([[color2_bgr]]), cv2.COLOR_BGR2LAB)[0][0]
            return np.linalg.norm(color1_lab.astype(float) - color2_lab.astype(float))
        except:
            return float('inf')

    def _merge_similar_colors(self, colors: List[np.ndarray], percentages: List[float]) -> Tuple[List[np.ndarray], List[float]]:
        """Gộp các màu tương tự trong bảng màu để kết quả gọn hơn."""
        if len(colors) < 2:
            return colors, percentages
        merged_colors = colors.copy()
        merged_percentages = percentages.copy()
        i = 0
        while i < len(merged_colors):
            j = i + 1
            while j < len(merged_colors):
                distance = self._calculate_color_distance_lab(merged_colors[i], merged_colors[j])
                if distance < self.MERGE_THRESHOLD:
                    total_percentage = merged_percentages[i] + merged_percentages[j]
                    new_color = ((merged_colors[i] * merged_percentages[i] + merged_colors[j] * merged_percentages[j]) / total_percentage)
                    merged_colors[i] = new_color
                    merged_percentages[i] = total_percentage
                    merged_colors.pop(j)
                    merged_percentages.pop(j)
                    j -= 1
                j += 1
            i += 1
        return merged_colors, merged_percentages

    def analyze_colors_simple(self, pixels: np.ndarray) -> Optional[List[Dict]]:
        """Phân tích màu đơn giản - chỉ lấy màu chủ đạo nhất (tương đương K-Means n=1)."""
        if pixels is None or len(pixels) < 10: # Cần một lượng pixel tối thiểu
            return None
        try:
            # Sử dụng K-Means với k=1 để tìm màu trung tâm, ổn định hơn mean
            kmeans = KMeans(n_clusters=1, n_init=1, random_state=42)
            kmeans.fit(pixels)
            dominant_color = kmeans.cluster_centers_[0]
            return [{"bgr": dominant_color.astype(int).tolist(), "percentage": 100.0}]
        except Exception as e:
            logger.debug(f"Lỗi phân tích màu đơn giản với K-Means(1): {e}")
            return None

    def analyze_colors_advanced(self, pixels: np.ndarray) -> Optional[List[Dict]]:
        """Phân tích màu nâng cao bằng K-Means để trích xuất bảng màu."""
        if pixels is None or len(pixels) < self.k_clusters:
            return self.analyze_colors_simple(pixels)
        try:
            pixels_rgb = cv2.cvtColor(np.uint8([pixels]), cv2.COLOR_BGR2RGB)[0]
            k = max(2, min(self.k_clusters, len(pixels) // 20 if len(pixels) >= 40 else 2))
            
            kmeans = KMeans(n_clusters=k, n_init=4, random_state=42)
            kmeans.fit(pixels_rgb)
            
            total_pixels = len(kmeans.labels_)
            counts = np.bincount(kmeans.labels_)
            colors_rgb = kmeans.cluster_centers_
            colors_bgr = cv2.cvtColor(np.uint8([colors_rgb]), cv2.COLOR_RGB2BGR)[0]
            
            valid_colors, valid_percentages = [], []
            for i, count in enumerate(counts):
                percentage = (count / total_pixels) * 100
                if percentage >= self.MIN_PERCENTAGE:
                    valid_colors.append(colors_bgr[i])
                    valid_percentages.append(percentage)
            
            if not valid_colors:
                return self.analyze_colors_simple(pixels)
                
            merged_colors, merged_percentages = self._merge_similar_colors(valid_colors, valid_percentages)
            sorted_pairs = sorted(zip(merged_percentages, merged_colors), key=lambda x: x[0], reverse=True)
            
            if len(sorted_pairs) == 1 or sorted_pairs[0][0] > self.MONOCHROMATIC_THRESHOLD:
                return [{"bgr": sorted_pairs[0][1].astype(int).tolist(), "percentage": round(sorted_pairs[0][0], 2)}]
                
            return [{"bgr": color.astype(int).tolist(), "percentage": round(percentage, 2)} for percentage, color in sorted_pairs]
        except Exception as e:
            logger.debug(f"Lỗi phân tích màu nâng cao: {e}")
            return self.analyze_colors_simple(pixels)

    def _analyze_limb_segment(self, image: np.ndarray, p1: tuple, p2: tuple) -> Optional[List[Dict]]:
        """Hàm helper để phân tích màu một đoạn chi (tay/chân)."""
        if not self._is_valid_point(p1) or not self._is_valid_point(p2):
            return None
        pixels = self._get_pixels_from_line(image, p1, p2)
        return self.analyze_colors_advanced(pixels) if pixels is not None else None

    def _update_fps(self):
        """Cập nhật tính toán FPS.""" 
        self.frame_count += 1
        current_time = asyncio.get_event_loop().time()
        elapsed_time = current_time - self.start_time
        if elapsed_time >= 1:
            self.fps = self.frame_count / elapsed_time
            logger.info(f"FPS POSE_COLOR: {self.fps:.2f}")
            self.start_time = current_time
            self.frame_count = 0

    async def process_and_classify(
        self, 
        image: np.ndarray, 
        keypoints: np.ndarray,
        classifier: ClothingClassifier, 
        kpts_z: Optional[np.ndarray] = None,
        external_data: Optional[Dict] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Pipeline xử lý chính với logic trích xuất màu theo ngữ cảnh.
        """
        if external_data is None: external_data = {}
        try:
            # BƯỚC ƯU TIÊN: TRÍCH XUẤT MÀU DA MẶT
            face_skin_bgr = await self._get_face_skin_color(image, keypoints)
            if face_skin_bgr:
                external_data['face_skin_bgr'] = face_skin_bgr

            # GIAI ĐOẠN 1: THU THẬP DỮ LIỆU
            logger.debug("Bắt đầu xác định vùng cơ thể tốt nhất...")
            
            # <<< FIX: Chuyển đổi BBox (x1,y1,x2,y2) sang danh sách điểm cho Polygon
            torso_box_rect = HumanDetection.get_torso_box(keypoints)
            torso_poly = None
            if torso_box_rect:
                x1, y1, x2, y2 = torso_box_rect
                torso_poly = Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])

            _arm_side, arm_coords = HumanDetection.select_best_arm(keypoints, kpts_z)
            _leg_side, leg_coords = HumanDetection.select_best_leg(keypoints, kpts_z)

            # GIAI ĐOẠN 2.1: PHÂN TÍCH NGỮ CẢNH TƯ THẾ
            is_forearm_inside = self._is_forearm_inside_torso(arm_coords, torso_poly)
            logger.debug(f"Phân tích ngữ cảnh: Cẳng tay trong thân? {'Có' if is_forearm_inside else 'Không'}")

            # GIAI ĐOẠN 2.2: TRÍCH XUẤT MÀU SẮC SONG SONG (VỚI LOGIC ĐIỀU KIỆN)
            tasks = {}

            # Tác vụ Thân áo (Torso)
            if torso_poly and not torso_poly.is_empty:
                torso_pixels = self._get_pixels_from_polygon(image, list(torso_poly.exterior.coords))
                if is_forearm_inside:
                    tasks['torso'] = asyncio.to_thread(self.analyze_colors_simple, torso_pixels)
                else:
                    tasks['torso'] = asyncio.to_thread(self.analyze_colors_advanced, torso_pixels)

            # Tác vụ Cánh tay trên (Brachium)
            if arm_coords:
                p_shoulder = tuple(map(int, (arm_coords['shoulder']['x'], arm_coords['shoulder']['y'])))
                p_elbow = tuple(map(int, (arm_coords['elbow']['x'], arm_coords['elbow']['y'])))
                tasks['brachium'] = asyncio.to_thread(self._analyze_limb_segment, image, p_shoulder, p_elbow)
            
            # Tác vụ Cẳng tay (Forearm)
            if arm_coords:
                p_elbow = tuple(map(int, (arm_coords['elbow']['x'], arm_coords['elbow']['y'])))
                p_wrist = tuple(map(int, (arm_coords['wrist']['x'], arm_coords['wrist']['y'])))
                tasks['forearm'] = asyncio.to_thread(self._analyze_limb_segment, image, p_elbow, p_wrist)

            # Tác vụ Chân (Leg)
            if leg_coords:
                p_hip = tuple(map(int, (leg_coords['hip']['x'], leg_coords['hip']['y'])))
                p_knee = tuple(map(int, (leg_coords['knee']['x'], leg_coords['knee']['y'])))
                p_ankle = tuple(map(int, (leg_coords['ankle']['x'], leg_coords['ankle']['y'])))
                tasks['thigh'] = asyncio.to_thread(self._analyze_limb_segment, image, p_hip, p_knee)
                tasks['shin'] = asyncio.to_thread(self._analyze_limb_segment, image, p_knee, p_ankle)

            # Thực thi đồng thời các tác vụ và thu thập kết quả
            task_keys = list(tasks.keys())
            if not task_keys: return None
            
            results = await asyncio.gather(*tasks.values(), return_exceptions=True)

            # Tạo từ điển regional_analysis
            regional_analysis = { f"{key}_colors": res for key, res in zip(task_keys, results) if not isinstance(res, Exception)}

            # GIAI ĐOẠN 3: GỌI CLASSIFIER
            data_for_classifier = {**external_data, "regional_analysis": regional_analysis}
            classification_result = await asyncio.to_thread(classifier.classify, data_for_classifier)

            self._update_fps()

            # GIAI ĐOẠN 4: TRẢ KẾT QUẢ
            return {
                "classification": classification_result, 
                "raw_color_data": regional_analysis,
                "processing_info": {
                    "is_forearm_inside_torso": is_forearm_inside,
                    "face_skin_detected": face_skin_bgr is not None,
                    "best_arm_side": _arm_side,
                    "best_leg_side": _leg_side
                }
            }
        except Exception as e:
            logger.error(f"Lỗi trong pipeline phân tích màu: {e}", exc_info=True)
            return None

def create_analyzer(face_detector: FaceDetection, line_thickness: int = 30, k_clusters: int = 3) -> PoseColorAnalyzer:
    """Factory function để tạo analyzer với config mặc định."""
    return PoseColorAnalyzer(face_detector=face_detector, line_thickness=line_thickness, k_clusters=k_clusters)

