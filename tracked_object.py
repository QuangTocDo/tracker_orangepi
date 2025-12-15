# tracker/tracked_object.py
from collections import deque, Counter
import config

class TrackedObject:
    """
    Một lớp để lưu trữ và quản lý toàn bộ trạng thái và thông tin
    của một đối tượng được theo dõi.
    """
    def __init__(self, track_id, bbox):
        # --- THUỘC TÍNH RE-ID & TRACKING ---
        self.track_id = track_id
        self.status = 'pending'  # pending -> tentative -> identified/confirmed
        self.final_id = f"Temp_{track_id}"
        self.bbox = bbox
        self.reid_vectors = deque(maxlen=config.MOVING_AVERAGE_WINDOW)
        self.face_vectors = deque(maxlen=config.MOVING_AVERAGE_WINDOW)
        self.disappeared_frames = 0
        self.quality_score = 0.0
        self.identification_score = 0.0
        self.identification_source = None
        self.db_face_vector_count = 0

        # --- THUỘC TÍNH PHÂN TÍCH (MỚI) ---
        self.is_identified = False # Cờ để dừng phân tích thuộc tính
        self.frames_since_analysis = 0 # Bộ đếm để điều tiết tần suất phân tích

        # Lưu trữ lịch sử kết quả phân tích thô từ mỗi khung hình
        self.attribute_history = {
            'gender': [],
            'sleeve_type': [],
            'pants_type': [],
            'brachium_colors': [], # Màu sắc cánh tay/áo
            'thigh_colors': [],    # Màu sắc đùi/quần
            'skin_tone_bgr': []
        }
        # Lưu trữ thuộc tính cuối cùng sau khi đã "chốt" để hiển thị
        self.display_attributes = {}

    def _get_most_common(self, items):
        """Tìm phần tử xuất hiện nhiều nhất trong một danh sách."""
        if not items:
            return None
        return Counter(items).most_common(1)[0][0]

    def _get_dominant_colors(self, color_list_of_lists):
        """Lấy danh sách các màu chiếm ưu thế từ lịch sử."""
        if not color_list_of_lists:
            return []
        # Flatten the list of lists into a single list of color info dicts
        flat_list = [item for sublist in color_list_of_lists if sublist for item in sublist]
        if not flat_list:
            return []
        # For simplicity, we'll just return the color info from the last valid frame.
        # A more complex approach could involve clustering or averaging.
        return color_list_of_lists[-1]

    def finalize_attributes(self):
        """
        Tổng hợp từ attribute_history để tìm ra thuộc tính ổn định nhất.
        Hàm này được gọi một lần khi đối tượng được định danh.
        """
        # Xử lý các thuộc tính dạng chuỗi (string)
        self.display_attributes['gender'] = self._get_most_common(self.attribute_history['gender'])
        self.display_attributes['sleeve_type'] = self._get_most_common(self.attribute_history['sleeve_type'])
        self.display_attributes['pants_type'] = self._get_most_common(self.attribute_history['pants_type'])

        # Xử lý các thuộc tính màu sắc
        self.display_attributes['brachium_colors'] = self._get_dominant_colors(self.attribute_history['brachium_colors'])
        self.display_attributes['thigh_colors'] = self._get_dominant_colors(self.attribute_history['thigh_colors'])

        # Xử lý màu da (lấy mẫu cuối cùng)
        if self.attribute_history['skin_tone_bgr']:
            self.display_attributes['skin_tone_bgr'] = self.attribute_history['skin_tone_bgr'][-1]
        
        print(f"✅ [ID: {self.track_id}] -> {self.final_id}: Đã chốt thuộc tính: {self.display_attributes}")
        
        # Giải phóng bộ nhớ sau khi đã tổng hợp
        self.attribute_history.clear()