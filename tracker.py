# tracker.py
import numpy as np
from collections import deque, Counter
import config
import threading
from utils.logging_python_orangepi import logging

# --- THÊM MỚI: Import thư viện cần thiết cho voting và clustering ---
try:
    from sklearn.cluster import KMeans
except ImportError:
    print("LỖI: Thư viện scikit-learn chưa được cài đặt. Vui lòng chạy: pip install scikit-learn")
    exit()

class TrackManager:
    """
    Quản lý trạng thái của từng đối tượng, điều phối việc nhận dạng và phân tích thuộc tính.
    """
    def __init__(self, analyzer, db_manager):
        self.analyzer = analyzer
        self.db_manager = db_manager
        self.tracked_objects = {}
        self.next_person_id = 1
        self.id_lock = threading.Lock()

    def _find_dominant_color(self, colors, k=3):
        if not colors:
            return None
        
        pixels = np.array(colors)
        if len(pixels) < k:
            unique_colors, counts = np.unique(pixels, axis=0, return_counts=True)
            dominant = unique_colors[counts.argmax()]
            return dominant.tolist()

        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        kmeans.fit(pixels)
        
        unique_labels, counts = np.unique(kmeans.labels_, return_counts=True)
        dominant_cluster_label = unique_labels[counts.argmax()]
        
        dominant_color = kmeans.cluster_centers_[dominant_cluster_label]
        return dominant_color.astype(int).tolist()

    def _consolidate_attributes(self, track_id):
        if track_id not in self.tracked_objects: return
        obj_data = self.tracked_objects[track_id]
        history_deque = obj_data.get('history_attributes', deque(maxlen=100))

        history = []
        if obj_data['status'] in ['identified', 'confirmed']:
            history = list(history_deque)[-10:]
            print(f"🗳️  [ID: {track_id}] Re-vote trên {len(history)} mẫu gần nhất...")
        else:
            history = list(history_deque)
            print(f"🗳️  [ID: {track_id}] Bắt đầu vote lần đầu trên {len(history)} kết quả thuộc tính...")

        if not history:
            print(f"⚠️ [ID: {track_id}] Không có lịch sử thuộc tính để tổng hợp.")
            return

        genders, upper_types, lower_types = [], [], []
        upper_colors, lower_colors = [], []
        skin_colors_bgr = []
        ages, races = [], [] # List cho Age/Race

        for result in history:
            if result.get('status') != 'success': continue
            
            # 1. Thu thập Gender
            if result.get('gender_analysis'):
                genders.append(result['gender_analysis'].get('gender'))
            
            # 2. Thu thập Clothing (SỬA LỖI Ở ĐÂY)
            clothing_res = result.get('clothing_analysis')
            
            # --- KIỂM TRA QUAN TRỌNG: Nếu clothing_res là None thì bỏ qua ---
            if clothing_res: 
                classification = clothing_res.get('classification')
                if classification:
                    upper_types.append(classification.get('sleeve_type'))
                    lower_types.append(classification.get('pants_type'))
                    
                    skin_bgr = classification.get('skin_tone_bgr')
                    if skin_bgr is not None:
                        skin_colors_bgr.append(skin_bgr)

                raw_colors = clothing_res.get('raw_color_data')
                if raw_colors:
                    brachium_colors = raw_colors.get('brachium_colors')
                    if brachium_colors:
                        upper_colors.extend([c['bgr'] for c in brachium_colors if 'bgr' in c])

                    thigh_colors = raw_colors.get('thigh_colors')
                    if thigh_colors:
                        lower_colors.extend([c['bgr'] for c in thigh_colors if 'bgr' in c])

            # 3. Thu thập Age & Race (MỚI)
            age_race_res = result.get('age_race_analysis')
            if age_race_res:
                ages.append(age_race_res.get('age'))
                races.append(age_race_res.get('race'))

        # --- TỔNG HỢP KẾT QUẢ ---
        final_attributes = {}

        if genders:
            valid = [g for g in genders if g]
            if valid: final_attributes['gender'] = Counter(valid).most_common(1)[0][0]
        
        # Vote Age
        if ages:
            valid = [a for a in ages if a]
            if valid: final_attributes['age'] = Counter(valid).most_common(1)[0][0]

        # Vote Race
        if races:
            valid = [r for r in races if r]
            if valid: final_attributes['race'] = Counter(valid).most_common(1)[0][0]

        if upper_types:
            valid = [t for t in upper_types if t]
            if valid: final_attributes['upper_type'] = Counter(valid).most_common(1)[0][0]

        if lower_types:
            valid = [t for t in lower_types if t]
            if valid: final_attributes['lower_type'] = Counter(valid).most_common(1)[0][0]

        dom_upper = self._find_dominant_color(upper_colors)
        if dom_upper: final_attributes['upper_color'] = dom_upper[::-1]

        dom_lower = self._find_dominant_color(lower_colors)
        if dom_lower: final_attributes['lower_color'] = dom_lower[::-1]

        dom_skin = self._find_dominant_color(skin_colors_bgr, k=1)
        if dom_skin: final_attributes['skin_tone_bgr'] = dom_skin

        obj_data['final_attributes'] = final_attributes
        print(f"✅ [ID: {track_id}] Đã cập nhật thuộc tính cuối cùng: {final_attributes}")
    def _get_query_vector(self, vectors_deque):
        if not vectors_deque: return None
        return np.mean(np.array(list(vectors_deque)), axis=0).tolist()

    def _get_query_vector_face(self, face_vectors_deque):
        valid_vectors = [v for v, c in face_vectors_deque if c >= config.FACE_CONFIDENCE_THRESHOLD]
        if not valid_vectors: return None
        return np.mean(np.array(valid_vectors), axis=0).tolist()

    def _identify_or_register1(self, track_id):
        if track_id not in self.tracked_objects: return
        obj_data = self.tracked_objects[track_id]
        
        if obj_data['status'] in ['identified', 'confirmed']:
             return

        print(f"\n🚀 [ID: {track_id}] Đạt ngưỡng Re-ID! Bắt đầu nhận dạng...")
        reid_query_vector = self._get_query_vector(obj_data['reid_vectors'])
        face_query_vector = self._get_query_vector_face(obj_data['face_vectors'])

        face_match_result = None
        reid_match_result = None

        if face_query_vector:
            face_match_result = self.db_manager.search_vector_with_voting(config.FACE_NAMESPACE, face_query_vector)
        if reid_query_vector:
            reid_match_result = self.db_manager.search_vector_with_voting(config.REID_NAMESPACE, reid_query_vector)

        final_id, final_score, final_source = None, 0.0, "None"
        face_id, face_score = face_match_result if face_match_result else (None, 0.0)
        reid_id, reid_score = reid_match_result if reid_match_result else (None, 0.0)
        
        if face_id and reid_id and face_id == reid_id:
            final_id, final_score, final_source = face_id, max(face_score, reid_score), "MẶT + TOÀN THÂN"
        elif face_id and reid_id and face_id != reid_id:
            final_id, final_score, final_source = face_id, face_score, "MÂU THUẪN (Ưu tiên Mặt)"
            obj_data['status'] = 'tentative'
        elif face_id:
            final_id, final_score, final_source = face_id, face_score, "MẶT"
        elif reid_id:
            final_id, final_score, final_source = reid_id, reid_score, "TOÀN THÂN"

        if not final_id:
            print(f"❌ [ID: {track_id}] QUYẾT ĐỊNH: Không khớp. Đăng ký NGƯỜI MỚI.")
            with self.id_lock:
                new_id = f"Person_{self.next_person_id}"
                self.next_person_id += 1
            obj_data.update({'final_id': new_id, 'status': 'confirmed'})
            if list(obj_data['reid_vectors']): self.db_manager.add_vectors(config.REID_NAMESPACE, new_id, list(obj_data['reid_vectors']))
            valid_face_vectors = [v for v, c in obj_data['face_vectors'] if c >= config.FACE_CONFIDENCE_THRESHOLD]
            if valid_face_vectors: self.db_manager.add_vectors(config.FACE_NAMESPACE, new_id, valid_face_vectors)
            self._consolidate_attributes(track_id)
        else:
            obj_data.update({'final_id': final_id, 'identification_score': final_score, 'identification_source': final_source})
            if obj_data['status'] != 'tentative':
                obj_data['status'] = 'identified' if final_score >= config.STABLE_IDENTIFICATION_THRESHOLD else 'tentative'
            print(f"🏁 [ID: {track_id}] QUYẾT ĐỊNH CUỐI CÙNG: ID={final_id}, Status='{obj_data['status']}', Score={final_score:.2f}, Nguồn='{final_source}'")
            if obj_data['status'] in ['identified', 'confirmed']:
                self._consolidate_attributes(track_id)
    '''
        # def process_analysis_results(self, result_queue):
        #     while not result_queue.empty():
        #         track_id, reid_vec, face_vec, face_conf = result_queue.get()
        #         if track_id not in self.tracked_objects: continue
        #         obj_data = self.tracked_objects[track_id]

        #         if obj_data['status'] in ['pending', 'tentative']:
        #             score_to_add = 0.0
        #             if reid_vec:
        #                 obj_data['reid_vectors'].append(reid_vec)
        #                 score_to_add += config.BASE_REID_SCORE
        #             if face_vec:
        #                 obj_data['face_vectors'].append((face_vec, face_conf))
        #                 if face_conf >= 0.95: score_to_add += config.HIGH_CONF_FACE_SCORE
        #                 elif face_conf >= config.FACE_CONFIDENCE_THRESHOLD: score_to_add += config.MID_CONF_FACE_SCORE
                    
        #             if score_to_add > 0:
        #                 obj_data['quality_score'] += score_to_add
                    
        #             if obj_data['quality_score'] >= config.QUALITY_SCORE_THRESHOLD and obj_data['status'] == 'pending':
        #                 self._identify_or_register(track_id)

        # Trong file tracker.py
    '''
    
    def _identify_or_register(self, track_id):
        if track_id not in self.tracked_objects: return
        obj_data = self.tracked_objects[track_id]
        
        if obj_data['status'] in ['identified', 'confirmed']:
                return

        print(f"\n🚀 [ID: {track_id}] Đạt ngưỡng Re-ID! Bắt đầu nhận dạng...")
        reid_query_vector = self._get_query_vector(obj_data['reid_vectors'])
        face_query_vector = self._get_query_vector_face(obj_data['face_vectors'])

        face_match_result = None
        reid_match_result = None

        if face_query_vector:
            face_match_result = self.db_manager.search_vector_with_voting(config.FACE_NAMESPACE, face_query_vector)
        if reid_query_vector:
            reid_match_result = self.db_manager.search_vector_with_voting(config.REID_NAMESPACE, reid_query_vector)

        final_id, final_score, final_source = None, 0.0, "None"
        face_id, face_score = face_match_result if face_match_result else (None, 0.0)
        reid_id, reid_score = reid_match_result if reid_match_result else (None, 0.0)
        
        if face_id and reid_id and face_id == reid_id:
            final_id, final_score, final_source = face_id, max(face_score, reid_score), "MẶT + TOÀN THÂN"
        elif face_id and reid_id and face_id != reid_id:
            final_id, final_score, final_source = face_id, face_score, "MÂU THUẪN (Ưu tiên Mặt)"
            obj_data['status'] = 'tentative'
        elif face_id:
            final_id, final_score, final_source = face_id, face_score, "MẶT"
        elif reid_id:
            final_id, final_score, final_source = reid_id, reid_score, "TOÀN THÂN"

        # ===================================================================
        # ==========     BẮT ĐẦU ĐOẠN LOGIC KIỂM TRA TRÙNG LẶP MỚI    ==========
        # ===================================================================
        if final_id:
            # BƯỚC A: Lấy danh sách tất cả các ID đã được xác nhận của các track khác đang hoạt động
            active_ids_in_frame = set()
            for other_track_id, other_obj_data in self.tracked_objects.items():
                # Bỏ qua chính track đang được xét
                if other_track_id == track_id:
                    continue
                
                # Chỉ xét những track đã có ID ổn định
                other_final_id = other_obj_data.get('final_id')
                if other_obj_data['status'] in ['identified', 'confirmed']:
                    active_ids_in_frame.add(other_final_id)

            # BƯỚC B: Kiểm tra xem ID vừa nhận dạng được có bị trùng không
            if final_id in active_ids_in_frame:
                # BƯỚC C: Nếu bị trùng, từ chối kết quả và chờ nhận dạng lại
                print(f"🚫 [ID: {track_id}] XUNG ĐỘT! Kết quả nhận dạng '{final_id}' đã được gán cho một người khác trong khung hình. Tạm thời từ chối.")
                # Đặt trạng thái về 'tentative' để nó có cơ hội được nhận dạng lại ở các khung hình sau
                # mà không bị đăng ký ngay thành người mới.
                obj_data['status'] = 'tentative' 
                return # Thoát khỏi hàm, không làm gì thêm ở frame này.

        # ===================================================================
        # ==========     KẾT THÚC ĐOẠN LOGIC MỚI, PHẦN CÒN LẠI GIỮ NGUYÊN    ==========
        # ===================================================================

        if not final_id:
            print(f"❌ [ID: {track_id}] QUYẾT ĐỊNH: Không khớp. Đăng ký NGƯỜI MỚI.")
            with self.id_lock:
                new_id = f"Person_{self.next_person_id}"
                self.next_person_id += 1
            obj_data.update({'final_id': new_id, 'status': 'confirmed'})
            if list(obj_data['reid_vectors']): self.db_manager.add_vectors(config.REID_NAMESPACE, new_id, list(obj_data['reid_vectors']))
            valid_face_vectors = [v for v, c in obj_data['face_vectors'] if c >= config.FACE_CONFIDENCE_THRESHOLD]
            if valid_face_vectors: self.db_manager.add_vectors(config.FACE_NAMESPACE, new_id, valid_face_vectors)
            self._consolidate_attributes(track_id)
        else:
            obj_data.update({'final_id': final_id, 'identification_score': final_score, 'identification_source': final_source})
            if obj_data['status'] != 'tentative':
                obj_data['status'] = 'identified' if final_score >= config.STABLE_IDENTIFICATION_THRESHOLD else 'tentative'
            print(f"🏁 [ID: {track_id}] QUYẾT ĐỊNH CUỐI CÙNG: ID={final_id}, Status='{obj_data['status']}', Score={final_score:.2f}, Nguồn='{final_source}'")
            if obj_data['status'] in ['identified', 'confirmed']:
                self._consolidate_attributes(track_id)

    def process_analysis_results(self, result_queue, reid_times_list=None):
        while not result_queue.empty():
            track_id, reid_vec, face_vec, face_conf, dt  = result_queue.get()
            if reid_times_list is not None:
                reid_times_list.append(dt)
            if track_id not in self.tracked_objects: continue
            obj_data = self.tracked_objects[track_id]

            # 1. LOGIC CŨ: Xử lý cho các đối tượng đang chờ nhận dạng
            if obj_data['status'] in ['pending', 'tentative']:
                score_to_add = 0.0
                if reid_vec:
                    obj_data['reid_vectors'].append(reid_vec)
                    score_to_add += config.BASE_REID_SCORE
                if face_vec:
                    obj_data['face_vectors'].append((face_vec, face_conf))
                    if face_conf >= 0.95: score_to_add += config.HIGH_CONF_FACE_SCORE
                    elif face_conf >= config.FACE_CONFIDENCE_THRESHOLD: score_to_add += config.MID_CONF_FACE_SCORE
                
                if score_to_add > 0:
                    obj_data['quality_score'] += score_to_add
                
                if obj_data['quality_score'] >= config.QUALITY_SCORE_THRESHOLD and obj_data['status'] == 'pending':
                    self._identify_or_register(track_id)
            
            # 2. ✨ LOGIC LÀM GIÀU DỮ LIỆU ĐÚNG VỊ TRÍ ✨
            #    Xử lý cho các đối tượng đã được nhận dạng
            elif obj_data['status'] in ['identified', 'confirmed']:
                person_id = obj_data.get('final_id')
                if not person_id or person_id.startswith("Temp_"):
                    continue # Bỏ qua nếu chưa có ID cuối cùng

                # 2.1. Làm giàu cho vector MẶT
                if face_vec and face_conf >= config.HIGH_CONFIDENCE_THRESHOLD_FOR_ENRICHMENT:
                    # Đếm số vector mặt hiện có
                    current_face_count = self.db_manager.count_vectors_for_id(config.FACE_NAMESPACE, person_id)
                    
                    # Nếu chưa đạt ngưỡng tối đa, tiến hành thêm
                    if current_face_count < config.MAX_FACE_VECTORS_PER_PROFILE:
                        print(f"💎 [Làm giàu FACE] ID: {person_id}, Count: {current_face_count+1}/{config.MAX_FACE_VECTORS_PER_PROFILE}, Conf: {face_conf:.2f}")
                        self.db_manager.add_vectors(config.FACE_NAMESPACE, person_id, [face_vec])

                # 2.2. Làm giàu cho vector TOÀN THÂN (Re-ID)
                if reid_vec:
                    # Đếm số vector toàn thân hiện có
                    current_reid_count = self.db_manager.count_vectors_for_id(config.REID_NAMESPACE, person_id)

                    # Nếu chưa đạt ngưỡng tối đa, tiến hành thêm
                    if current_reid_count < config.MAX_REID_VECTORS_PER_PROFILE:
                        print(f"💎 [Làm giàu RE-ID] ID: {person_id}, Count: {current_reid_count+1}/{config.MAX_REID_VECTORS_PER_PROFILE}")
                        self.db_manager.add_vectors(config.REID_NAMESPACE, person_id, [reid_vec])
    def process_attribute_results(self, attribute_result_queue, attr_times_list=None):
        while not attribute_result_queue.empty():
            track_id, analysis_result, dt = attribute_result_queue.get()
            if attr_times_list is not None:
                attr_times_list.append(dt)
            if track_id in self.tracked_objects and analysis_result:
                self.tracked_objects[track_id]['history_attributes'].append(analysis_result)

    def update_tracks(self, track_ids, bboxes, frame, reid_task_queue, attribute_task_queue):
        current_track_ids = set(track_ids)
        
        for i, track_id in enumerate(track_ids):
            bbox = bboxes[i]
            if track_id not in self.tracked_objects:
                print(f"✨ [ID: {track_id}] Track mới xuất hiện.")
                self.tracked_objects[track_id] = {
                    'status': 'pending', 'final_id': f"Temp_{track_id}", 'bbox': bbox,
                    'reid_vectors': deque(maxlen=config.MOVING_AVERAGE_WINDOW),
                    'face_vectors': deque(maxlen=config.MOVING_AVERAGE_WINDOW),
                    'disappeared_frames': 0, 'quality_score': 0.0,
                    'identification_score': 0.0, 'identification_source': None,
                    'history_attributes': deque(maxlen=100),
                    'final_attributes': None,
                    'frames_since_last_attr_analysis': 4,
                    'frames_since_last_consolidation': 0,
                }
            
            obj_data = self.tracked_objects[track_id]
            obj_data['bbox'] = bbox
            obj_data['disappeared_frames'] = 0
            obj_data['frames_since_last_attr_analysis'] += 1
            if obj_data['status'] in ['identified', 'confirmed']:
                obj_data['frames_since_last_consolidation'] += 1

            should_send_task = False
            if obj_data['status'] in ['pending', 'tentative']:
                if obj_data['frames_since_last_attr_analysis'] >= 5:
                    should_send_task = True
            elif obj_data['status'] in ['identified', 'confirmed']:
                if obj_data['frames_since_last_attr_analysis'] >= 10:
                    should_send_task = True

            if should_send_task:
                attribute_task_queue.put((track_id, frame.copy(), bbox))
                obj_data['frames_since_last_attr_analysis'] = 0

            if obj_data['status'] in ['identified', 'confirmed'] and obj_data['frames_since_last_consolidation'] >= 50:
                self._consolidate_attributes(track_id)
                obj_data['frames_since_last_consolidation'] = 0

            if obj_data['status'] in ['pending', 'tentative']:
                x1, y1, x2, y2 = [int(coord) for coord in bbox]
                crop = frame[y1:y2, x1:x2]
                if crop.size > 0:
                    reid_task_queue.put((track_id, crop.copy()))

        disappeared_ids = set(self.tracked_objects.keys()) - current_track_ids
        for track_id in disappeared_ids:
            self.tracked_objects[track_id]['disappeared_frames'] += 1

        cleanup_ids = [tid for tid, data in self.tracked_objects.items() if data['disappeared_frames'] > config.MAX_DISAPPEARED_FRAMES]
        for tid in cleanup_ids:
            print(f"🗑️ [ID: {tid}] Track đã bị xóa do mất dấu quá lâu.")
            del self.tracked_objects[tid]
