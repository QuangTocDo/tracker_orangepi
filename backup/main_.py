import cv2
import queue
import threading
import asyncio
import sys
import os
import numpy as np
import time
from collections import deque
import yaml
from types import SimpleNamespace
import torch

# --- CÁC IMPORT MỚI CHO RKNN VÀ TRACKER ---
try:
    from rknnlite.api import RKNNLite
    from ultralytics.trackers.bot_sort import BOTSORT
except ImportError as e:
    print(f"Lỗi import thư viện: {e}")
    print("Hãy chắc chắn bạn đã cài đặt các thư viện cần thiết (rknn-toolkit2, ultralytics, pyyaml, torch) và môi trường đã được kích hoạt.")
    sys.exit(1)


# --- CẤU HÌNH ĐƯỜNG DẪN (GIỮ NGUYÊN) ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)
UTILS_DIR = os.path.join(BASE_DIR, 'utils')
if UTILS_DIR not in sys.path:
    sys.path.append(UTILS_DIR)

# --- IMPORT CÁC MODULE HIỆN CÓ (GIỮ NGUYÊN) ---
import config
from analyzer import Analyzer as ReidFaceAnalyzer
from attributes_analyzer import AttributesAnalyzer
from vector_database import VectorDatabase_Manager
from tracker import TrackManager
from draw import draw_tracked_objects

# ===================================================================
# ==========     LỚP VÀ HÀM HỖ TRỢ CHO RKNN & TRACKER    ==========
# ===================================================================

class MockResults(SimpleNamespace):
    """
    Lớp giả lập đối tượng Results của Ultralytics để cung cấp đầu vào cho BoTSORT.
    """
    def __getitem__(self, idx):
        new_results = MockResults()
        new_results.orig_shape = self.orig_shape
        
        new_boxes = SimpleNamespace()
        new_boxes.xyxy = self.boxes.xyxy[idx]
        new_boxes.conf = self.boxes.conf[idx]
        new_boxes.cls = self.boxes.cls[idx]
        
        new_results.boxes = new_boxes
        new_results.conf = new_boxes.conf
        
        if hasattr(self, 'xywh'):
            new_results.xywh = self.xywh[idx]
        if hasattr(self, 'cls'):
            new_results.cls = self.cls[idx]
            
        return new_results
    
    def __len__(self):
        return len(self.boxes.xyxy)

def preprocess(frame, input_size=(640, 640)):
    """Chuẩn bị frame ảnh cho đầu vào của mô hình RKNN."""
    img = cv2.resize(frame, input_size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.expand_dims(img, axis=0)
    return img

def postprocess(outputs, orig_shape, conf_threshold=0.5, nms_threshold=0.6):
    """Giải mã đầu ra từ mô hình YOLO chạy trên RKNN."""
    orig_h, orig_w = orig_shape
    input_h, input_w = (640, 640)
    predictions = np.squeeze(outputs[0]).T
    
    class_scores = predictions[:, 4:]
    max_scores = np.max(class_scores, axis=1)
    class_ids = np.argmax(class_scores, axis=1)
    
    # Lọc theo class 'person' (class_id = 0) và ngưỡng tin cậy
    person_class_mask = class_ids == 0
    conf_mask = max_scores > conf_threshold
    valid_mask = person_class_mask & conf_mask
    
    valid_predictions = predictions[valid_mask]
    if len(valid_predictions) == 0:
        return [], [], []
        
    valid_scores = max_scores[valid_mask]
    valid_class_ids = class_ids[valid_mask]
    
    x, y, w, h = valid_predictions[:, 0], valid_predictions[:, 1], valid_predictions[:, 2], valid_predictions[:, 3]
    x_factor = orig_w / input_w
    y_factor = orig_h / input_h
    
    left = ((x - 0.5 * w) * x_factor)
    top = ((y - 0.5 * h) * y_factor)
    width = (w * x_factor)
    height = (h * y_factor)
    
    boxes = np.column_stack((left, top, width, height))
    indices = cv2.dnn.NMSBoxes(boxes.tolist(), valid_scores.tolist(), conf_threshold, nms_threshold)
    
    final_boxes, final_scores, final_class_ids = [], [], []
    if len(indices) > 0:
        for i in indices.flatten():
            l, t, w, h = boxes[i]
            final_boxes.append([int(l), int(t), int(l + w), int(t + h)])
            final_scores.append(valid_scores[i])
            final_class_ids.append(valid_class_ids[i])
            
    return final_boxes, final_scores, final_class_ids

# ===================================================================
# ========== CÁC HÀM WORKER ĐA LUỒNG (GIỮ NGUYÊN) =========
# ===================================================================

def is_image_quality_good(image, min_size=(64, 128), blur_threshold=80.0):
    if image is None or image.size == 0: return False
    h, w, _ = image.shape
    if w < min_size[0] or h < min_size[1]: return False
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if cv2.Laplacian(gray, cv2.CV_64F).var() < blur_threshold: return False
    return True

def reid_face_worker(task_queue, result_queue, analyzer):
    """Worker cho luồng 1: Trích xuất vector Re-ID và Face."""
    print("🚀 Worker Re-ID/Face đã bắt đầu.")
    while True:
        try:
            track_id, image_crop = task_queue.get(block=True)
            if not is_image_quality_good(image_crop):
                task_queue.task_done()
                continue
            reid_vector = analyzer.extract_reid_feature(image_crop)
            face_vector, face_confidence = analyzer.extract_face_feature(image_crop)
            result_queue.put((track_id, reid_vector, face_vector, face_confidence))
            task_queue.task_done()
        except Exception as e:
            print(f"Lỗi trong worker Re-ID/Face: {e}")

def attribute_analysis_worker(task_queue, result_queue, analyzer):
    """Worker cho luồng 2: Phân tích thuộc tính (giới tính, quần áo)."""
    print("🚀 Worker phân tích thuộc tính đã bắt đầu.")
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    while True:
        try:
            track_id, frame, bbox = task_queue.get(block=True)
            analysis_result = loop.run_until_complete(
                analyzer.analyze_person_by_bbox(frame, bbox, track_id)
            )
            result_queue.put((track_id, analysis_result))
            task_queue.task_done()
        except Exception as e:
            print(f"Lỗi trong worker phân tích thuộc tính: {e}")

# ===================================================================
# ==========              HÀM MAIN ĐÃ ĐƯỢC NÂNG CẤP            ==========
# ===================================================================

def main():
    print("--- BẮT ĐẦU HỆ THỐNG TRACKING & PHÂN TÍCH (RKNN Accelerated) ---")
    db_manager = None
    rknn = None
    cap = None
    try:
        # --- 1. Khởi tạo các thành phần logic (GIỮ NGUYÊN) ---
        db_manager = VectorDatabase_Manager()
        reid_face_analyzer = ReidFaceAnalyzer(face_model_name="deepface")
        attributes_analyzer = AttributesAnalyzer()
        print("Đang tải models cho phân tích thuộc tính, vui lòng đợi...")
        asyncio.run(attributes_analyzer.load_models())
        print("✅ Tải models thuộc tính hoàn tất.")

        last_id = db_manager.get_max_person_id()
        track_manager = TrackManager(reid_face_analyzer, db_manager)
        track_manager.next_person_id = last_id + 1
        print(f"✅ ID lớn nhất trong CSDL: {last_id}. ID tiếp theo: {last_id + 1}.")

        # --- 2. Thiết lập xử lý đa luồng (GIỮ NGUYÊN) ---
        reid_task_queue = queue.Queue(maxsize=200)
        reid_result_queue = queue.Queue()
        reid_worker = threading.Thread(
            target=reid_face_worker, args=(reid_task_queue, reid_result_queue, reid_face_analyzer), daemon=True
        )
        reid_worker.start()

        attribute_task_queue = queue.Queue(maxsize=100)
        attribute_result_queue = queue.Queue()
        attribute_worker = threading.Thread(
            target=attribute_analysis_worker, args=(attribute_task_queue, attribute_result_queue, attributes_analyzer), daemon=True
        )
        attribute_worker.start()

        # --- 3. THAY THẾ: Khởi tạo RKNN và Tracker BoTSORT ---
        if not os.path.exists(config.RKNN_MODEL_PATH):
            print(f"❌ Lỗi: Không tìm thấy file model RKNN tại: {config.RKNN_MODEL_PATH}")
            print("Vui lòng kiểm tra lại đường dẫn trong file config.py")
            sys.exit(1)
        if not os.path.exists(config.TRACKER_CONFIG_PATH):
            print(f"❌ Lỗi: Không tìm thấy file cấu hình tracker tại: {config.TRACKER_CONFIG_PATH}")
            sys.exit(1)
        
        rknn = RKNN()
        print("--> Đang tải model RKNN...")
        ret = rknn.load_rknn(config.RKNN_MODEL_PATH)
        if ret != 0:
            print(f"❌ Tải model RKNN thất bại! (Mã lỗi: {ret})")
            exit(ret)
        print("✅ Tải model thành công.")

        print("--> Khởi tạo RKNN runtime...")
        ret = rknn.init_runtime(target="rk3588") # Đảm bảo target phù hợp với phần cứng của bạn
        if ret != 0:
            print(f"❌ Khởi tạo runtime thất bại! (Mã lỗi: {ret})")
            exit(ret)
        print("✅ Khởi tạo runtime thành công.")

        print("--> Khởi tạo tracker BoTSORT...")
        with open(config.TRACKER_CONFIG_PATH, "r") as f:
            tracker_config = yaml.safe_load(f)
        args = SimpleNamespace(**tracker_config)
        tracker = BOTSORT(args=args, frame_rate=30)
        print("✅ Khởi tạo tracker thành công.")

        # --- 4. THAY THẾ: Mở nguồn video thủ công ---
        cap = cv2.VideoCapture(0) # Hoặc đường dẫn file video
        if not cap.isOpened():
            print("❌ Lỗi: Không thể mở camera hoặc nguồn video.")
            return

        print("\n✅ Hệ thống đã sẵn sàng. Bắt đầu xử lý video...")
        prev_time = 0
        # --- VÒNG LẶP CHÍNH ĐÃ ĐƯỢC TÁI CẤU TRÚC ---
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Lỗi: Không thể nhận frame. Kết thúc...")
                break

            # BƯỚC 1: SUY LUẬN VỚI RKNN
            img_in = preprocess(frame)
            outputs = rknn.inference(inputs=[img_in])
            
            # BƯỚC 2: HẬU XỬ LÝ KẾT QUẢ
            raw_bboxes, raw_scores, raw_cls_ids = postprocess(outputs, frame.shape[:2], conf_threshold=0.35)

            # BƯỚC 3: ĐÓNG GÓI KẾT QUẢ VÀ CẬP NHẬT TRACKER
            results = MockResults()
            results.orig_shape = frame.shape[:2]
            
            boxes_ns = SimpleNamespace()
            boxes_ns.xyxy = torch.tensor(raw_bboxes, dtype=torch.float32)
            boxes_ns.conf = torch.tensor(raw_scores, dtype=torch.float32)
            boxes_ns.cls = torch.tensor(raw_cls_ids, dtype=torch.float32)
            results.boxes = boxes_ns
            results.conf = boxes_ns.conf
            results.cls = boxes_ns.cls
            
            # Tracker yêu cầu định dạng xywh
            xyxy = results.boxes.xyxy
            if xyxy.numel() > 0:
                x1, y1, x2, y2 = xyxy[:, 0], xyxy[:, 1], xyxy[:, 2], xyxy[:, 3]
                w, h = x2 - x1, y2 - y1
                cx, cy = x1 + w / 2, y1 + h / 2
                results.xywh = torch.stack((cx, cy, w, h), dim=1)
            else:
                results.xywh = torch.empty((0, 4), dtype=torch.float32)
            
            online_targets = tracker.update(results, frame)

            # BƯỚC 4: CHUẨN BỊ DỮ LIỆU CHO CÁC LUỒNG PHÂN TÍCH
            track_ids, bboxes = [], []
            if online_targets is not None and len(online_targets) > 0:
                for target in online_targets:
                    x1, y1, x2, y2, track_id, score, cls_id = target[:7]
                    bbox = [int(x1), int(y1), int(x2), int(y2)]
                    # Lọc theo diện tích bbox như logic cũ
                    bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                    if config.MIN_BBOX_AREA < bbox_area < config.MAX_BBOX_AREA:
                        track_ids.append(int(track_id))
                        bboxes.append(bbox)

            # --- CÁC BƯỚC XỬ LÝ PHÍA SAU GIỮ NGUYÊN ---
            track_manager.update_tracks(track_ids, bboxes, frame, reid_task_queue, attribute_task_queue)
            track_manager.process_analysis_results(reid_result_queue)
            track_manager.process_attribute_results(attribute_result_queue)
            
            frame_with_drawings = draw_tracked_objects(frame, track_manager.tracked_objects)
            # --- TÍNH TOÁN VÀ HIỂN THỊ FPS ---
            current_time = time.time()
            # Thêm 1e-9 để tránh lỗi chia cho 0 ở frame đầu tiên
            fps = 1 / (current_time - prev_time + 1e-9) 
            prev_time = current_time

            # Chuẩn bị text để vẽ lên frame
            fps_text = f"FPS: {fps:.2f}"

            # Vẽ text FPS lên góc trên bên trái của frame đã có các hình vẽ
            cv2.putText(frame_with_drawings, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.imshow("Tracking & Analysis System (RKNN Accelerated)", frame_with_drawings)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except Exception as e:
        import traceback
        print(f"Lỗi nghiêm trọng trong luồng chính: {e}")
        traceback.print_exc()
    finally:
        # --- Giải phóng tài nguyên ---
        if cap:
            cap.release()
        if rknn:
            rknn.release()
        if db_manager:
            db_manager.save_all_databases()
        cv2.destroyAllWindows()
        print("--- HỆ THỐNG ĐÃ DỪNG ---")

if __name__ == "__main__":
    main()
