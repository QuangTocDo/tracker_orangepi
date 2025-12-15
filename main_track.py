# import cv2
# import queue
# import threading
# import asyncio
# import sys
# import os
# import numpy as np
# from ultralytics import YOLO

# # --- Cấu hình đường dẫn ---
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# if BASE_DIR not in sys.path:
#     sys.path.append(BASE_DIR)
# # Thêm thư mục utils vào path nếu cần
# UTILS_DIR = os.path.join(BASE_DIR, 'utils')
# if UTILS_DIR not in sys.path:
#     sys.path.append(UTILS_DIR)

# import config
# from analyzer import Analyzer as ReidFaceAnalyzer
# from attributes_analyzer import AttributesAnalyzer
# from vector_database import VectorDatabase_Manager
# from tracker import TrackManager
# from draw import draw_tracked_objects

# def is_image_quality_good(image, min_size=(64, 128), blur_threshold=80.0):
#     if image is None or image.size == 0: return False
#     h, w, _ = image.shape
#     if w < min_size[0] or h < min_size[1]: return False
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     if cv2.Laplacian(gray, cv2.CV_64F).var() < blur_threshold: return False
#     return True

# def reid_face_worker(task_queue, result_queue, analyzer):
#     """Worker cho luồng 1: Trích xuất vector Re-ID và Face."""
#     print("🚀 Worker Re-ID/Face đã bắt đầu.")
#     while True:
#         try:
#             track_id, image_crop = task_queue.get(block=True)
#             if not is_image_quality_good(image_crop):
#                 task_queue.task_done()
#                 continue
#             reid_vector = analyzer.extract_reid_feature(image_crop)
#             face_vector, face_confidence = analyzer.extract_face_feature(image_crop)
#             result_queue.put((track_id, reid_vector, face_vector, face_confidence))
#             task_queue.task_done()
#         except Exception as e:
#             print(f"Lỗi trong worker Re-ID/Face: {e}")

# # <<< WORKER MỚI >>>
# def attribute_analysis_worker(task_queue, result_queue, analyzer):
#     """Worker cho luồng 2: Phân tích thuộc tính (giới tính, quần áo)."""
#     print("🚀 Worker phân tích thuộc tính đã bắt đầu.")
#     # Mỗi luồng cần có event loop asyncio riêng
#     loop = asyncio.new_event_loop()
#     asyncio.set_event_loop(loop)
    
#     while True:
#         try:
#             track_id, frame, bbox = task_queue.get(block=True)
#             # Chạy hàm async trong event loop của luồng này
#             analysis_result = loop.run_until_complete(
#                 analyzer.analyze_person_by_bbox(frame, bbox, track_id)
#             )
#             result_queue.put((track_id, analysis_result))
#             task_queue.task_done()
#         except Exception as e:
#             print(f"Lỗi trong worker phân tích thuộc tính: {e}")

# def main():
#     print("--- BẮT ĐẦU HỆ THỐNG TRACKING & PHÂN TÍCH TOÀN DIỆN ---")
#     db_manager = None
#     try:
#         # --- 1. Khởi tạo các thành phần ---
#         db_manager = VectorDatabase_Manager()
        
#         # Analyzer cho Luồng 1
#         reid_face_analyzer = ReidFaceAnalyzer(face_model_name="mobilefacenet")
        
#         # Analyzer cho Luồng 2
#         attributes_analyzer = AttributesAnalyzer()
#         print("Đang tải models cho phân tích thuộc tính, vui lòng đợi...")
#         asyncio.run(attributes_analyzer.load_models()) # Tải model bất đồng bộ
#         print("✅ Tải models thuộc tính hoàn tất.")

#         last_id = db_manager.get_max_person_id()
#         track_manager = TrackManager(reid_face_analyzer, db_manager)
#         track_manager.next_person_id = last_id + 1
#         print(f"✅ ID lớn nhất trong CSDL: {last_id}. ID tiếp theo: {last_id + 1}.")

#         # --- 2. Thiết lập xử lý đa luồng ---
#         # Hàng đợi cho luồng 1 (Re-ID)
#         reid_task_queue = queue.Queue(maxsize=100)
#         reid_result_queue = queue.Queue()
#         reid_worker = threading.Thread(
#             target=reid_face_worker,
#             args=(reid_task_queue, reid_result_queue, reid_face_analyzer),
#             daemon=True
#         )
#         reid_worker.start()

#         # Hàng đợi cho luồng 2 (Thuộc tính)
#         attribute_task_queue = queue.Queue(maxsize=50)
#         attribute_result_queue = queue.Queue()
#         attribute_worker = threading.Thread(
#             target=attribute_analysis_worker,
#             args=(attribute_task_queue, attribute_result_queue, attributes_analyzer),
#             daemon=True
#         )
#         attribute_worker.start()

#         # --- 3. Load model YOLO ---
#         model = YOLO(config.YOLO_MODEL_PATH)
#         results_generator = model.track(source=0, show=False, conf=0.5, verbose=False, iou=0.5, classes=[0], tracker=config.TRACKER_CONFIG_PATH, stream=True)

#         print("\n✅ Hệ thống đã sẵn sàng. Bắt đầu xử lý video...")
#         for result in results_generator:
#             frame = result.orig_img
#             track_ids, bboxes = [], []
#             if result.boxes.id is not None:
#                 original_track_ids = result.boxes.id.int().cpu().tolist()
#                 original_bboxes = result.boxes.xyxy.cpu().tolist()
#                 for track_id, bbox in zip(original_track_ids, original_bboxes):
#                     if (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) > config.MIN_BBOX_AREA and (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) < config.MAX_BBOX_AREA:
#                         track_ids.append(track_id)
#                         bboxes.append(bbox)

#             # --- Cập nhật và xử lý ---
#             # Luôn gọi hàm update để xử lý cả đối tượng xuất hiện và biến mất
#             track_manager.update_tracks(track_ids, bboxes, frame, reid_task_queue, attribute_task_queue)
            
#             # Xử lý kết quả từ các luồng worker
#             track_manager.process_analysis_results(reid_result_queue)
#             track_manager.process_attribute_results(attribute_result_queue)
            
#             # Vẽ kết quả
#             frame_with_drawings = draw_tracked_objects(frame, track_manager.tracked_objects)
#             cv2.imshow("Tracking & Analysis System", frame_with_drawings)

#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break
    
#     except Exception as e:
#         print(f"Lỗi nghiêm trọng trong luồng chính: {e}")
#     finally:
#         if db_manager:
#             db_manager.save_all_databases()
#         cv2.destroyAllWindows()
#         print("--- HỆ THỐNG ĐÃ DỪNG ---")

# if __name__ == "__main__":
#     main()


import cv2
import queue
import threading
import asyncio
import sys
import os
import numpy as np
from ultralytics import YOLO
import time  # Thư viện để đo thời gian
from collections import deque # Thư viện để lưu trữ lịch sử FPS

# --- Cấu hình đường dẫn ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)
# Thêm thư mục utils vào path nếu cần
UTILS_DIR = os.path.join(BASE_DIR, 'utils')
if UTILS_DIR not in sys.path:
    sys.path.append(UTILS_DIR)

import config
from analyzer import Analyzer as ReidFaceAnalyzer
from attributes_analyzer import AttributesAnalyzer
from vector_database import VectorDatabase_Manager
from tracker import TrackManager
from draw import draw_tracked_objects

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

# <<< WORKER MỚI >>>
def attribute_analysis_worker(task_queue, result_queue, analyzer):
    """Worker cho luồng 2: Phân tích thuộc tính (giới tính, quần áo)."""
    print("🚀 Worker phân tích thuộc tính đã bắt đầu.")
    # Mỗi luồng cần có event loop asyncio riêng
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    while True:
        try:
            track_id, frame, bbox = task_queue.get(block=True)
            # Chạy hàm async trong event loop của luồng này
            analysis_result = loop.run_until_complete(
                analyzer.analyze_person_by_bbox(frame, bbox, track_id)
            )
            result_queue.put((track_id, analysis_result))
            task_queue.task_done()
        except Exception as e:
            print(f"Lỗi trong worker phân tích thuộc tính: {e}")

def main():
    print("--- BẮT ĐẦU HỆ THỐNG TRACKING & PHÂN TÍCH TOÀN DIỆN ---")
    db_manager = None
    try:
        # --- 1. Khởi tạo các thành phần ---
        db_manager = VectorDatabase_Manager()
        
        # Analyzer cho Luồng 1
        reid_face_analyzer = ReidFaceAnalyzer(face_model_name="deepface")
        
        # Analyzer cho Luồng 2
        attributes_analyzer = AttributesAnalyzer()
        print("Đang tải models cho phân tích thuộc tính, vui lòng đợi...")
        asyncio.run(attributes_analyzer.load_models()) # Tải model bất đồng bộ
        print("✅ Tải models thuộc tính hoàn tất.")

        last_id = db_manager.get_max_person_id()
        track_manager = TrackManager(reid_face_analyzer, db_manager)
        track_manager.next_person_id = last_id + 1
        print(f"✅ ID lớn nhất trong CSDL: {last_id}. ID tiếp theo: {last_id + 1}.")

        # --- 2. Thiết lập xử lý đa luồng ---
        # Hàng đợi cho luồng 1 (Re-ID)
        reid_task_queue = queue.Queue(maxsize=100)
        reid_result_queue = queue.Queue()
        reid_worker = threading.Thread(
            target=reid_face_worker,
            args=(reid_task_queue, reid_result_queue, reid_face_analyzer),
            daemon=True
        )
        reid_worker.start()

        # Hàng đợi cho luồng 2 (Thuộc tính)
        attribute_task_queue = queue.Queue(maxsize=50)
        attribute_result_queue = queue.Queue()
        attribute_worker = threading.Thread(
            target=attribute_analysis_worker,
            args=(attribute_task_queue, attribute_result_queue, attributes_analyzer),
            daemon=True
        )
        attribute_worker.start()

        # --- 3. Load model YOLO ---
        model = YOLO(config.YOLO_MODEL_PATH)
        results_generator = model.track(source=0, show=False, conf=0.5, verbose=False, iou=0.5, classes=[0], tracker=config.TRACKER_CONFIG_PATH, stream=True)

        print("\n✅ Hệ thống đã sẵn sàng. Bắt đầu xử lý video...")

        # --- KHỞI TẠO BIẾN ĐO FPS ---
        # Sử dụng deque để tính trung bình trượt trên 30 khung hình gần nhất
        history_len = 30
        fps_data = {
            "overall": deque(maxlen=history_len),
            "yolo": deque(maxlen=history_len),
            "update_tracker": deque(maxlen=history_len),
            "process_results": deque(maxlen=history_len),
            "draw": deque(maxlen=history_len)
        }
        # Biến đếm thời gian đặc biệt cho YOLO generator
        last_yolo_time = time.perf_counter()
        
        for result in results_generator:
            # === BẮT ĐẦU ĐO THỜI GIAN TỔNG THỂ ===
            start_overall = time.perf_counter()

            # --- Đo thời gian cho YOLO ---
            current_time = time.perf_counter()
            fps_data["yolo"].append(current_time - last_yolo_time)
            last_yolo_time = current_time

            frame = result.orig_img
            track_ids, bboxes = [], []
            if result.boxes.id is not None:
                original_track_ids = result.boxes.id.int().cpu().tolist()
                original_bboxes = result.boxes.xyxy.cpu().tolist()
                for track_id, bbox in zip(original_track_ids, original_bboxes):
                    if (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) > config.MIN_BBOX_AREA and (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) < config.MAX_BBOX_AREA:
                        track_ids.append(track_id)
                        bboxes.append(bbox)

            # --- Đo thời gian Cập nhật Tracker ---
            start_update = time.perf_counter()
            track_manager.update_tracks(track_ids, bboxes, frame, reid_task_queue, attribute_task_queue)
            fps_data["update_tracker"].append(time.perf_counter() - start_update)
            
            # --- Đo thời gian Xử lý kết quả ---
            start_process = time.perf_counter()
            track_manager.process_analysis_results(reid_result_queue)
            track_manager.process_attribute_results(attribute_result_queue)
            fps_data["process_results"].append(time.perf_counter() - start_process)
            
            # --- Đo thời gian Vẽ ---
            start_draw = time.perf_counter()
            frame_with_drawings = draw_tracked_objects(frame, track_manager.tracked_objects)
            
            # Tính toán và hiển thị FPS
            avg_fps = {}
            for key, deq in fps_data.items():
                if len(deq) > 0:
                    # FPS = 1 / (thời gian xử lý trung bình)
                    avg_fps[key] = 1 / (sum(deq) / len(deq))
                else:
                    avg_fps[key] = 0

            y_pos = 30
            # Màu xanh lá
            text_color = (0, 255, 0)
            cv2.putText(frame_with_drawings, f"Overall FPS: {avg_fps['overall']:.2f}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)
            y_pos += 30
            cv2.putText(frame_with_drawings, f" - YOLO: {avg_fps['yolo']:.2f} FPS", (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
            y_pos += 25
            cv2.putText(frame_with_drawings, f" - Update Logic: {avg_fps['update_tracker']:.2f} FPS", (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
            y_pos += 25
            cv2.putText(frame_with_drawings, f" - Process Results: {avg_fps['process_results']:.2f} FPS", (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
            y_pos += 25
            cv2.putText(frame_with_drawings, f" - Draw: {avg_fps['draw']:.2f} FPS", (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
            
            fps_data["draw"].append(time.perf_counter() - start_draw)

            # === KẾT THÚC ĐO THỜI GIAN TỔNG THỂ ===
            fps_data["overall"].append(time.perf_counter() - start_overall)

            cv2.imshow("Tracking & Analysis System", frame_with_drawings)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except Exception as e:
        print(f"Lỗi nghiêm trọng trong luồng chính: {e}")
    finally:
        if db_manager:
            db_manager.save_all_databases() 
        cv2.destroyAllWindows()
        print("--- HỆ THỐNG ĐÃ DỪNG ---")

if __name__ == "__main__":
    main()