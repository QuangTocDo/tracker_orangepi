'''
# import cv2
# import queue
# import threading
# import asyncio
# import sys
# import os
# import numpy as np
# from ultralytics import YOLO
# import zmq # [THAY ĐỔI] Thêm thư viện ZMQ
# import time # [THÊM VÀO]
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

# # --- [THAY ĐỔI] Cấu hình ZMQ và Xử lý Frame ---
# ZMQ_IP = "localhost"
# ZMQ_PORT = 5555
# FRAME_QUEUE_MAX_SIZE = 100  # Giới hạn số frame trong hàng đợi
# FRAME_SKIP_RATE = 3        # Chỉ xử lý 1 trên mỗi 2 frame nhận được. Đặt là 1 để xử lý tất cả.

# # --- Các hàm worker và tiện ích (Không thay đổi) ---
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
#             track_id, image_crop = task_queue.get(block=True, timeout=1)
#             if not is_image_quality_good(image_crop):
#                 task_queue.task_done()
#                 continue
#             reid_vector = analyzer.extract_reid_feature(image_crop)
#             face_vector, face_confidence = analyzer.extract_face_feature(image_crop)
#             result_queue.put((track_id, reid_vector, face_vector, face_confidence))
#             task_queue.task_done()
#         except queue.Empty:
#             continue
#         except Exception as e:
#             print(f"Lỗi trong worker Re-ID/Face: {e}")

# def attribute_analysis_worker(task_queue, result_queue, analyzer):
#     """Worker cho luồng 2: Phân tích thuộc tính (giới tính, quần áo)."""
#     print("🚀 Worker phân tích thuộc tính đã bắt đầu.")
#     loop = asyncio.new_event_loop()
#     asyncio.set_event_loop(loop)
#     while True:
#         try:
#             track_id, frame, bbox = task_queue.get(block=True, timeout=1)
#             analysis_result = loop.run_until_complete(
#                 analyzer.analyze_person_by_bbox(frame, bbox, track_id)
#             )
#             result_queue.put((track_id, analysis_result))
#             task_queue.task_done()
#         except queue.Empty:
#             continue
#         except Exception as e:
#             print(f"Lỗi trong worker phân tích thuộc tính: {e}")

# # --- [THAY ĐỔI] Luồng nhận dữ liệu từ ZMQ ---
# def network_receiver_worker(context, stop_event, frame_queue):
#     """
#     Luồng chuyên nhận dữ liệu từ ZMQ và đẩy vào hàng đợi.
#     Tích hợp logic bỏ qua frame (frame skipping).
#     """
#     socket = context.socket(zmq.SUB)
#     socket.connect(f"tcp://{ZMQ_IP}:{ZMQ_PORT}")
#     socket.setsockopt_string(zmq.SUBSCRIBE, '')
    
#     print(f"✅ [Luồng Mạng] Đã kết nối tới tcp://{ZMQ_IP}:{ZMQ_PORT} và đang lắng nghe...")
    
#     frame_counter = 0
    
#     while not stop_event.is_set():
#         try:
#             if socket.poll(timeout=100): # Chờ 100ms
#                 image_bytes = socket.recv() # Chỉ nhận ảnh, không cần frame_id
#                 frame_counter += 1

#                 if frame_counter % FRAME_SKIP_RATE == 0:
#                     try:
#                         frame_queue.put_nowait(image_bytes)
#                     except queue.Full:
#                         print(f"⚠️ [Luồng Mạng] Hàng đợi frame đầy. Bỏ qua frame.")
        
#         except zmq.ZMQError as e:
#             print(f"❌ [Luồng Mạng] Lỗi ZMQ: {e}")
#             break
#         except Exception as e:
#             print(f"❌ [Luồng Mạng] Lỗi không xác định: {e}")
#             break
            
#     print("🛑 [Luồng Mạng] Đang dừng...")
#     socket.close()

# # --- [THAY ĐỔI] Hàm xử lý chính được tái cấu trúc từ hàm main() cũ ---
# def processing_loop(frame_queue, stop_event):
#     """
#     Luồng chuyên xử lý ảnh: lấy ảnh từ hàng đợi, chạy model,
#     và thực hiện các tác vụ phân tích.
#     """
#     print("--- [Luồng Xử Lý] BẮT ĐẦU HỆ THỐNG TRACKING & PHÂN TÍCH ---")
#     db_manager = None
#     try:
#         # --- 1. Khởi tạo các thành phần ---
#         db_manager = VectorDatabase_Manager()
#         reid_face_analyzer = ReidFaceAnalyzer(face_model_name="mobilefacenet")
#         attributes_analyzer = AttributesAnalyzer()
#         print("[Luồng Xử Lý] Đang tải models cho phân tích thuộc tính, vui lòng đợi...")
#         asyncio.run(attributes_analyzer.load_models())
#         print("✅ [Luồng Xử Lý] Tải models thuộc tính hoàn tất.")

#         last_id = db_manager.get_max_person_id()
#         track_manager = TrackManager(reid_face_analyzer, db_manager)
#         track_manager.next_person_id = last_id + 1
#         print(f"✅ ID lớn nhất trong CSDL: {last_id}. ID tiếp theo: {last_id + 1}.")

#         # --- 2. Thiết lập xử lý đa luồng cho các worker phân tích ---
#         reid_task_queue = queue.Queue(maxsize=100)
#         reid_result_queue = queue.Queue()
#         reid_worker = threading.Thread(target=reid_face_worker, args=(reid_task_queue, reid_result_queue, reid_face_analyzer), daemon=True)
#         reid_worker.start()

#         attribute_task_queue = queue.Queue(maxsize=50)
#         attribute_result_queue = queue.Queue()
#         attribute_worker = threading.Thread(target=attribute_analysis_worker, args=(attribute_task_queue, attribute_result_queue, attributes_analyzer), daemon=True)
#         attribute_worker.start()

#         # --- 3. Load model YOLO ---
#         model = YOLO(config.YOLO_MODEL_PATH)
        
#         print("\n✅ [Luồng Xử Lý] Hệ thống đã sẵn sàng. Chờ frame từ ZMQ...")
        
#         # --- [THAY ĐỔI] Vòng lặp chính: Lấy frame từ queue và xử lý ---
#         while not stop_event.is_set():
#             try:
#                 # Lấy frame từ hàng đợi, có timeout để không bị block mãi mãi
#                 image_bytes = frame_queue.get(timeout=1.0)
                
#                 # Giải mã ảnh
#                 np_array = np.frombuffer(image_bytes, dtype=np.uint8)
#                 frame = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

#                 if frame is None:
#                     print("⚠️ [Luồng Xử Lý] Không thể giải mã frame, bỏ qua.")
#                     continue

#                 # Chạy model trên một frame duy nhất
#                 results = model.track(source=frame, show=False, conf=0.5, verbose=False, iou=0.5, classes=[0], tracker=config.TRACKER_CONFIG_PATH, stream=False, persist=True)
                
#                 # Vì chỉ xử lý 1 frame, vòng lặp này thực chất chỉ chạy 1 lần
#                 for result in results:
#                     track_ids, bboxes = [], []
#                     if result.boxes.id is not None:
#                         original_track_ids = result.boxes.id.int().cpu().tolist()
#                         original_bboxes = result.boxes.xyxy.cpu().tolist()
#                         for track_id, bbox in zip(original_track_ids, original_bboxes):
#                             if (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) > config.MIN_BBOX_AREA and (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) < config.MAX_BBOX_AREA:
#                                 track_ids.append(track_id)
#                                 bboxes.append(bbox)

#                     track_manager.update_tracks(track_ids, bboxes, frame, reid_task_queue, attribute_task_queue)
#                     track_manager.process_analysis_results(reid_result_queue)
#                     track_manager.process_attribute_results(attribute_result_queue)
                    
#                     frame_with_drawings = draw_tracked_objects(frame, track_manager.tracked_objects)
#                     cv2.imshow("Tracking & Analysis System", frame_with_drawings)

#                 if cv2.waitKey(1) & 0xFF == ord('q'):
#                     print("ℹ️ Nhấn 'q', gửi tín hiệu dừng...")
#                     stop_event.set() # Gửi tín hiệu dừng cho tất cả các luồng
#                     break

#                 frame_queue.task_done()

#             except queue.Empty:
#                 # Không có frame nào trong hàng đợi, tiếp tục vòng lặp
#                 continue
#             except Exception as e:
#                 print(f"❌ [Luồng Xử Lý] Lỗi nghiêm trọng: {e}")
#                 stop_event.set()
#                 break

#     finally:
#         if db_manager:
#             db_manager.save_all_databases()
#         cv2.destroyAllWindows()
#         print("--- [Luồng Xử Lý] HỆ THỐNG ĐÃ DỪNG ---")


# # --- [THAY ĐỔI] Hàm main() mới: Điều phối các luồng ---
# def main():
#     """
#     Hàm main chính để khởi tạo và quản lý các luồng.
#     """
#     frame_queue = queue.Queue(maxsize=FRAME_QUEUE_MAX_SIZE)
#     stop_event = threading.Event()
#     zmq_context = zmq.Context()

#     # Tạo và khởi chạy các luồng
#     network_t = threading.Thread(target=network_receiver_worker, args=(zmq_context, stop_event, frame_queue), daemon=True)
#     processing_t = threading.Thread(target=processing_loop, args=(frame_queue, stop_event), daemon=True)
    
#     print("🚀 Bắt đầu khởi chạy các luồng...")
#     network_t.start()
#     processing_t.start()

#     try:
#         # Giữ luồng chính sống để bắt tín hiệu Ctrl+C
#         # Hoặc chờ cho đến khi luồng xử lý kết thúc (do nhấn 'q')
#         processing_t.join()
#     except KeyboardInterrupt:
#         print("ℹ️ Bắt được tín hiệu Ctrl+C, gửi tín hiệu dừng...")
#         stop_event.set()

#     # Đảm bảo các luồng đã dừng hoàn toàn
#     network_t.join(timeout=2)
    
#     # Dọn dẹp
#     zmq_context.term()
#     print("✅ Tất cả các luồng đã kết thúc. Thoát chương trình.")


# if __name__ == "__main__":
#     main()
'''
import cv2
import queue
import threading
import asyncio
import sys
import os
import numpy as np
from ultralytics import YOLO
import zmq
import time

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

# --- Cấu hình ZMQ và Xử lý Frame ---
ZMQ_IP = "localhost"
ZMQ_PORT = 5555
FRAME_QUEUE_MAX_SIZE = 100
FRAME_SKIP_RATE = 10 # Đặt là 1 để xử lý tất cả frame cho việc đo FPS chính xác

# --- Các hàm worker và tiện ích (Không thay đổi) ---
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
            track_id, image_crop = task_queue.get(block=True, timeout=1)
            if not is_image_quality_good(image_crop):
                task_queue.task_done()
                continue
            reid_vector = analyzer.extract_reid_feature(image_crop)
            face_vector, face_confidence = analyzer.extract_face_feature(image_crop)
            result_queue.put((track_id, reid_vector, face_vector, face_confidence))
            task_queue.task_done()
        except queue.Empty:
            continue
        except Exception as e:
            print(f"Lỗi trong worker Re-ID/Face: {e}")

def attribute_analysis_worker(task_queue, result_queue, analyzer):
    """Worker cho luồng 2: Phân tích thuộc tính (giới tính, quần áo)."""
    print("🚀 Worker phân tích thuộc tính đã bắt đầu.")
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    while True:
        try:
            track_id, frame, bbox = task_queue.get(block=True, timeout=1)
            analysis_result = loop.run_until_complete(
                analyzer.analyze_person_by_bbox(frame, bbox, track_id)
            )
            result_queue.put((track_id, analysis_result))
            task_queue.task_done()
        except queue.Empty:
            continue
        except Exception as e:
            print(f"Lỗi trong worker phân tích thuộc tính: {e}")

# --- Luồng nhận dữ liệu từ ZMQ ---
def network_receiver_worker(context, stop_event, frame_queue):
    """
    Luồng chuyên nhận dữ liệu từ ZMQ và đẩy vào hàng đợi.
    Tích hợp logic bỏ qua frame (frame skipping) và đo FPS mạng.
    """
    socket = context.socket(zmq.SUB)
    socket.connect(f"tcp://{ZMQ_IP}:{ZMQ_PORT}")
    socket.setsockopt_string(zmq.SUBSCRIBE, '')
    
    print(f"✅ [Luồng Mạng] Đã kết nối tới tcp://{ZMQ_IP}:{ZMQ_PORT} và đang lắng nghe...")
    
    frame_counter = 0
    # [THÊM VÀO] Khởi tạo biến để đo FPS mạng
    fps_start_time = time.time()
    fps_frame_count = 0
    
    while not stop_event.is_set():
        try:
            if socket.poll(timeout=100):
                image_bytes = socket.recv()
                
                # [THÊM VÀO] Tăng biến đếm FPS mạng
                fps_frame_count += 1
                
                frame_counter += 1
                if frame_counter % FRAME_SKIP_RATE == 0:
                    try:
                        frame_queue.put_nowait(image_bytes)
                    except queue.Full:
                        print(f"⚠️ [Luồng Mạng] Hàng đợi frame đầy. Bỏ qua frame.")
            
            # [THÊM VÀO] Tính toán và log FPS mạng mỗi giây
            elapsed_time = time.time() - fps_start_time
            if elapsed_time >= 1.0:
                network_fps = fps_frame_count / elapsed_time
                print(f"📊 [Luồng Mạng] FPS Nhận: {network_fps:.2f}")
                # Reset để đo cho giây tiếp theo
                fps_start_time = time.time()
                fps_frame_count = 0

        except zmq.ZMQError as e:
            print(f"❌ [Luồng Mạng] Lỗi ZMQ: {e}")
            break
        except Exception as e:
            print(f"❌ [Luồng Mạng] Lỗi không xác định: {e}")
            break
            
    print("🛑 [Luồng Mạng] Đang dừng...")
    socket.close()

# --- Hàm xử lý chính ---
def processing_loop(frame_queue, stop_event):
    """
    Luồng chuyên xử lý ảnh: lấy ảnh từ hàng đợi, chạy model,
    thực hiện các tác vụ phân tích và đo FPS xử lý.
    """
    print("--- [Luồng Xử Lý] BẮT ĐẦU HỆ THỐNG TRACKING & PHÂN TÍCH ---")
    db_manager = None
    try:
        # --- 1. Khởi tạo các thành phần ---
        db_manager = VectorDatabase_Manager()
        reid_face_analyzer = ReidFaceAnalyzer(face_model_name="mobilefacenet")
        attributes_analyzer = AttributesAnalyzer()
        print("[Luồng Xử Lý] Đang tải models cho phân tích thuộc tính, vui lòng đợi...")
        asyncio.run(attributes_analyzer.load_models())
        print("✅ [Luồng Xử Lý] Tải models thuộc tính hoàn tất.")

        last_id = db_manager.get_max_person_id()
        track_manager = TrackManager(reid_face_analyzer, db_manager)
        track_manager.next_person_id = last_id + 1
        print(f"✅ ID lớn nhất trong CSDL: {last_id}. ID tiếp theo: {last_id + 1}.")

        # --- 2. Thiết lập xử lý đa luồng ---
        reid_task_queue = queue.Queue(maxsize=100)
        reid_result_queue = queue.Queue()
        reid_worker = threading.Thread(target=reid_face_worker, args=(reid_task_queue, reid_result_queue, reid_face_analyzer), daemon=True)
        reid_worker.start()

        attribute_task_queue = queue.Queue(maxsize=50)
        attribute_result_queue = queue.Queue()
        attribute_worker = threading.Thread(target=attribute_analysis_worker, args=(attribute_task_queue, attribute_result_queue, attributes_analyzer), daemon=True)
        attribute_worker.start()

        # --- 3. Load model YOLO ---
        model = YOLO(config.PERSON_MODEL_PATH)
        
        print("\n✅ [Luồng Xử Lý] Hệ thống đã sẵn sàng. Chờ frame từ ZMQ...")
        
        # [THÊM VÀO] Khởi tạo biến để đo FPS xử lý
        fps_start_time = time.time()
        fps_frame_count = 0
        processing_fps = 0

        # --- Vòng lặp chính ---
        while not stop_event.is_set():
            try:
                image_bytes = frame_queue.get(timeout=1.0)
                
                np_array = np.frombuffer(image_bytes, dtype=np.uint8)
                frame = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

                if frame is None:
                    print("⚠️ [Luồng Xử Lý] Không thể giải mã frame, bỏ qua.")
                    continue

                # [THÊM VÀO] Tăng biến đếm FPS xử lý
                fps_frame_count += 1

                # Chạy model
                results = model.track(source=frame, show=False, conf=0.5, verbose=False, iou=0.5, classes=[0], tracker=config.TRACKER_CONFIG_PATH, stream=False, persist=True)
                
                for result in results:
                    track_ids, bboxes = [], []
                    if result.boxes.id is not None:
                        original_track_ids = result.boxes.id.int().cpu().tolist()
                        original_bboxes = result.boxes.xyxy.cpu().tolist()
                        for track_id, bbox in zip(original_track_ids, original_bboxes):
                            if (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) > config.MIN_BBOX_AREA and (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) < config.MAX_BBOX_AREA:
                                track_ids.append(track_id)
                                bboxes.append(bbox)

                    track_manager.update_tracks(track_ids, bboxes, frame, reid_task_queue, attribute_task_queue)
                    track_manager.process_analysis_results(reid_result_queue)
                    track_manager.process_attribute_results(attribute_result_queue)
                    
                    frame_with_drawings = draw_tracked_objects(frame, track_manager.tracked_objects)
                    
                    # [THÊM VÀO] Tính toán và hiển thị FPS xử lý lên frame
                    elapsed_time = time.time() - fps_start_time
                    if elapsed_time >= 1.0:
                        processing_fps = fps_frame_count / elapsed_time
                        # Reset để đo cho giây tiếp theo
                        fps_start_time = time.time()
                        fps_frame_count = 0
                    
                    # Vẽ FPS lên màn hình
                    fps_text = f"Processing FPS: {processing_fps:.2f}"
                    cv2.putText(frame_with_drawings, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                    cv2.imshow("Tracking & Analysis System", frame_with_drawings)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("ℹ️ Nhấn 'q', gửi tín hiệu dừng...")
                    stop_event.set()
                    break

                frame_queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                print(f"❌ [Luồng Xử Lý] Lỗi nghiêm trọng: {e}")
                stop_event.set()
                break

    finally:
        if db_manager:
            db_manager.save_all_databases()
        cv2.destroyAllWindows()
        print("--- [Luồng Xử Lý] HỆ THỐNG ĐÃ DỪNG ---")


# --- Hàm main() mới: Điều phối các luồng ---
def main():
    """
    Hàm main chính để khởi tạo và quản lý các luồng.
    """
    frame_queue = queue.Queue(maxsize=FRAME_QUEUE_MAX_SIZE)
    stop_event = threading.Event()
    zmq_context = zmq.Context()

    # Tạo và khởi chạy các luồng
    network_t = threading.Thread(target=network_receiver_worker, args=(zmq_context, stop_event, frame_queue), daemon=True)
    processing_t = threading.Thread(target=processing_loop, args=(frame_queue, stop_event), daemon=True)
    
    print("🚀 Bắt đầu khởi chạy các luồng...")
    network_t.start()
    processing_t.start()

    try:
        # Giữ luồng chính sống để bắt tín hiệu Ctrl+C
        processing_t.join()
    except KeyboardInterrupt:
        print("ℹ️ Bắt được tín hiệu Ctrl+C, gửi tín hiệu dừng...")
        stop_event.set()

    # Đảm bảo các luồng đã dừng hoàn toàn
    network_t.join(timeout=2)
    
    # Dọn dẹp
    zmq_context.term()
    print("✅ Tất cả các luồng đã kết thúc. Thoát chương trình.")
 

if __name__ == "__main__":
    main()



