# main_.py
import cv2
import queue
import threading
import asyncio
import sys
import os
import numpy as np
import time
from multiprocessing import shared_memory 

# --- CẤU HÌNH ĐƯỜNG DẪN ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)
UTILS_DIR = os.path.join(BASE_DIR, 'utils')
if UTILS_DIR not in sys.path:
    sys.path.append(UTILS_DIR)

# --- IMPORT MODULE DỰ ÁN ---
import config
from analyzer import Analyzer as ReidFaceAnalyzer
from attributes_analyzer import AttributesAnalyzer
from vector_database import VectorDatabase_Manager
from tracker import TrackManager
from draw import draw_tracked_objects
from utils.profiler import Profiler

# --- IMPORT AI FRAMEWORKS ---
try:
    from ultralytics import YOLO
except ImportError:
    print("❌ Lỗi: Thiếu thư viện 'ultralytics'. Hãy cài đặt: pip install ultralytics")
    sys.exit(1)

# Kiểm tra model
if not os.path.exists(config.YOLO_MODEL_PATH):
    print(f"❌ Lỗi: Không tìm thấy model YOLO tại {config.YOLO_MODEL_PATH}")
    sys.exit(1)

# ===================================================================
# ==========              CÁC HÀM HỖ TRỢ                   ==========
# ===================================================================

def is_image_quality_good(image, min_size=(64, 128), blur_threshold=80.0):
    if image is None or image.size == 0: return False
    h, w, _ = image.shape
    if w < min_size[0] or h < min_size[1]: return False
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if cv2.Laplacian(gray, cv2.CV_64F).var() < blur_threshold: return False
    return True

# ===================================================================
# ==========                WORKER THREADS                 ==========
# ===================================================================

def reid_face_worker(task_queue, result_queue, analyzer):
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
            print(f"Lỗi Worker 1: {e}")

def attribute_analysis_worker(task_queue, result_queue, analyzer):
    print("🚀 Worker Thuộc tính đã bắt đầu.")
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
            print(f"Lỗi Worker 2: {e}")

class SharedMemoryFrameProvider:
    def __init__(self, name="video_shm", shape=(480, 640, 3)):
        self.shape = shape
        try:
            self.shm = shared_memory.SharedMemory(name=name)
            self.frame = np.ndarray(shape, dtype=np.uint8, buffer=self.shm.buf)
        except FileNotFoundError:
            raise RuntimeError("Shared Memory chưa tồn tại")

    def read(self):
        return True, self.frame.copy()

    def isOpened(self):
        return True

    def release(self):
        try:
            self.shm.close()
        except:
            pass

# ===================================================================
# ==========                 HÀM MAIN                      ==========
# ===================================================================

def main():
    print("--- BẮT ĐẦU HỆ THỐNG TRACKING & PHÂN TÍCH (PYTORCH ONLY) ---")
    
    profiler = Profiler()
    frame_count = 0
    db_manager = None
    yolo_model = None
    frame_provider = None

    try:
        # Khởi tạo Database và Analyzer
        db_manager = VectorDatabase_Manager()
        reid_face_analyzer = ReidFaceAnalyzer(face_model_name="mobilefacenet")
        attributes_analyzer = AttributesAnalyzer()
        
        print("Đang tải models thuộc tính...")
        asyncio.run(attributes_analyzer.load_models())
        print("✅ Models thuộc tính OK.")

        # Khởi tạo Tracker Manager
        last_id = db_manager.get_max_person_id()
        track_manager = TrackManager(reid_face_analyzer, db_manager)
        track_manager.next_person_id = last_id + 1

        # Khởi tạo Threads
        reid_task_queue = queue.Queue(maxsize=200)
        reid_result_queue = queue.Queue()
        threading.Thread(target=reid_face_worker, args=(reid_task_queue, reid_result_queue, reid_face_analyzer), daemon=True).start()

        attribute_task_queue = queue.Queue(maxsize=100)
        attribute_result_queue = queue.Queue()
        threading.Thread(target=attribute_analysis_worker, args=(attribute_task_queue, attribute_result_queue, attributes_analyzer), daemon=True).start()

        # Load Model YOLO (PyTorch)
        print(f"--> [MODE: PyTorch] Đang tải {config.YOLO_MODEL_PATH}...")
        yolo_model = YOLO(config.YOLO_MODEL_PATH) 
        print("✅ YOLO PyTorch OK.")

        # Mở Camera
        print("--> Đang kết nối Shared Memory...")
        shm_name = getattr(config, 'SHM_NAME', 'video_shm')
        shm_shape = getattr(config, 'SHM_FRAME_SHAPE', (480, 640, 3))
        
        while True:
            try:
                frame_provider = SharedMemoryFrameProvider(name=shm_name, shape=shm_shape)
                break
            except RuntimeError as e:
                print(f"⏳ {e} - Thử lại sau 1s...")
                time.sleep(1)

        print("\n🚀 HỆ THỐNG ĐÃ SẴN SÀNG. PRESS 'Q' TO EXIT.")

        while True:
            frame_count += 1
            profiler.start("Total_Frame")
            
            profiler.start("Read_Frame")
            ret, frame = frame_provider.read()
            profiler.stop("Read_Frame")
            if not ret: break

            # [PYTORCH FLOW]
            profiler.start("Inference")
            # Chạy YOLO track với config từ file yaml
            results = yolo_model.track(frame, persist=True, verbose=False, conf=0.45, iou=0.5, classes=[0], tracker=config.TRACKER_CONFIG_PATH)
            profiler.stop("Inference")
            
            track_ids, bboxes = [], []
            if results[0].boxes.id is not None:
                track_ids = results[0].boxes.id.int().cpu().tolist()
                
                # [QUAN TRỌNG] FIX LỖI "Bounding box không hợp lệ"
                # Chuyển float sang int và kẹp vào khung hình
                h_img, w_img = frame.shape[:2]
                for box in results[0].boxes.xyxy.cpu().numpy():
                    x1 = max(0, int(box[0]))
                    y1 = max(0, int(box[1]))
                    x2 = min(w_img, int(box[2]))
                    y2 = min(h_img, int(box[3]))
                    bboxes.append([x1, y1, x2, y2])

            profiler.start("Logic_Update")
            clean_ids, clean_bboxes = [], []
            for tid, bbox in zip(track_ids, bboxes):
                # Kiểm tra kỹ lần cuối: x2 > x1 và y2 > y1
                if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]: continue 
                
                area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                if config.MIN_BBOX_AREA < area < config.MAX_BBOX_AREA:
                    clean_ids.append(tid)
                    clean_bboxes.append(bbox)

            track_manager.update_tracks(clean_ids, clean_bboxes, frame, reid_task_queue, attribute_task_queue)
            track_manager.process_analysis_results(reid_result_queue)
            track_manager.process_attribute_results(attribute_result_queue)
            profiler.stop("Logic_Update")

            profiler.start("Drawing")
            frame_out = draw_tracked_objects(frame, track_manager.tracked_objects)
            
            stats, cpu_p, mem_mb = profiler.get_stats()
            avg_time = stats.get("Total_Frame", 0)
            fps_est = 1000 / (avg_time + 1e-5)
            color_info = (0, 255, 0) # Màu xanh lá cho PyTorch mode
            
            cv2.putText(frame_out, f"FPS: {fps_est:.1f} | CPU: {cpu_p:.0f}%", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_info, 2)
            cv2.imshow("System",frame_out)
            profiler.stop("Drawing")
            
            profiler.stop("Total_Frame")
            profiler.print_report(frame_count)
 
            if cv2.waitKey(1) & 0xFF == ord('q'): break

    except Exception as e:
        import traceback
        print(f"❌ LỖI: {e}")
        traceback.print_exc()
    finally:
        if frame_provider: frame_provider.release()
        if db_manager: db_manager.save_all_databases()
        cv2.destroyAllWindows()
        print("--- END ---")

if __name__ == "__main__":
    main()
