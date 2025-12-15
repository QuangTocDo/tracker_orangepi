import faiss
import numpy as np
import os
import pickle
import config
from typing import Optional, Tuple, List
import threading
from collections import defaultdict

class VectorDatabase_Manager:
    """
    Quản lý cơ sở dữ liệu vector Faiss.
    - Hỗ trợ lưu nhiều vector cho một ID.
    - <<< MỚI >>> Sử dụng cơ chế bỏ phiếu kết hợp, ưu tiên ID có điểm trung bình cao nhất
      trong số các ứng viên đã đạt đủ số phiếu tối thiểu.
    """
    def __init__(self, index_dir="faiss_indexes"):
        self.index_dir = index_dir
        os.makedirs(self.index_dir, exist_ok=True)
        
        self.dimensions = {
            config.REID_NAMESPACE: config.OSNET_VECTOR_DIM,
            config.FACE_NAMESPACE: config.FACE_VECTOR_DIM
        }

        self.indexes = {ns: self._load_index(ns) for ns in self.dimensions}
        self.id_maps = {ns: self._load_id_map(ns) for ns in self.dimensions}
        
        self.db_lock = threading.Lock()
        self.has_unsaved_changes = {ns: False for ns in self.dimensions}
        
        print("✅ Khởi tạo Faiss Vector DB Manager (Local) thành công.")

    def _get_paths(self, namespace: str) -> Tuple[str, str]:
        index_path = os.path.join(self.index_dir, f"{namespace}.index")
        id_map_path = os.path.join(self.index_dir, f"{namespace}.pkl")
        return index_path, id_map_path

    def _load_index(self, namespace: str) -> Optional[faiss.Index]:
        index_path, _ = self._get_paths(namespace)
        if os.path.exists(index_path):
            print(f"Đang tải index cho namespace '{namespace}' từ file...")
            return faiss.read_index(index_path)
        return faiss.IndexFlatIP(self.dimensions[namespace]) # Khởi tạo index trống nếu không có file

    def _load_id_map(self, namespace: str) -> list:
        _, id_map_path = self._get_paths(namespace)
        if os.path.exists(id_map_path):
            with open(id_map_path, 'rb') as f:
                return pickle.load(f)
        return []

    def _save_data(self, namespace: str):
        with self.db_lock:
            index_path, id_map_path = self._get_paths(namespace)
            if self.indexes[namespace] and self.has_unsaved_changes[namespace]:
                faiss.write_index(self.indexes[namespace], index_path)
                with open(id_map_path, 'wb') as f:
                    pickle.dump(self.id_maps[namespace], f)
                print(f"Đã lưu index và ID map cho namespace '{namespace}'.")
                self.has_unsaved_changes[namespace] = False

    def save_all_databases(self):
        print("Đang lưu tất cả thay đổi trong cơ sở dữ liệu vector...")
        for namespace in self.dimensions.keys():
            self._save_data(namespace)
        print("Lưu hoàn tất.")

    def add_vectors(self, namespace: str, vector_id: str, vectors_data: List[list]):
        """Thêm một danh sách các vector cho cùng một ID."""
        if not vectors_data: return
        if namespace not in self.indexes:
            print(f"Lỗi: Namespace '{namespace}' không hợp lệ.")
            return

        with self.db_lock:
            vectors_np = np.array(vectors_data, dtype='float32')
            faiss.normalize_L2(vectors_np)
            self.indexes[namespace].add(vectors_np)
            self.id_maps[namespace].extend([vector_id] * len(vectors_data))
            self.has_unsaved_changes[namespace] = True
            
        print(f"Thêm {len(vectors_data)} vector cho ID '{vector_id}' vào namespace '{namespace}' thành công.")

    def search_vector_with_voting(self, namespace: str, query_vector: list) -> Optional[Tuple[str, float]]:
        """
        <<< THAY ĐỔI LỚN >>>
        Tìm kiếm bằng cơ chế bỏ phiếu ưu tiên chất lượng:
        1. Tìm tất cả các ID có số phiếu >= ngưỡng tối thiểu.
        2. Trong số đó, chọn ID có điểm trung bình cao nhất làm người chiến thắng.
        """
        index = self.indexes.get(namespace)
        if index is None or index.ntotal == 0:
            return None

        if namespace == config.FACE_NAMESPACE:
            similarity_threshold = config.FACE_DB_SEARCH_SIMILARITY_THRESHOLD
            min_votes = config.FACE_MIN_VOTES_FOR_MATCH
        elif namespace == config.REID_NAMESPACE:
            similarity_threshold = config.REID_DB_SEARCH_SIMILARITY_THRESHOLD
            min_votes = config.REID_MIN_VOTES_FOR_MATCH
        else: # Mặc định
            similarity_threshold = config.REID_DB_SEARCH_SIMILARITY_THRESHOLD
            min_votes = config.REID_MIN_VOTES_FOR_MATCH
        
        query_np = np.array([query_vector], dtype='float32')
        faiss.normalize_L2(query_np)
        distances, indices = index.search(query_np, config.SEARCH_TOP_K)
        print(distances)
        
        # --- Phần debug giữ nguyên ---
        print(f"   [DEBUG] Các ứng viên Top-{config.SEARCH_TOP_K} cho '{namespace}' (Ngưỡng: {similarity_threshold}, Phiếu tối thiểu: {min_votes}):")
        for i, idx in enumerate(indices[0]):
            if idx == -1: continue
            score = float(distances[0][i])
            match_id = self.id_maps[namespace][idx]
            status = "✅ Hợp lệ" if score >= similarity_threshold else "❌ Loại"
            print(f"     - ID: {match_id:<15} | Score: {score:.4f} | Status: {status}")

        # <<< THAY ĐỔI: Bắt đầu logic bỏ phiếu mới >>>
        
        # Bước 1: Thu thập tất cả các phiếu bầu hợp lệ và điểm số của chúng
        scores_by_id = defaultdict(list)
        for i, idx in enumerate(indices[0]):
            if idx == -1: continue
            score = float(distances[0][i])
            if score >= similarity_threshold:
                match_id = self.id_maps[namespace][idx]
                scores_by_id[match_id].append(score)

        # Bước 2: Lọc ra những ứng viên cuối cùng (finalists) đạt đủ số phiếu tối thiểu
        finalists = []
        for match_id, scores in scores_by_id.items():
            if len(scores) >= min_votes:
                avg_score = np.mean(scores)
                finalists.append({'id': match_id, 'avg_score': avg_score, 'votes': len(scores)})

        if not finalists:
            print("     - Không có ứng viên nào đạt đủ số phiếu tối thiểu.")
            return None

        # Bước 3: Sắp xếp các ứng viên cuối cùng theo điểm trung bình giảm dần
        finalists.sort(key=lambda x: x['avg_score'], reverse=True)
        
        # In ra bảng xếp hạng các ứng viên cuối cùng để dễ debug
        print("     - Bảng xếp hạng các ứng viên cuối cùng (đã đủ phiếu):")
        for finalist in finalists:
            print(f"       - ID: {finalist['id']:<15} | Avg Score: {finalist['avg_score']:.4f} | Votes: {finalist['votes']}")

        # Người chiến thắng là người có điểm trung bình cao nhất
        winner = finalists[0]
        
        return winner['id'], float(winner['avg_score'])
    def count_vectors_for_id(self, namespace: str, vector_id: str) -> int:
        """Đếm số lượng vector thuộc về một ID cụ thể trong một namespace."""
        with self.db_lock:
            if namespace not in self.id_maps:
                return 0
            
            # Đếm số lần vector_id xuất hiện trong danh sách id_maps
            count = self.id_maps[namespace].count(vector_id)
            return count

    # Thêm vào cuối class VectorDatabase_Manager trong vector_database.py
    def get_max_person_id(self) -> int:
        """
        Tìm số ID lớn nhất từ tất cả các ID có dạng 'Person_X' trong CSDL.
        """
        max_id = 0
        all_ids = set()
        # Lấy tất cả các ID duy nhất từ tất cả các namespace
        for namespace in self.id_maps:
            all_ids.update(self.id_maps[namespace])

        for person_id in all_ids:
            if isinstance(person_id, str) and person_id.startswith("Person_"):
                try:
                    # Tách số từ chuỗi 'Person_X'
                    num = int(person_id.split('_')[1])
                    if num > max_id:
                        max_id = num
                except (ValueError, IndexError):
                    # Bỏ qua các ID không đúng định dạng
                    continue
        return max_id