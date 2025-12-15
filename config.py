import cv2
import os

#======================================================================================

# <<< PATH CONFIGURATION - CẤU HÌNH ĐƯỜNG DẪN >>>

# Vui lòng đảm bảo các file model nằm trong thư mục "models"

# ======================================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.path.join(BASE_DIR, "models")
# -----------------------------------------------------------------------------
# CẤU HÌNH CHO VIỆC VẼ (DRAWING CONFIGURATIONS)
# -----------------------------------------------------------------------------

# --- Cấu hình cho Bảng thông tin thuộc tính (Info Panel) ---
INFO_PANEL_BG = (40, 40, 40)         # Màu nền của bảng thông tin (BGR)
FONT = cv2.FONT_HERSHEY_SIMPLEX      # Font chữ chung
FONT_SCALE_INFO = 0.5                # Kích thước font cho thông tin chi tiết
FONT_THICKNESS = 1                   # Độ dày nét chữ

# --- Màu sắc cho các loại văn bản khác nhau ---
COLOR_INFO_TEXT = (255, 255, 255)    # Màu trắng cho thông tin chung (giới tính, da)
COLOR_CLOTHING_TEXT = (200, 200, 200) # Màu xám nhạt cho thông tin quần áo


# # Đường dẫn tới các file model

PERSON_MODEL_PATH = os.path.join(MODEL_DIR, "yolo11n.pt")

GENDER_FACE_MODEL_PATH = os.path.join(MODEL_DIR, "GDF_038_93.pt")

GENDER_POSE_MODEL_PATH = os.path.join(MODEL_DIR, "GDP_038_91.pt")
RKNN_MODEL_PATH = os.path.join(BASE_DIR, "yolo11n_rknn_model/models/yolo11n-rk3588.rknn")
#Model RKNN
#PERSON_MODEL_PATH = os.path.join(BASE_DIR, "yolo11n_rknn_model/models/yolo11n-rk3588.rknn")

# GENDER_FACE_MODEL_PATH = os.path.join(BASE_DIR, "GDF_038_93_rknn_model/models/GDF_038_93-rk3588.rknn")

# GENDER_POSE_MODEL_PATH = os.path.join(BASE_DIR, "GDP_038_91_rknn_model/models/GDP_038_91-rk3588.rknn")



POSE_MODEL_PATH = os.path.join(MODEL_DIR, "pose_landmarker.task")
SKIN_CSV_PATH = os.path.join(MODEL_DIR, "skin_tone.csv")



# Danh sách các file model bắt buộc phải có

REQUIRED_MODEL_PATHS = {

"Person Detector": PERSON_MODEL_PATH,

"Pose Estimator": POSE_MODEL_PATH,

"Gender Face Model": GENDER_FACE_MODEL_PATH,

"Gender Pose Model": GENDER_POSE_MODEL_PATH,

"Skin Tone CSV": SKIN_CSV_PATH

}



# ======================================================================================
# <<< PATH CONFIGURATION - CẤU HÌNH ĐƯỜNG DẪN >>>
# ======================================================================================
YOLO_MODEL_PATH = "models/yolo11n.pt"
TRACKER_CONFIG_PATH = "botsort.yaml"
MAX_DISAPPEARED_FRAMES_BEFORE_DELETION = 10
ATTRIBUTE_ANALYSIS_INTERVAL = 5


# ======================================================================================
# <<< TRACKER LOGIC CONFIGURATION - CẤU HÌNH LOGIC TRACKER >>>
# ======================================================================================
# --- Cấu hình Thu thập & Nhận dạng Thông minh ---
QUALITY_SCORE_THRESHOLD = 100.0     # <<< MỚI >>> Ngưỡng điểm chất lượng để kích hoạt nhận dạng
HIGH_CONF_FACE_SCORE = 5.0       # <<< MỚI >>> Điểm cộng thêm khi có khuôn mặt rất rõ nét
MID_CONF_FACE_SCORE = 2.0        # <<< MỚI >>> Điểm cộng thêm khi có khuôn mặt khá rõ nét
BASE_REID_SCORE = 2.0             # <<< MỚI >>> Điểm cơ bản cho mỗi lần thu thập được vector toàn thân

STABLE_IDENTIFICATION_THRESHOLD = 0.7 # <<< MỚI >>> Ngưỡng điểm tin cậy để coi là 'identified', dưới ngưỡng này là 'tentative'
FACE_CONFIDENCE_THRESHOLD = 0.8       # Ngưỡng tin cậy của model face để tính điểm
# --- Cấu hình cho việc làm giàu dữ liệu (Data Enrichment) ---
# Số lượng vector mặt tối thiểu một ID nên có trong DB. Nếu ít hơn, hệ thống sẽ cố gắng bổ sung.
MAX_FACE_VECTORS_PER_PROFILE = 15 # Số vector mặt tối đa
MAX_REID_VECTORS_PER_PROFILE = 25
# Ngưỡng confidence tối thiểu của một khuôn mặt để được xem xét bổ sung vào DB (nên đặt rất cao).
HIGH_CONFIDENCE_THRESHOLD_FOR_ENRICHMENT = 0.95
# --- Cấu hình chung ---
MAX_DISAPPEARED_FRAMES = 10       # Số frame tối đa cho phép một track biến mất trước khi bị xóa
MOVING_AVERAGE_WINDOW = 25        # Kích thước cửa sổ để lưu trữ các vector tạm thời cho mỗi track
# Ngưỡng diện tích bounding box tối thiểu (pixel)
# Bất kỳ box nào có diện tích nhỏ hơn ngưỡng này sẽ bị bỏ qua.
# Ví dụ: 50*80 = 4000
MIN_BBOX_AREA = 5000
MAX_BBOX_AREA = 150000
# ======================================================================================
# <<< VECTOR DATABASE CONFIGURATION - CẤU HÌNH CSDL VECTOR >>>
# ======================================================================================
# --- Namespaces & Dimensions ---
REID_NAMESPACE = "reid_full_body"
FACE_NAMESPACE = "face_features"
OSNET_VECTOR_DIM = 512
FACE_VECTOR_DIM = 128 # Tùy thuộc vào model face của bạn, MobileFaceNet thường là 128 hoặc 512
# Ngưỡng khác biệt tối thiểu để lưu một vector mới (dựa trên khoảng cách Euclidean)
# Nếu khoảng cách giữa vector mới và vector cuối cùng nhỏ hơn ngưỡng này, nó sẽ bị bỏ qua.
VECTOR_DIFFERENCE_THRESHOLD = 0.2
# --- Cấu hình Tìm kiếm & Bỏ phiếu (Voting) ---
SEARCH_TOP_K = 15                 # Lấy K vector gần nhất từ DB để bỏ phiếu

# Ngưỡng cho NHẬN DẠNG KHUÔN MẶT (Face Recognition)
FACE_DB_SEARCH_SIMILARITY_THRESHOLD = 0.85 # Ngưỡng tương đồng để một vector mặt được tính là hợp lệ
FACE_MIN_VOTES_FOR_MATCH = 5              # Số phiếu tối thiểu cần có để xác nhận một match từ mặt

# Ngưỡng cho NHẬN DẠNG TOÀN THÂN (Re-ID)
REID_DB_SEARCH_SIMILARITY_THRESHOLD = 0.8 # Ngưỡng tương đồng cho vector toàn thân
REID_MIN_VOTES_FOR_MATCH = 10              # Số phiếu tối thiểu cần có để xác nhận một match từ toàn thân

# ======================================================================================
# <<< DRAWING CONFIGURATION - CẤU HÌNH HIỂN THỊ >>>
# ======================================================================================
# --- Fonts ---
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.7
FONT_THICKNESS = 2

# --- BGR Colors ---
TEMP_ID_COLOR = (0, 0, 255)             # Đỏ - Trạng thái 'pending'
TENTATIVE_ID_COLOR = (0, 165, 255)      # Cam - Trạng thái 'tentative' <<< MỚI >>>
CONFIRMED_ID_COLOR = (0, 255, 0)        # Xanh lá - Trạng thái 'confirmed' hoặc 'identified'