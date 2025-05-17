# Cấu hình cho API
DEBUG = True
PORT = 5000
MODEL_PATH = "../../data/model.pkl"  # Đường dẫn đến model
MODEL_VECTORIZER_PATH = "../../data/vectorizer.pkl"  # Đường dẫn đến vectorizer (nếu có)
MODEL_TOKENIZER_PATH = "../../data/tokenizer.pkl"  # Đường dẫn đến tokenizer (nếu có)

# Cấu hình CORS (Cross-Origin Resource Sharing)
CORS_HEADERS = {
    "Content-Type": "application/json",
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "POST, GET, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type"
}

# Các nhãn dự đoán
LABELS = {
    0: "Tiêu cực",
    1: "Tích cực"
}

# Các thông số khác
MAX_REVIEW_LENGTH = 5000  # Giới hạn độ dài review