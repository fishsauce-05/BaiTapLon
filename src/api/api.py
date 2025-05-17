from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import traceback
import time
import os
import sys
import logging
from logging.handlers import RotatingFileHandler

# Thêm đường dẫn của thư mục cha vào sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import các thành phần từ config
from src.api.config import DEBUG, PORT, MODEL_PATH, CORS_HEADERS, LABELS, MAX_REVIEW_LENGTH

# Thiết lập logging
if not os.path.exists('logs'):
    os.mkdir('logs')
file_handler = RotatingFileHandler('logs/api.log', maxBytes=10240, backupCount=10)
file_handler.setFormatter(logging.Formatter(
    '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
))
file_handler.setLevel(logging.INFO)

# Khởi tạo Flask app
app = Flask(__name__)
app.logger.setLevel(logging.INFO)
app.logger.addHandler(file_handler)
CORS(app)

# Hàm load model
def load_model():
    try:
        app.logger.info("Đang tải model từ {}".format(MODEL_PATH))
        model = joblib.load(MODEL_PATH)
        app.logger.info("Tải model thành công")
        return model
    except Exception as e:
        app.logger.error("Lỗi khi tải model: {}".format(e))
        app.logger.error(traceback.format_exc())
        return None

# Hàm tiền xử lý dữ liệu review
def preprocess_review(review_text):
    """
    Tiền xử lý review trước khi dự đoán
    Phần này có thể cần điều chỉnh theo yêu cầu model
    """
    # Ví dụ đơn giản: Cắt review nếu quá dài
    if len(review_text) > MAX_REVIEW_LENGTH:
        return review_text[:MAX_REVIEW_LENGTH]
    return review_text

# Load model khi khởi động app
model = load_model()

@app.route('/health', methods=['GET'])
def health_check():
    """Endpoint kiểm tra trạng thái API"""
    if model is None:
        return jsonify({"status": "error", "message": "Model chưa được tải"}), 500
    return jsonify({"status": "ok", "message": "API đang hoạt động bình thường"})

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint dự đoán nhãn từ review"""
    # Bắt đầu tính thời gian
    start_time = time.time()
    
    # Kiểm tra model đã được tải chưa
    if model is None:
        return jsonify({"error": "Model chưa được tải, vui lòng khởi động lại API"}), 500
    
    # Lấy dữ liệu từ request
    try:
        data = request.get_json(force=True)
    except Exception as e:
        app.logger.error("Lỗi khi parse JSON: {}".format(e))
        return jsonify({"error": "Dữ liệu không đúng định dạng JSON"}), 400
    
    # Kiểm tra review có tồn tại không
    review_text = data.get("review", "")
    if not review_text:
        return jsonify({"error": "Thiếu trường 'review' trong dữ liệu gửi lên"}), 400
    
    try:
        # Tiền xử lý review
        processed_review = preprocess_review(review_text)
        app.logger.info(f"Review đã được tiền xử lý: {processed_review[:50]}...")
        
        # Dự đoán với model
        prediction = model.predict([processed_review])[0]
        probabilities = model.predict_proba([processed_review])[0]
        
        # Lấy xác suất cao nhất
        confidence = max(probabilities)
        
        # Chuyển đổi nhãn số thành nhãn có ý nghĩa
        label = LABELS.get(prediction, str(prediction))
        
        # Tạo kết quả trả về
        result = {
            "label": label,
            "confidence": float(confidence),
            "processing_time": time.time() - start_time
        }
        
        app.logger.info(f"Dự đoán thành công: {label} với độ tin cậy {confidence:.4f}")
        return jsonify(result)
    
    except Exception as e:
        app.logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.logger.info(f"Starting API server on port {PORT}")
    app.run(debug=DEBUG, host='0.0.0.0', port=PORT)