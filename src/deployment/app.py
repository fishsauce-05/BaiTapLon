import os
import sys
import subprocess
import time
import webbrowser
import threading
import logging
from logging.handlers import RotatingFileHandler

# Thêm đường dẫn của thư mục cha vào sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Thiết lập logging
log_dir = 'logs'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        RotatingFileHandler(f'{log_dir}/deployment.log', maxBytes=10240, backupCount=5),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("deployment")

def start_api():
    """Khởi động API backend"""
    try:
        logger.info("Đang khởi động API backend...")
        # Chạy api.py từ thư mục api
        api_path = os.path.join(os.path.dirname(__file__), '../api/api.py')
        subprocess.Popen([sys.executable, api_path], 
                         stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE)
        logger.info("API đã được khởi động")
    except Exception as e:
        logger.error(f"Lỗi khi khởi động API: {str(e)}")
        raise

def start_ui():
    """Khởi động giao diện người dùng Streamlit"""
    try:
        logger.info("Đang khởi động giao diện người dùng...")
        # Chạy app.py từ thư mục ui
        ui_path = os.path.join(os.path.dirname(__file__), '../ui/app.py')
        
        # Sử dụng streamlit run với các tham số
        process = subprocess.Popen(
            ["streamlit", "run", ui_path, "--server.port", "8501", "--server.headless", "true"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        logger.info("Giao diện người dùng đã được khởi động")
        return process
    except Exception as e:
        logger.error(f"Lỗi khi khởi động UI: {str(e)}")
        raise

def check_api_health(max_retries=10, delay=1):
    """Kiểm tra xem API đã sẵn sàng chưa"""
    import requests
    
    retries = 0
    while retries < max_retries:
        try:
            response = requests.get("http://localhost:5000/health")
            if response.status_code == 200:
                logger.info("API đã sẵn sàng")
                return True
            else:
                logger.warning(f"API trả về mã trạng thái: {response.status_code}")
        except requests.exceptions.ConnectionError:
            logger.info(f"API chưa sẵn sàng, thử lại lần {retries+1}/{max_retries}")
        
        retries += 1
        time.sleep(delay)
    
    logger.error("Không thể kết nối đến API sau nhiều lần thử")
    return False

def open_browser():
    """Mở trình duyệt với địa chỉ của UI"""
    # Đợi một chút để Streamlit có thời gian khởi động
    time.sleep(3)
    webbrowser.open_new("http://localhost:8501")
    logger.info("Đã mở trình duyệt web")

def main():
    """Hàm chính để khởi động toàn bộ hệ thống"""
    try:
        logger.info("Đang khởi động hệ thống phân loại review...")
        
        # Khởi động API
        start_api()
        
        # Kiểm tra API đã sẵn sàng chưa
        if not check_api_health():
            logger.error("Không thể khởi động API, đang thoát...")
            sys.exit(1)
        
        # Khởi động UI
        ui_process = start_ui()
        
        # Mở trình duyệt web
        threading.Thread(target=open_browser).start()
        
        # Giữ chương trình chạy
        logger.info("Hệ thống đang chạy. Nhấn Ctrl+C để thoát.")
        ui_process.wait()
    
    except KeyboardInterrupt:
        logger.info("Nhận được tín hiệu thoát, đang dừng hệ thống...")
    except Exception as e:
        logger.error(f"Lỗi không xác định: {str(e)}")
    finally:
        # Tạm thời chưa làm gì khi kết thúc
        logger.info("Hệ thống đã dừng")

if __name__ == "__main__":
    main()