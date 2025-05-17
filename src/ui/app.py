import streamlit as st
import requests
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# Thêm đường dẫn của thư mục cha vào sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Cấu hình Streamlit
st.set_page_config(
    page_title="Hệ thống phân loại review",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# URL của API
API_URL = "http://localhost:5000/predict"

# Hàm gọi API để dự đoán
def predict_review(review_text):
    try:
        response = requests.post(
            API_URL,
            json={"review": review_text},
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Lỗi từ API (HTTP {response.status_code}): {response.text}")
            return None
    except Exception as e:
        st.error(f"Lỗi kết nối đến API: {str(e)}")
        return None

# Hàm tạo biểu đồ từ dữ liệu EDA
def load_and_display_eda():
    try:
        # Tải dữ liệu EDA (thay bằng đường dẫn thực tế)
        eda_data_path = "../../data/processed/eda_data.csv"
        
        # Kiểm tra xem file có tồn tại không
        if not os.path.exists(eda_data_path):
            st.warning("Dữ liệu EDA không có sẵn. Hiển thị dữ liệu mẫu thay thế.")
            # Tạo dữ liệu mẫu để demo
            df = pd.DataFrame({
                "length": np.random.normal(100, 30, 1000),
                "sentiment": np.random.choice(["Tích cực", "Tiêu cực"], 1000, p=[0.7, 0.3])
            })
        else:
            df = pd.read_csv(eda_data_path)
        
        # Biểu đồ phân bố độ dài review
        fig_length = px.histogram(
            df, 
            x="length", 
            title="Phân bố độ dài review",
            labels={"length": "Độ dài review (ký tự)", "count": "Số lượng"},
            color_discrete_sequence=["#3498db"]
        )
        st.plotly_chart(fig_length, use_container_width=True)
        
        # Biểu đồ phân bố nhãn
        if "sentiment" in df.columns:
            sentiment_counts = df["sentiment"].value_counts().reset_index()
            sentiment_counts.columns = ["Sentiment", "Count"]
            
            fig_sentiment = px.pie(
                sentiment_counts, 
                values="Count", 
                names="Sentiment",
                title="Phân bố nhãn sentiment",
                color_discrete_sequence=["#2ecc71", "#e74c3c"]
            )
            st.plotly_chart(fig_sentiment, use_container_width=True)
        
        # Vẽ word cloud nếu có dữ liệu
        if "tokens" in df.columns and "sentiment" in df.columns:
            st.subheader("Word Cloud theo loại review")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Word cloud cho review tích cực
                positive_tokens = " ".join(df[df["sentiment"] == "Tích cực"]["tokens"].astype(str))
                if positive_tokens:
                    wordcloud_positive = WordCloud(
                        width=800, height=400, 
                        background_color="white", 
                        max_words=100,
                        colormap="viridis"
                    ).generate(positive_tokens)
                    
                    fig_pos, ax_pos = plt.subplots(figsize=(10, 5))
                    ax_pos.imshow(wordcloud_positive, interpolation='bilinear')
                    ax_pos.axis("off")
                    ax_pos.set_title("Từ khóa phổ biến trong review tích cực")
                    st.pyplot(fig_pos)
            
            with col2:
                # Word cloud cho review tiêu cực
                negative_tokens = " ".join(df[df["sentiment"] == "Tiêu cực"]["tokens"].astype(str))
                if negative_tokens:
                    wordcloud_negative = WordCloud(
                        width=800, height=400, 
                        background_color="white", 
                        max_words=100,
                        colormap="magma"
                    ).generate(negative_tokens)
                    
                    fig_neg, ax_neg = plt.subplots(figsize=(10, 5))
                    ax_neg.imshow(wordcloud_negative, interpolation='bilinear')
                    ax_neg.axis("off")
                    ax_neg.set_title("Từ khóa phổ biến trong review tiêu cực")
                    st.pyplot(fig_neg)
    
    except Exception as e:
        st.error(f"Lỗi khi tải và hiển thị dữ liệu EDA: {str(e)}")

# Giao diện Streamlit
def main():
    st.title("🔍 Hệ Thống Phân Loại Review")
    
    # Sidebar cho các tùy chọn
    st.sidebar.title("Tùy chọn")
    
    # Option để hiển thị dashboard
    show_dashboard = st.sidebar.checkbox("Hiển thị dashboard EDA", value=True)
    
    # Tab cho các chức năng khác nhau
    tab1, tab2 = st.tabs(["Phân loại Review", "Thông tin dự án"])
    
    with tab1:
        # Chia layout thành 2 cột
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.subheader("📝 Nhập review của bạn")
            review_input = st.text_area(
                "Nhập nội dung review:",
                height=200,
                placeholder="Nhập review sản phẩm của bạn ở đây...",
            )
            
            # Nút submit với loading spinner
            if st.button("Phân loại review", type="primary"):
                if review_input:
                    with st.spinner("Đang phân tích review..."):
                        result = predict_review(review_input)
                    
                    if result and "label" in result:
                        # Hiển thị kết quả
                        st.success("✅ Phân tích hoàn tất!")
                        
                        # Tạo container với border
                        with st.container():
                            st.markdown("""
                            <style>
                            .result-container {
                                border-radius: 10px;
                                padding: 20px;
                                background-color: #f8f9fa;
                                margin-bottom: 20px;
                            }
                            </style>
                            """, unsafe_allow_html=True)
                            
                            st.markdown('<div class="result-container">', unsafe_allow_html=True)
                            
                            # Hiển thị nhãn và độ tin cậy
                            st.subheader("Kết quả phân loại:")
                            
                            # Màu khác nhau cho các nhãn
                            if result["label"] == "Tích cực":
                                label_color = "#2ecc71"  # Xanh lá cho tích cực
                            else:
                                label_color = "#e74c3c"  # Đỏ cho tiêu cực
                                
                            st.markdown(f"<h3 style='color: {label_color};'>Nhãn: {result['label']}</h3>", unsafe_allow_html=True)
                            
                            # Hiển thị độ tin cậy bằng progress bar
                            st.markdown(f"**Độ tin cậy:** {result['confidence']:.2%}")
                            st.progress(result["confidence"])
                            
                            # Thông tin thêm về thời gian xử lý
                            if "processing_time" in result:
                                st.markdown(f"**Thời gian xử lý:** {result['processing_time']:.3f} giây")
                                
                            st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.warning("Vui lòng nhập review trước khi phân loại.")
        
        with col2:
            st.subheader("💡 Review mẫu")
            st.markdown("""
            **Bạn có thể sử dụng các review mẫu sau:**
            
            **Review tích cực:**
            ```
            Sản phẩm tuyệt vời, đóng gói cẩn thận, giao hàng nhanh. Chất lượng vượt trội so với giá tiền. Rất hài lòng với trải nghiệm mua hàng này!
            ```
            
            **Review tiêu cực:**
            ```
            Thất vọng với sản phẩm này. Hàng kém chất lượng, không như mô tả. Đã liên hệ với shop nhiều lần nhưng không được phản hồi. Sẽ không mua lại.
            ```
            """)
            
            # Thêm các nút để chèn review mẫu
            if st.button("Chèn review tích cực mẫu"):
                st.session_state.review_example = "Sản phẩm tuyệt vời, đóng gói cẩn thận, giao hàng nhanh. Chất lượng vượt trội so với giá tiền. Rất hài lòng với trải nghiệm mua hàng này!"
                st.rerun()
                
            if st.button("Chèn review tiêu cực mẫu"):
                st.session_state.review_example = "Thất vọng với sản phẩm này. Hàng kém chất lượng, không như mô tả. Đã liên hệ với shop nhiều lần nhưng không được phản hồi. Sẽ không mua lại."
                st.rerun()
    
    # Dashboard EDA
    if show_dashboard:
        st.markdown("---")
        st.header("📊 Dashboard Phân Tích Dữ Liệu")
        load_and_display_eda()
    
    with tab2:
        st.header("📋 Thông tin về dự án")
        st.markdown("""
        ### Giới thiệu
        Dự án này là một hệ thống phân loại review sản phẩm sử dụng các kỹ thuật Xử lý Ngôn ngữ Tự nhiên (NLP) và Học máy (Machine Learning).
        
        ### Các thành phần chính
        1. **Tiền xử lý dữ liệu:** Làm sạch và tokenize dữ liệu review.
        2. **Mô hình phân loại:** Sử dụng các thuật toán học máy để phân loại sentiment của review.
        3. **API Backend:** API RESTful xử lý dữ liệu và trả về kết quả dự đoán.
        4. **Giao diện người dùng:** Frontend tương tác với người dùng, hiển thị kết quả và dashboard.
        
        ### Công nghệ sử dụng
        - **Backend:** Flask/FastAPI, Scikit-learn
        - **Frontend:** Streamlit
        - **Phân tích dữ liệu:** Pandas, NumPy, Matplotlib, Plotly
        - **Deploy:** Heroku/Streamlit Sharing
        
        ### Tác giả
        Nhóm sinh viên thực hiện đồ án môn học.
        """)

# Kiểm tra xem có review mẫu trong session state không
if "review_example" in st.session_state:
    review_input = st.session_state.review_example
    # Xóa session state để tránh lặp lại
    del st.session_state.review_example

# Chạy ứng dụng
if __name__ == "__main__":
    main()