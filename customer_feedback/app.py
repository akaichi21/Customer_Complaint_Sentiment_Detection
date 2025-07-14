import streamlit as st
import re
import torch
from transformers import AutoTokenizer, BertForSequenceClassification
import joblib
from safetensors.torch import load_file

# Check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("models/tokenizer", use_fast=False)

# Load complaint model
complaint_model = BertForSequenceClassification.from_pretrained(
    "vinai/phobert-base",
    num_labels=len(joblib.load("models/complaint_model/label_encoder_complaint.pkl").classes_)
)
complaint_model.load_state_dict(load_file("models/complaint_model/model.safetensors"))
complaint_model.to(device)
complaint_model.eval()

# Load sentiment model
sentiment_model = BertForSequenceClassification.from_pretrained(
    "vinai/phobert-base",
    num_labels=len(joblib.load("models/sentiment_model/label_encoder_sentiment.pkl").classes_)
)
sentiment_model.load_state_dict(load_file("models/sentiment_model/model.safetensors"))
sentiment_model.to(device)
sentiment_model.eval()

# Load label encoders
label_encoder_complaint = joblib.load("models/complaint_model/label_encoder_complaint.pkl")
label_encoder_sentiment = joblib.load("models/sentiment_model/label_encoder_sentiment.pkl")

# Preprocess
def preprocess_for_phobert(text):
    # Convert lower text
    text = text.lower()

    # Delete special characters and number
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\d+", " ", text)

    # Delete extra space
    text = re.sub(r"\s+", " ", text).strip()

    return text

# Predict
def predict(text):
    cleaned_text = preprocess_for_phobert(text)

    encoding = tokenizer(
        cleaned_text,
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        output_complaint = complaint_model(input_ids=input_ids, attention_mask=attention_mask)
        predict_complaint = torch.argmax(output_complaint.logits, dim=1).item()

        output_sentiment = sentiment_model(input_ids=input_ids, attention_mask=attention_mask)
        predict_sentiment = torch.argmax(output_sentiment.logits, dim=1).item()

    complaint_label = label_encoder_complaint.inverse_transform([predict_complaint])[0]
    sentiment_label = label_encoder_sentiment.inverse_transform([predict_sentiment])[0]

    return complaint_label, sentiment_label

# Streamlit interface
# Map emoji for sentiment
emoji_map = {
    "Bình thường": "😐",
    "Bực bội": "😠",
    "Hài lòng": "😄"
}

st.set_page_config(
    page_title="Classification Customer Complaint & Sentiment",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown("<h2 style='text-align:center;'> <span style='color:#ff6600;'>Boss Lover</span> - Classification Customer Complaint & Sentiment</h2>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align:center;'> Áp dụng cho các sản phẩm <span style='color:#ff6600;'>chai lọ</span> tại <span style='color:#ff6600;'>Boss Lover (Shopee)</span></h4>", unsafe_allow_html=True)
st.markdown("=====")

text_input = st.text_area("📝 Phản Hồi Của Bạn Về Các Sản Phẩm Chai Lọ Của Shop: ", height=100)

if st.button("🔍 Xác Nhận & Dự Đoán"):
    if text_input.strip() == "":
        st.warning("⚠️Vui Lòng Nhập Ý Kiến Của Bạn Trước Khi Dự Đoán.")
    else:
        complaint, sentiment = predict(text_input)
        st.success("☑️Dự Đoán Thành Công")

        st.markdown(f"""
                <div style='background-color:#ff6600;padding:20px;border-radius:10px;margin-top:20px'>
                    <h4>📌 <b>Loại Phản Hồi:</b> <span style='color:#d6336c'>{complaint}</span></h4>
                    <h4>💬 <b>Cảm Xúc Khách Hàng:</b> <span style='color:#1f77b4'>{sentiment} {emoji_map.get(sentiment, '')}</span></h4>
                </div>""", unsafe_allow_html=True)
