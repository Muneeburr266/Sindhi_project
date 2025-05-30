import streamlit as st
st.set_page_config(page_title='Sindhi Alphabet Recognition', page_icon='🔤', layout='centered', initial_sidebar_state='auto')
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
import os
import glob
import joblib
import torch
import timm
from torchvision import transforms

# Constants
IMG_SIZE = (64, 64)
MODEL_PATH = 'sindhi_alphabet_cnn (1).h5'
LABEL_MAP_PATH = 'label_map.txt'

# Load label map
label_map = {}
with open(LABEL_MAP_PATH, encoding='utf-8') as f:
    for line in f:
        k, v = line.strip().split('\t')
        label_map[int(v)] = k

# Load model
@st.cache_resource
def load_model():
    return keras.models.load_model(MODEL_PATH)
model = load_model()

def predict_image(image):
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, IMG_SIZE)
    image = image / 255.0
    image = image.reshape(1, IMG_SIZE[0], IMG_SIZE[1], 1)
    pred = model.predict(image)
    class_idx = np.argmax(pred)
    return label_map[class_idx]

# Custom CSS for background and style
def local_css():
    st.markdown(
        '''<style>
        .stApp {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        }
        .stButton>button {
            background-color: #2563eb;
            color: #fff;
            border-radius: 8px;
            font-size: 18px;
            padding: 0.5em 2em;
        }
        .stFileUploader>div>div {
            background-color: #e0e7ef;
            border-radius: 8px;
        }
        .stSuccess {
            background-color: #e0ffe0 !important;
            color: #065f46 !important;
            font-weight: bold;
        }
        .stError {
            background-color: #ffe0e0 !important;
            color: #991b1b !important;
            font-weight: bold;
        }
        h1, h2, h4, .stMarkdown h4 {
            color: #111827 !important;
        }
        span[style*="color:#4F8BF9;"] {
            color: #2563eb !important;
            font-weight: bold;
        }
        </style>''', unsafe_allow_html=True)
local_css()

st.title('🔤 Sindhi Alphabet Recognition')
st.markdown('<h4 style="color:#4F8BF9;">Upload an image of a Sindhi character to recognize it using a trained model (CNN or PyTorch).</h4>', unsafe_allow_html=True)

model_choice = st.selectbox('Select model to use:', ['CNN (Deep Learning)', 'PyTorch (Advanced CNN)'])

uploaded_file = st.file_uploader('Choose an image...', type=['png', 'jpg', 'jpeg'])

# Load models only when needed
def load_cnn():
    return keras.models.load_model('sindhi_alphabet_cnn (1).h5')
def load_pytorch():
    device = torch.device('cpu')
    # Get number of classes from label_map
    num_classes = len(label_map)
    model = timm.create_model('resnet18', pretrained=False, num_classes=num_classes, in_chans=3)
    model.load_state_dict(torch.load('best_model_scratch.pth', map_location=device))
    model.eval()
    return model

# Prediction functions
def predict_cnn(image):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, IMG_SIZE)
    image = image / 255.0
    image = image.reshape(1, IMG_SIZE[0], IMG_SIZE[1], 1)
    model = load_cnn()
    pred = model.predict(image)
    class_idx = np.argmax(pred)
    return label_map[class_idx]

def predict_pytorch(image):
    device = torch.device('cpu')
    model = load_pytorch()
    # Convert to RGB if needed
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Resize and normalize
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    img_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img_tensor)
        _, pred = torch.max(outputs, 1)
        class_idx = pred.item()
    return label_map[class_idx]

# Sindhi Unicode mapping for each Romanized class
sindhi_unicode_map = {
    'Ain': 'ع', 'Alif': 'ا', 'Bay': 'ب', 'Be': 'ٻ', 'Bhe': 'ڀ', 'Chay': 'چ', 'Chhe': 'ڇ',
    'Daad': 'ض', 'Daal': 'د', 'DDaal': 'ڊ', 'DDe': 'ڏ', 'DDh': 'ڌ', 'Dh': 'ڌُ', 'Do chashmi hey': 'ھ',
    'Fay': 'ف', 'Ga': 'ڳ', 'Gaaf': 'گ', 'Ghain': 'غ', 'Ghay': 'ڱ', 'Haa': 'ح', 'Hamzah': 'ء',
    'Jeem': 'ج', 'Jh': 'جھ', 'Jheem': 'ڄ', 'Kaaf': 'ڪ', 'Khaa': 'خ', 'Khay': 'ڃ', 'Laam': 'ل',
    'Meem': 'م', 'Ngaa': 'ڻ', 'Njeem': 'ڙ', 'NNoon': 'ن̤', 'Noon': 'ن', 'Pay': 'پ', 'Phay': 'ڦ',
    'Qaaf': 'ق', 'Ray': 'ر', 'RRe': 'ڙ', 'Saad': 'ص', 'Say': 'ث', 'Seen': 'س', 'Sheen': 'ش',
    'Tay': 'ط', 'Teee': 'ت', 'The': 'ٿ', 'TTe': 'ٽ', 'Tway': 'ٺ', 'Waaw': 'و', 'Yay': 'ي',
    'Zaal': 'ذ', 'Zay': 'ز', 'Zway': 'ظ'
}

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    st.image(img, caption='Uploaded Image', use_container_width=True)
    try:
        if model_choice == 'CNN (Deep Learning)':
            pred_label = predict_cnn(img)
            sindhi_char = sindhi_unicode_map.get(pred_label, '')
            st.success(f'Predicted label (CNN): {pred_label}')
            st.markdown(f'<h2 style="color:#1e293b;">Computer Text: <span style="color:#4F8BF9; font-family: Noto Nastaliq Urdu, Noto Sans Arabic, Arial;">{pred_label}</span></h2>', unsafe_allow_html=True)
            if sindhi_char:
                st.markdown(f'<h2 style="color:black; font-family: Noto Nastaliq Urdu, Noto Sans Arabic, Arial;">Sindhi Character: {sindhi_char}</h2>', unsafe_allow_html=True)
        else:
            pred_label = predict_pytorch(img)
            sindhi_char = sindhi_unicode_map.get(pred_label, '')
            st.success(f'Predicted label (PyTorch): {pred_label}')
            st.markdown(f'<h2 style="color:#1e293b;">Computer Text: <span style="color:#4F8BF9; font-family: Noto Nastaliq Urdu, Noto Sans Arabic, Arial;">{pred_label}</span></h2>', unsafe_allow_html=True)
            if sindhi_char:
                st.markdown(f'<h2 style="color:black; font-family: Noto Nastaliq Urdu, Noto Sans Arabic, Arial;">Sindhi Character: {sindhi_char}</h2>', unsafe_allow_html=True)
    except Exception as e:
        st.error(f'Error in prediction: {e}')
else:
    st.info('Please upload an image file.')
