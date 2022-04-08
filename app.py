import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing import image
import streamlit as st
from PIL import Image

st.set_page_config(page_title='African Wildlife', page_icon='favicon.png')
st.title('African Wildlife Animal Classifier')
st.subheader('Upload either Buffalo/Elephant/Rhino/Zebra image for prediction')
image_file = st.file_uploader("Upload Images", type=["png","jpg","jpeg"])

@st.cache(show_spinner=False)
def load_image(image_file):
	img = Image.open(image_file)
	return img

@st.cache(show_spinner=False)
def predict(img):
    model = keras.models.load_model('model.h5')
    resized = img.resize((256,256))
    array = image.img_to_array(resized)
    expanded = np.expand_dims(array,axis=0)
    proba = model.predict(expanded)
    return ['Buffalo','Elephant','Rhino','Zebra'][proba.argmax()]

if st.button('Predict'):
    if image_file is not None:
        img = load_image(image_file)
        st.image(img,width=256)
        with st.spinner('Predicting...'):
            prediction = predict(img)  
            st.success(prediction)
