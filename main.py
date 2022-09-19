import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
import requests
import os
import cv2
st.write("Crop Recommendation and Disease Detection using Deep Neural Networks (ANN and CNN)")
tab1, tab2 = st.tabs(["Crop", "Disease"])
model = keras.models.load_model("./dl_model.h5")
model_2 = keras.models.load_model("./dl_model_2.h5")


def get_key(val, dict):
    for key, value in dict.items():
         if val == value:
             return key

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)    

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:
                f.write(chunk)

classes = {'rice': 0, 'maize': 1, 'chickpea': 2, 'kidneybeans': 3, 'pigeonpeas': 4,
       'mothbeans': 5, 'mungbean': 6, 'blackgram': 7, 'lentil': 8, 'pomegranate': 9,
       'banana': 10, 'mango': 11, 'grapes': 12, 'watermelon': 13, 'muskmelon': 14, 'apple': 15,
       'orange': 16, 'papaya': 17, 'coconut': 18, 'cotton': 19, 'jute': 20, 'coffee': 21}

with tab1:
    st.write(" Crop Recommendation")
    N = st.text_input("Nitrogen")
    P = st.text_input("Phosphorus")
    K = st.text_input("Potassium")
    temperature = st.text_input("Temperature")
    humidity = st.text_input("Humidity")
    ph = st.text_input("pH")
    rainfall = st.text_input("Rainfall")

    if st.button("Get Crop"):
        prediction = np.argmax(model.predict([[int(N), int(P), int(K), float(temperature), float(humidity), float(ph), float(rainfall)]]))
        predicted_class = get_key(prediction, classes)
        st.write(f"## Recommended Crop is {predicted_class}")
    
with tab2:
    st.write(" Plant Disease Detection")
    url = st.text_input("Url")
    # https://drive.google.com/file/d/1LjsIh50M3B87LdwbgATVucfFvFpaB2o7/view?usp=sharing
    if st.button("Get Status of Crop"):
        file_id = url.split("/")[-2]
        download_file_from_google_drive(file_id, "temp.jpg")
        img = cv2.imread("temp.jpg")
        img = cv2.resize(img, (100, 100))
        img = np.array(img)
        img = img.astype("float32")
        img /= 255
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        prediction = model_2.predict(gray.reshape(1, 100, 100, 1), batch_size=1)
        result = "unhealthy"
        if prediction[0][0] >= 0.5:
            result = "healthy"
        st.image("temp.jpg", width=100)
        st.write(f"## Crop is {result}")
        os.remove("temp.jpg")

