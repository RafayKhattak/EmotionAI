# Import necessary libraries and modules
import numpy as np
import cv2
import streamlit as st
from tensorflow import keras
from keras.models import load_model
from keras.utils import image_utils
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration, WebRtcMode

# Set page configuration and title for Streamlit
st.set_page_config(page_title="EmotionAI", page_icon="👁", layout="wide")

# Add header with title and description
st.markdown(
    '<p style="display:inline-block;font-size:40px;font-weight:bold;">🤗EmotionAI </p>'
    ' <p style="display:inline-block;font-size:16px;">EmotionAI is an Emotion Detection tool powered by artificial intelligence technology. It leverages a powerful CNN model to analyze facial expressions and accurately recognize emotions such as happiness, sadness, anger, and more. <br><br></p>',
    unsafe_allow_html=True
)

# Load model
emotion_dict = {0:'Angry',1:'Disgust',2:'Fear',3:'Happy',4:'Neutral', 5:'Sad', 6:'Surprise'}
classifier = load_model("model.h5")  # Load pre-trained model

# Load face cascade classifier
try:
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
except Exception:
    st.write("Error loading cascade classifiers")

RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

class Faceemotion(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Convert image to grayscale
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detect faces in the grayscale image
        faces = face_cascade.detectMultiScale(image=img_gray, scaleFactor=1.3, minNeighbors=5)
        
        # Process each detected face
        for (x, y, w, h) in faces:
            cv2.rectangle(img=img, pt1=(x, y), pt2=(x + w, y + h), color=(255, 0, 0), thickness=2)
            roi_gray = img_gray[y:y + h, x:x + w]
            
            # Resize the region of interest (ROI) to match the input size of the model
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
            
            if np.sum([roi_gray]) != 0:
                roi = roi_gray.astype('float') / 255.0
                roi = image_utils.img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)
                
                # Make prediction using the loaded model
                prediction = classifier.predict(roi)[0]
                maxindex = int(np.argmax(prediction))
                finalout = emotion_dict[maxindex]
                output = str(finalout)
            
            label_position = (x, y)
            cv2.putText(img, output, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return img


st.header("Webcam Live Feed")
st.write("Click start to detect your Face emotion")
webrtc_streamer(key="example", mode=WebRtcMode.SENDRECV, rtc_configuration=RTC_CONFIGURATION,
video_processor_factory=Faceemotion)

# Hide Streamlit header, footer, and menu
hide_st_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
"""

# Apply CSS code to hide header, footer, and menu
st.markdown(hide_st_style, unsafe_allow_html=True)
