import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2
import pandas as pd

# Load trained model
model = tf.keras.models.load_model("lung_disease_model.h5")

classes = ["LUNG_CANCER","NORMAL"]

st.title("🫁 AI Lung Disease Detection System")

st.write("Upload a Chest X-ray image to detect lung disease using AI.")

# Upload image
uploaded_file = st.file_uploader("Upload Chest X-ray",type=["jpg","png","jpeg"])

# Select district
district = st.selectbox(
    "Select your district",
    ["Chennai","Coimbatore","Madurai","Trichy","Salem","Tirunelveli","Vellore"]
)

def preprocess(img):

    img = np.array(img)
    img = cv2.resize(img,(224,224))
    img = img/255.0
    img = np.expand_dims(img,axis=0)

    return img


if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(image,caption="Uploaded X-ray",use_column_width=True)

    img = preprocess(image)

    prediction = model.predict(img)

    disease = classes[np.argmax(prediction)]

    confidence = np.max(prediction)*100

    st.subheader("Prediction Result")

    st.success(f"Disease: {disease}")

    st.write(f"Confidence Level: {confidence:.2f}%")

    # Probability chart
    st.subheader("Prediction Probability")

    prob_data = pd.DataFrame(
        prediction,
        columns=classes
    )

    st.bar_chart(prob_data)

    # Doctor recommendation
    st.subheader("Recommended Doctor")

    if disease == "LUNG_CANCER":
        st.write("Consult an **Oncologist (Cancer Specialist)**")
    else:
        st.write("No disease detected. Maintain healthy lifestyle.")

    # Hospital suggestion
    st.subheader("Suggested Hospitals")

    if district == "Chennai":
        st.write("Apollo Hospitals Chennai")
        st.write("Government General Hospital Chennai")

    elif district == "Coimbatore":
        st.write("KMCH Hospital")
        st.write("PSG Hospitals")

    elif district == "Madurai":
        st.write("Meenakshi Mission Hospital")

    elif district == "Trichy":
        st.write("Kauvery Hospital")

    elif district == "Salem":
        st.write("SKS Hospital")

    elif district == "Tirunelveli":
        st.write("Shifa Hospital")

    elif district == "Vellore":
        st.write("Christian Medical College Vellore")

    # Treatment suggestion
    st.subheader("Health Advice")

    if disease == "LUNG_CANCER":
        st.write("Seek immediate medical consultation. Early diagnosis improves treatment success.")
    else:
        st.write("Your lungs appear healthy. Continue regular health checkups.")
