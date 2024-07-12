import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

st.set_page_config(page_title="Emerging Technology 2 in CpE", layout="wide")


st.title("Model Deployment Project")
st.markdown("""
Name:
**William Laurence M. Ramos**

Course/Section: **CPE019** | **CPE32S1**

Date Submitted: **July, 12, 2024**
""")

@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model("final_model.h5")
    return model

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

model = load_model()

st.title("Fashion Item Classification")
st.write("Upload an image to classify the type of fashion item.(It only recognize black and white fashion items")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

def import_and_predict(image_data, model):
    size = (28, 28)  
    image = ImageOps.fit(image_data, size)
    img = np.asarray(image.convert('L'))  
    img = np.expand_dims(img, axis=0)
    img = np.expand_dims(img, axis=-1)
    img = img / 255.0
    prediction = model.predict(img)
    return prediction

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    prediction = import_and_predict(image, model)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)

    st.write(f"Prediction: {predicted_class}")
    st.write(f"Confidence: {confidence:.2f}")
