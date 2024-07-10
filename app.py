import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
# Set the configuration of the Streamlit app
st.set_page_config(page_title="Emerging Technology 2 in CpE", layout="wide")

# Set the title and introductory markdown for the app
st.title("Model Deployment Project")
st.markdown("""

Name:
**William Laurence M. Ramos**

Course/Section: **CPE019** | **CPE31S1**

Date Submitted: **July 12, 2024**
""")

# Function to load the pre-trained model with caching to avoid reloading it multiple times
@st.cache(allow_output_mutation=True)
@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model("final_model.h5")
    return model
    
# Define the class names corresponding to the model's output classes
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Load the model by calling the load_model function
model = load_model()

# Set the title for the image classification section
st.title("Fashion Item Classification")
st.write("Upload an image to classify the type of fashion item.")

# Create a file uploader widget to allow users to upload an image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Function to preprocess the uploaded image and make a prediction using the model
def import_and_predict(image_data, model):
    size = (28, 28)  
    image = ImageOps.fit(image_data, size)
    img = np.asarray(image.convert('L'))  
    img = np.expand_dims(img, axis=0)
    img = np.expand_dims(img, axis=-1)
    img = img / 255.0
    prediction = model.predict(img)
    return prediction
    
# Check if an image has been uploaded
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Classifying...")

# Make a prediction on the uploaded image
    prediction = import_and_predict(image, model)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)

# Display the prediction and confidence
    st.write(f"Prediction: {predicted_class}")
    st.write(f"Confidence: {confidence:.2f}")
