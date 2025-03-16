import streamlit as st # importing streamlit library
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow import keras


# Load the pre-trained model and cache it (so that we don't have to load it every time)
@st.cache_resource
def load_model():
    return keras.models.load_model('efficientnet_model2.keras')


# Define Animals-10 class names
class_names = ['cane', 'cavallo', 'elefante', 'farfalla', 'gallina', 'gatto', 'mucca', 'pecora', 'ragno', 'scoiattolo']


# Set up the Streamlit app
st.title('Animal Image Classifier with EfficientNet model')
st.write('Upload an image and the model will predict its class')

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

def preprocess_image(image):
    image = image.convert('RGB')
    image = image.resize((224, 224)) 
    image_array = np.array(image) / 255.0  
    image_array = np.expand_dims(image_array, axis=0)  
    return image_array

# If an image is uploaded, we are going to make a prediction and display the results
if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_container_width=True)
    
    # Preprocess the image
    image_array = preprocess_image(image)
    
    # Load model and make prediction
    model = load_model()
    predictions = model.predict(image_array)
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = np.max(predictions[0]) * 100
    
    # Display results
    st.write(f"Prediction: {predicted_class}")
    st.write(f"Confidence: {confidence:.2f}%")
    
    # Display bar chart of all predictions
    st.bar_chart({class_names[i]: float(predictions[0][i]) for i in range(10)})