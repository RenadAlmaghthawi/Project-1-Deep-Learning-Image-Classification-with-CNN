import streamlit as st  # importing streamlit library
import tensorflow as tf
import numpy as np
from PIL import Image

@st.cache_resource
def load_model():
    return tf.keras.models.load_model('efficientnet_model3.h5')

class_names = [
    'cane', 'cavallo','elefante', 'farfalla', 'gallina',
    'gatto', 'mucca', 'pecora', 'ragno', 'scoiattolo'
]

# Set up the Streamlit app
st.title('Animal Image Classifier')
st.write('Upload an image and the model will predict its class')

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

def preprocess_image(image):
    image = image.convert('RGB')
    image = image.resize((224, 224))  # Ensure the size matches the model input
    image_array = np.array(image)
    image_array = image_array.astype('float32') / 255.0
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
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
    st.write(f"{predictions}")
    # Get the predicted class and confidence
    predicted_class_index = np.argmax(predictions[0])
    predicted_class = class_names[predicted_class_index]
    confidence = predictions[0][predicted_class_index] * 100
    
    # Display results
    st.write(f"Prediction: {predicted_class}")
    st.write(f"Confidence: {confidence:.2f}%")

    # Display bar chart of all predictions
    st.bar_chart({class_names[i]: float(predictions[0][i]) for i in range(10)})
