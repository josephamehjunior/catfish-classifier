import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import time
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras import backend as K

# Get the absolute path to the assets directory
assets_dir = os.path.join(os.path.dirname(__file__), 'assets')
logo_path = os.path.join(assets_dir, 'logo.jpeg')

# Center align the logo with columns
col1, col2, col3 = st.columns([1,2,1])
with col2:
    st.image(logo_path, width=300, use_container_width=True)

# Define the input shape for the model
input_shape = (224, 224, 3)

# Clear the session before loading the model to prevent internal TensorFlow issues
K.clear_session()

def load_inceptionv3_model():
    """Load the InceptionV3 model with pre-trained weights on ImageNet."""
    inceptionv3_base = InceptionV3(weights='imagenet', include_top=False, input_shape=input_shape)
    
    for layer in inceptionv3_base.layers:
        layer.trainable = False

    x = inceptionv3_base.output
    x = Flatten()(x)

    # Classification branch
    classification_branch = Dense(64, activation='relu')(x)
    classification_output = Dense(3, activation='softmax', name='classification_output')(classification_branch)

    # Regression branch
    regression_branch = Dense(64, activation='relu')(x)
    regression_output = Dense(3, activation='linear', name='regression_output')(regression_branch)

    model = Model(inputs=inceptionv3_base.input, outputs=[classification_output, regression_output])
    model.compile(
        optimizer='adam',
        loss={
            'classification_output': 'sparse_categorical_crossentropy',
            'regression_output': 'mse'
        },
        metrics={
            'classification_output': ['accuracy'],
            'regression_output': ['mae']
        }
    )

    try:
        model.load_weights("InceptionV3_best_model.weights.h5")
    except Exception as e:
        st.error(f"Error loading model weights: {e}")
        st.stop()

    return model

def predict_fish_attributes(image_path, model):
    """Predict fish class and attributes (SL, TL, Weight) from an image."""
    try:
        img = Image.open(image_path).resize((224, 224)).convert("RGB")
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        predictions = model.predict(img_array)
        classification_pred = predictions[0]
        regression_pred = predictions[1]

        predicted_class = np.argmax(classification_pred, axis=1)[0]
        return predicted_class, regression_pred[0], None
    except FileNotFoundError:
        return None, None, "Image file not found."
    except Exception as e:
        return None, None, str(e)

# Load the model
model_inceptionv3 = load_inceptionv3_model()

# Streamlit App Interface
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>Catfish Attribute Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Upload a catfish image or capture one using your webcam to predict its class (fingerling, juvenile, adult), Standard Length (SL_cm), Total Length (TL_cm), and Weight (Weight_g).</p>", unsafe_allow_html=True)

# Section for selecting image source (upload or webcam capture)
st.header("Select Image Source")
image_source = st.radio("Choose how to upload the image", ("Upload Image", "Capture from Webcam"))

# Initialize variables
image_file = None

# Handle image source selection
if image_source == "Upload Image":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image_file = uploaded_file
        st.image(image_file, caption="Uploaded Image.", use_container_width=True)
elif image_source == "Capture from Webcam":
    camera_input = st.camera_input("Take a picture")
    if camera_input is not None:
        image_file = camera_input
        st.image(image_file, caption="Captured Image.", use_container_width=True)

# Process the image if available
if image_file is not None:
    st.write("Processing image and making predictions...")
    temp_image_path = "temp_image.jpg"
    with open(temp_image_path, "wb") as f:
        f.write(image_file.getbuffer())

    predicted_class_label, predicted_regression_values, error = predict_fish_attributes(temp_image_path, model_inceptionv3)
    os.remove(temp_image_path)

    if error is None:
        # Map predicted class label back to class name
        label_to_class = {0: "fingerling", 1: "juvenile", 2: "adult"}
        predicted_class_name = label_to_class.get(predicted_class_label, "Unknown")

        st.subheader("Prediction Results")
        st.write(f"**Predicted Class:** {predicted_class_name} (Label: {predicted_class_label})")
        st.write(f"**Predicted Standard Length (SL_cm):** {predicted_regression_values[0]:.2f} cm")
        st.write(f"**Predicted Total Length (TL_cm):** {predicted_regression_values[1]:.2f} cm")
        st.write(f"**Predicted Weight (Weight_g):** {predicted_regression_values[2]:.2f} g")
    else:
        st.error(f"Prediction failed: {error}")
else:
    st.info("Please select an image source and upload or capture an image to get predictions.")
