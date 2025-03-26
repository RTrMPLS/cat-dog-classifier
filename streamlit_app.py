import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
from PIL import Image

# Load your trained model (same name as your uploaded .h5 file)
model = load_model('cats_vs_dogs_mobilenetv2.h5')

# Streamlit UI
st.title("🐶🐱 Cat vs Dog Classifier")
st.write("Upload an image and I'll try to tell you if it's a cat or a dog.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_column_width=True)

    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    prediction = model.predict(img_array)[0][0]
    label = "Dog" if prediction > 0.5 else "Cat"
    confidence = prediction if prediction > 0.5 else 1 - prediction
    threshold = 0.98  # set your desired minimum confidence level

    if confidence >= threshold:
        st.markdown(f"### Prediction: **{label}** ({confidence:.2%} confidence)")
    else:
        st.markdown(f"⚠️ I'm **not confident** this is a cat or a dog. ({confidence:.2%})")
        st.markdown("Please try uploading a clearer image of a **cat** or **dog**.")