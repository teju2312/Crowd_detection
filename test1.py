import streamlit as st
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.inception_v3 import preprocess_input as inception_preprocess

# Load the saved model
loaded_model = load_model(r"C:\Users\dell\OneDrive\Documents\all_document\Desktop\ResNet50\ensemble_model.h5")

# Function to preprocess the uploaded image
def preprocess_image(image_path, model_type='resnet'):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Apply the preprocessing function based on the model type
    if model_type == 'resnet':
        img_array = resnet_preprocess(img_array)
    elif model_type == 'inception':
        img_array = inception_preprocess(img_array)
    
    return img_array

# Function to make predictions
def predict_count(image_array_resnet, image_array_inception):
    # Make predictions using the loaded model
    prediction = loaded_model.predict([image_array_resnet, image_array_inception])

    return np.floor(prediction[0][0])  # Modify this according to your model's output

# Main Streamlit app
def main():
    st.title('Crowd Count Prediction')
    st.write('Upload an image and get the predicted crowd count.')

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
        st.write("")
        st.write("Predicting...")

        # Preprocess the uploaded image for both ResNet and Inception
        img_array_resnet = preprocess_image(uploaded_file, model_type='resnet')
        img_array_inception = preprocess_image(uploaded_file, model_type='inception')

        # Make predictions
        prediction = predict_count(img_array_resnet, img_array_inception)

        st.write(f'Predicted crowd count: {prediction:.0f}')  # Display predicted count as integer

if __name__ == "__main__":
    main()
