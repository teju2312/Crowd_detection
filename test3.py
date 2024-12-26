import streamlit as st
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.inception_v3 import preprocess_input as inception_preprocess
from tf_keras_vis.gradcam import Gradcam, GradcamPlusPlus
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
from tf_keras_vis.utils.scores import CategoricalScore
import matplotlib.pyplot as plt

# Load the saved model
loaded_model = load_model(r"C:\Users\dell\OneDrive\Documents\all_document\Desktop\ResNet50\ensemble_model.h5")

# Function to preprocess the uploaded image
def preprocess_image(image_file, model_type='resnet'):
    img = image.load_img(image_file, target_size=(224, 224))
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

# Function to generate Grad-CAM heatmap using tf-keras-vis
def generate_gradcam_heatmap_tfkerasvis(model, image_arrays, method='gradcam'):
    if method == 'gradcam':
        explainer = Gradcam(model, model_modifier=ReplaceToLinear())
    elif method == 'gradcam++':
        explainer = GradcamPlusPlus(model, model_modifier=ReplaceToLinear())
    
    score = CategoricalScore(0)  # Assuming the class index is 0; modify based on your requirements
    grid = explainer(score, image_arrays)
    heatmap = grid[0]
    
    # Normalize the heatmap
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
    
    # Convert single-channel heatmap to 3-channel image
    heatmap = np.uint8(255 * heatmap)
    heatmap = np.stack([heatmap] * 3, axis=-1)  # Convert to 3-channel
    
    # Reshape heatmap to remove the batch dimension
    heatmap = heatmap.reshape(heatmap.shape[1:])
    
    # Print heatmap information for debugging
    print("Heatmap shape:", heatmap.shape)
    print("Heatmap values min/max:", np.min(heatmap), np.max(heatmap))

    return heatmap


# Main Streamlit app
def main():
    st.title('Crowd Count Prediction with Visualization Maps')
    st.write('Upload an image and get the predicted crowd count along with various visualization maps.')

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image_path = uploaded_file.name
        with open(image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.image(image_path, caption='Uploaded Image.', use_column_width=True)
        st.write("")
        st.write("Predicting...")

        try:
            # Preprocess the uploaded image for both ResNet and Inception
            img_array_resnet = preprocess_image(image_path, model_type='resnet')
            img_array_inception = preprocess_image(image_path, model_type='inception')

            # Make predictions
            prediction = predict_count(img_array_resnet, img_array_inception)

            st.write(f'Predicted crowd count: {prediction:.0f}')  # Display predicted count as integer

            # Generate Grad-CAM heatmap for ResNet using tf-keras-vis
            heatmap_gradcam = generate_gradcam_heatmap_tfkerasvis(loaded_model, [img_array_resnet, img_array_inception], method='gradcam')
            st.image(heatmap_gradcam, caption='Grad-CAM Heatmap', use_column_width=True)

            # Generate Smooth Grad-CAM heatmap for ResNet using tf-keras-vis
            heatmap_smooth_gradcam = generate_gradcam_heatmap_tfkerasvis(loaded_model, [img_array_resnet, img_array_inception], method='gradcam++')
            st.image(heatmap_smooth_gradcam, caption='Smooth Grad-CAM Heatmap', use_column_width=True)


        except Exception as e:
            st.error(f"Error: {e}")

if __name__ == "__main__":
    main()
