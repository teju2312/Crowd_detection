import streamlit as st
from keras.models import load_model, Model
from keras.preprocessing import image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
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

# Function to generate Grad-CAM heatmap
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, classifier_layer_names):
    grad_model = Model(
        inputs=[model.input[0], model.input[1]], 
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

# Function to display Grad-CAM heatmap
def display_gradcam(image_path, heatmap, cam_path="cam.jpg", alpha=0.4):
    img = image.load_img(image_path)
    img = image.img_to_array(img)

    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed_img = heatmap * alpha + img
    cv2.imwrite(cam_path, superimposed_img)

    return cam_path

# Main Streamlit app
def main():
    st.title('Crowd Count Prediction with Grad-CAM')
    st.write('Upload an image and get the predicted crowd count along with a heatmap.')

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

        # Generate Grad-CAM heatmap for ResNet
        heatmap_resnet = make_gradcam_heatmap([img_array_resnet, img_array_inception], loaded_model, "conv5_block3_out", ["avg_pool", "dense"])

        # Display Grad-CAM heatmap
        cam_path = display_gradcam(uploaded_file, heatmap_resnet)
        st.image(cam_path, caption='Grad-CAM Heatmap', use_column_width=True)

if __name__ == "__main__":
    main()
