import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Load the trained model
model = tf.keras.models.load_model("final_hidden_mickey_model.h5")

# Define image properties
IMG_SIZE = 96
test_folder = "test_images"

# Ensure the test folder exists
if not os.path.exists(test_folder):
    print(f"Folder '{test_folder}' not found.")
    exit()

# Load and preprocess test images
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))  # Load image and resize
    img_array = img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Classify images
for filename in os.listdir(test_folder):
    if filename.lower().endswith(('.png')):  # Supported formats
        img_path = os.path.join(test_folder, filename)
        img_array = preprocess_image(img_path)

        # Predict the class
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction)  # Get the class with the highest probability
        confidence = np.max(prediction) * 100  # Convert to percentage

        # Display results
        label = "Hidden Mickey" if predicted_class == 1 else "No Mickey"
        print(f"{filename}: {label} ({confidence:.2f}%)")
