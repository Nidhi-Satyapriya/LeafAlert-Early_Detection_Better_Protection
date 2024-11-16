import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Load the model
model = tf.keras.models.load_model("res50model.h5")

# Define class names for each index (update this list based on your model's classes)
class_names = ["Healthy", "Early Blight", "Late Blight"]

# Define image preprocessing function
def preprocess_image(image_path, target_size=(256, 256)):
    image = Image.open(image_path)
    image = image.resize(target_size)
    image_array = np.array(image) / 256.0  # Normalize the image
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array

# Function to get prediction
def predict(image_path):
    # Preprocess the image
    processed_image = preprocess_image(image_path)
    
    # Make a prediction
    prediction = model.predict(processed_image)
    class_idx = np.argmax(prediction, axis=1)[0]
    output_class = class_names[class_idx]
    
    return output_class, prediction

# Main function to upload and test an image
if __name__ == "__main__":
    # Specify the path to the image
    image_path = (r"./testing_imgs/Healthy_16.jpg")
    
    if os.path.exists(image_path):
        # Run prediction
        output_class, prediction = predict(image_path)
        
        # Output result
        print(f"The predicted class is: {output_class}")
        print(f"Prediction confidence: {prediction}")
    else:
        print("Image path does not exist. Please check the path and try again.")
# Load the model
# model = tf.keras.models.load_model("res50model.h5")

# # Get the input shape
# input_shape = model.input_shape
# print("Model Input Shape:", input_shape)
