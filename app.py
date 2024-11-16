# app.py (FastAPI Backend)
from fastapi import FastAPI, UploadFile, File
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io

app = FastAPI()

# Load the pre-trained model
model = load_model("res50model.h5")
img_height, img_width = 256, 256

@app.get("/")
async def root():
    return {"message": "Welcome to the Potato Classification API!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read the image file directly into a PIL image
        image = Image.open(file.file)

        # Resize and process the image to match model input
        image = image.resize((img_height, img_width))
        image = np.array(image)

        # Ensure the image is in RGB format (if grayscale or RGBA)
        if image.shape[-1] == 4:  # RGBA to RGB
            image = image[..., :3]
        elif len(image.shape) == 2:  # Grayscale to RGB
            image = np.stack((image,)*3, axis=-1)

        # Normalize the image and expand dimensions (for batch size of 1)
        image = np.expand_dims(image, axis=0) / 256.0

        # Predict using the model
        predictions = model.predict(image)
        predicted_class = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class])

        return {"predicted_class": int(predicted_class), "confidence": confidence}
    
    except Exception as e:
        return {"error": f"Prediction error: {str(e)}"}
#