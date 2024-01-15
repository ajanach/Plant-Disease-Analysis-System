from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

app = FastAPI()

# load models
PROD_MODEL = tf.keras.models.load_model("saved_models/potato/1") # trained on 50 epochs
BETA_MODEL = tf.keras.models.load_model("saved_models/potato/2") # trained on 60 epochs

CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

@app.get("/ping")
async def ping():
    return "Pong."

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

# model version 1
@app.post("/predict/v1")
async def predict(file: UploadFile): # argument will be a file sent by mobile app or website
    image = read_file_as_image(await file.read())
    image_batch = np.expand_dims(image, 0)

    predictions = PROD_MODEL.predict(image_batch)

    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])

    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }

# model version 2
@app.post("/predict/v2")
async def predict(file: UploadFile): # argument will be a file sent by mobile app or website
    image = read_file_as_image(await file.read())
    image_batch = np.expand_dims(image, 0)

    predictions = BETA_MODEL.predict(image_batch)

    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])

    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8888)
