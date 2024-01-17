from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import requests
import os

# run tf serving in Docker:
# docker run -t --rm -p 8502:8501   -v /mnt/f/hackaton/api/saved_models/potato:/models/potato   -e MODEL_NAME=potato_model   -t tensorflow/serving --model_config_file=/models/potato/tf.config

app = FastAPI()

origins = [
    os.getenv("CORS_URI_1", "http://localhost"),
    os.getenv("CORS_URI_2", "http://localhost:80"),
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# versioning
# Set default endpoints or use environment variables if available
endpoint_v1 = os.getenv("ENDPOINT_V1", "http://172.18.0.2:8501/v1/models/potato_model/versions/1:predict")
endpoint_v2 = os.getenv("ENDPOINT_V2", "http://172.18.0.2:8501/v1/models/potato_model/versions/2:predict")

CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.get("/ping")
async def ping():
    return "Pong."

# endpoint for version 1
@app.post("/predict/v1")
async def predict(
    file: UploadFile = File(..., max_bytes=40000000)
):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)

    json_data = {
        "instances": img_batch.tolist()
    }

    response = requests.post(endpoint_v1, json=json_data)
    prediction = np.array(response.json()["predictions"][0])

    predicted_class = CLASS_NAMES[np.argmax(prediction)]
    confidence = np.max(prediction)

    return {
        "class": predicted_class,
        "confidence": float(confidence)
    }

# endpoint for version 2
@app.post("/predict/v2")
async def predict(
    file: UploadFile = File(..., max_bytes=40000000)
):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)

    json_data = {
        "instances": img_batch.tolist()
    }

    response = requests.post(endpoint_v2, json=json_data)
    prediction = np.array(response.json()["predictions"][0])

    predicted_class = CLASS_NAMES[np.argmax(prediction)]
    confidence = np.max(prediction)

    return {
        "class": predicted_class,
        "confidence": float(confidence)
    }

if __name__ == "__main__":
    uvicorn.run(app, host = "0.0.0.0", port = 8080)
