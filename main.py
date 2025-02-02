from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
import io
import cv2
import numpy as np
import matplotlib.pyplot as plt

app = FastAPI()

# Function to convert the processed image to an in-memory file
def image_to_bytes(image):
    is_success, buffer = cv2.imencode(".png", image)
    io_buf = io.BytesIO(buffer)
    return io_buf

@app.post("/process_image/")
async def process_image(file: UploadFile = File(...)):
    # Read the image from the uploaded file
    img = await file.read()
    np_img = np.frombuffer(img, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian Blur
    blur = cv2.GaussianBlur(gray, (3, 3), cv2.BORDER_DEFAULT)

    # Apply Canny Edge Detection
    canny = cv2.Canny(img, 150, 175)

    # Prepare the results (as PNG images)
    gray_image = image_to_bytes(gray)
    blur_image = image_to_bytes(blur)
    canny_image = image_to_bytes(canny)

    # Return the processed images as a streaming response
    return {
        "gray_image": StreamingResponse(gray_image, media_type="image/png"),
        "blur_image": StreamingResponse(blur_image, media_type="image/png"),
        "canny_image": StreamingResponse(canny_image, media_type="image/png"),
    }
