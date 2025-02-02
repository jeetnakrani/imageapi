from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
import io
import cv2
import numpy as np
import os

app = FastAPI()

# Function to convert the processed image to an in-memory file
def image_to_file(image, filename):
    # Save image to a temporary file
    path = f"/tmp/{filename}.png"
    cv2.imwrite(path, image)
    return path

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

    # Save processed images to temporary files
    gray_image_path = image_to_file(gray, "gray_image")
    blur_image_path = image_to_file(blur, "blur_image")
    canny_image_path = image_to_file(canny, "canny_image")

    # Return the processed images as file responses
    return {
        "gray_image": FileResponse(gray_image_path, media_type="image/png", filename="gray_image.png"),
        "blur_image": FileResponse(blur_image_path, media_type="image/png", filename="blur_image.png"),
        "canny_image": FileResponse(canny_image_path, media_type="image/png", filename="canny_image.png"),
    }

# Run the application with Uvicorn:
# uvicorn filename:app --reload
