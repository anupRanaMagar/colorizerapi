from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from io import BytesIO
from PIL import Image
from fastapi.middleware.cors import CORSMiddleware
from keras.preprocessing.image import load_img, img_to_array, array_to_img
from model.model import preprocess
from keras.models import load_model
from pathlib import Path

BASE_DIR = Path(__file__).resolve(strict=True).parent
print(BASE_DIR)


app = FastAPI()

model = load_model(f"{BASE_DIR}/model/model.h5", compile=False)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # List your allowed origins here
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)
@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/colorize")
async def colorize_image_endpoint(file: UploadFile = File(...)):
    try:
        # Read the uploaded image file
        image_bytes = await file.read()

        # print(image_bytes)
        # Open the image from the uploaded bytes and convert to RGB
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        original_size = image.size

        # Call the colorization function (assume this is the function defined earlier)
        img_arr = preprocess(image, 1)
        colorized_image = array_to_img(model.predict(img_arr)[0] * 255)

        colorized_image = colorized_image.resize(original_size)

        # Save the colorized image into a BytesIO object
        colorized_image_io = BytesIO()
        colorized_image.save(colorized_image_io, format="JPEG" ,quality=85)
        colorized_image_io.seek(0)  # Reset the stream position to the start

        # Return the colorized image as a streaming response
        return StreamingResponse(colorized_image_io, media_type="image/jpeg")

    except Exception as e:
        # Handle any errors that occur during the process
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")
