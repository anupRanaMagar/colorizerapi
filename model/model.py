from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from PIL import Image
import numpy as np
from keras.preprocessing.image import load_img, img_to_array, array_to_img
import numpy as np
TARGET_SIZE = 512
#




def resize_img_to_target_size(img_path, target_size, num_channels):
  
    if isinstance(img_path, Image.Image):
        img = img_path
    else:
        color_mode = 'grayscale' if num_channels == 1 else 'rgb'
        img = Image.open(img_path).convert(color_mode.upper())
    
    # Convert to grayscale if num_channels == 1
    if num_channels == 1:
        img = img.convert('L')  # Convert to grayscale ('L' mode in PIL)
    
    # Convert the image to a numpy array
    img_array = img_to_array(img)
    height, width = img_array.shape[:2]
    
    # Calculate scaling factor and scaled dimensions
    max_dim = max(width, height)
    scale_factor = target_size / max_dim
    scaled_width, scaled_height = int(width * scale_factor), int(height * scale_factor)
    
    # Resize the image while maintaining aspect ratio
    img_resized = img.resize((scaled_width, scaled_height))
    img_resized_array = img_to_array(img_resized) / 255.0  # Normalize the image
    
    # Create a blank canvas of the target size
    resized_img_array = np.zeros((target_size, target_size, num_channels))
    
    # Compute padding to center the image on the canvas
    x_offset = (target_size - scaled_width) // 2
    y_offset = (target_size - scaled_height) // 2
    
    # Place the resized image on the center of the canvas
    resized_img_array[y_offset:y_offset + scaled_height, 
                      x_offset:x_offset + scaled_width, :] = img_resized_array
    
    # If grayscale (num_channels == 1), add a channel dimension
    if num_channels == 1:
        resized_img_array = resized_img_array[:, :, 0:1]
    
    return resized_img_array



def preprocess(img_path, num_channels, target_size=TARGET_SIZE):
  img_arr = resize_img_to_target_size(img_path, target_size, num_channels)
  img_arr = img_arr.reshape( (1, ) + img_arr.shape )
  return img_arr

   



   
  



