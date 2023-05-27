import os
import sys
import glob
import csv
import numpy as np
from PIL import Image
import tensorflow as tf

# Load the saved model
model = tf.keras.models.load_model('./model')

# Mapping table for converting predicted class index to ASCII index
mapping_table = {
    0: 48, 1: 49, 2: 50, 3: 51, 4: 52, 5: 53, 6: 54, 7: 55, 8: 56, 9: 57,
    10: 65, 11: 66, 12: 67, 13: 68, 14: 69, 15: 70, 16: 71, 17: 72, 18: 73, 19: 74,
    20: 75, 21: 76, 22: 77, 23: 78, 24: 79, 25: 80, 26: 81, 27: 82, 28: 83, 29: 84,
    30: 85, 31: 86, 32: 87, 33: 88, 34: 89, 35: 90, 36: 97, 37: 98, 38: 100, 39: 101,
    40: 102, 41: 103, 42: 104, 43: 110, 44: 113, 45: 114, 46: 116
}

# Function to preprocess the image
def preprocess_image(image_path):
    image = Image.open(image_path).convert('L')  # Convert image to grayscale

 # Get the pixel values of the grayscale image
    pixels = image.getdata()

    # Count the number of black and white pixels
    black_count = sum(1 for pixel in pixels if pixel < 100)
    white_count = sum(1 for pixel in pixels if pixel > 150)

    # Determine the dominant color

    if black_count < white_count:
        # Transform black pixels to white and vice versa
        transformed_image = image.point(lambda pixel: 255 if pixel < 150 else 0)
        
    else:
        # No transformation needed
        transformed_image = image.point(lambda pixel: 255 if pixel > 150 else 0)

    image = transformed_image.resize((28, 28))  # Resize image to match model input shape
    image_array = np.array(image) / 255.0  # Normalize pixel values to [0, 1]
    image_array = np.expand_dims(image_array, axis=-1)
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array

# Get the path to the directory with image samples from CLI argument
directory_path = sys.argv[1]

# Find all image files in the directory
image_files = glob.glob(os.path.join(directory_path, '*.png')) + glob.glob(os.path.join(directory_path, '*.jpg'))

# Create a CSV writer to print the output in CSV format
csv_writer = csv.writer(sys.stdout)

# Process each image and print the output
for image_file in image_files:
    # Preprocess the image
    image_array = preprocess_image(image_file)

    # Perform inference
    predictions = model.predict(image_array, verbose=0)
    predicted_class = np.argmax(predictions[0])

    # Get the ASCII index from the mapping table
    ascii_index = mapping_table.get(predicted_class, None)



    if ascii_index is not None:
        # Write the output in CSV format
        csv_writer.writerow([ascii_index, os.path.abspath(image_file)])
