# classify_images.py
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
from pathlib import Path
import json
import sys

# --- Configuration ---
IMG_DIR = Path("./images")
OUTPUT_FILE = Path("./image_categories.json")
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg'}

def classify_images():
    """
    Scans the image directory, uses a pre-trained MobileNetV2 model to classify
    each image, and saves the results to a JSON file.
    """
    # --- 1. Load the Pre-trained Model ---
    print("Loading MobileNetV2 model...")
    # The 'weights' parameter specifies which weight checkpoint to use.
    # 'imagenet' means we are using weights from a model pre-trained on the ImageNet dataset.
    # The first time this runs, it will download the model weights (approx. 14MB).
    try:
        model = MobileNetV2(weights='imagenet')
    except Exception as e:
        print(f"Error loading model: {e}", file=sys.stderr)
        print("Please ensure you have an active internet connection the first time you run this.", file=sys.stderr)
        sys.exit(1)
        
    print("Model loaded successfully.")

    # --- 2. Find all images in the directory ---
    image_paths = [p for p in IMG_DIR.glob('**/*') if p.suffix.lower() in IMAGE_EXTENSIONS]
    if not image_paths:
        print(f"Error: No images found in {IMG_DIR}", file=sys.stderr)
        sys.exit(1)
        
    print(f"Found {len(image_paths)} images to classify. This may take a while...")

    # --- 3. Loop through, process, and classify each image ---
    image_categories = {}
    for i, image_path in enumerate(image_paths):
        try:
            # MobileNetV2 expects images to be 224x224 pixels.
            img = image.load_img(image_path, target_size=(224, 224))
            
            # Convert the image to a format the model understands (a numpy array)
            img_array = image.img_to_array(img)
            
            # Create a "batch" of 1 image
            img_batch = np.expand_dims(img_array, axis=0)
            
            # Pre-process the image for the MobileNetV2 model
            img_preprocessed = preprocess_input(img_batch)
            
            # Make the prediction
            predictions = model.predict(img_preprocessed, verbose=0)
            
            # Decode the prediction into a human-readable label
            # We take the top prediction ([0]) and its common name ([1])
            decoded = decode_predictions(predictions, top=1)[0]
            label = decoded[0][1]
            
            # Store the result
            image_categories[str(image_path)] = label
            
            # Print progress
            print(f"  ({i+1}/{len(image_paths)}) {image_path.name} -> Classified as: {label}")

        except Exception as e:
            print(f"Error processing {image_path.name}: {e}", file=sys.stderr)

    # --- 4. Save the results to a JSON file ---
    print(f"\nClassification complete. Saving results to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(image_categories, f, indent=4)
        
    print("Done.")

if __name__ == '__main__':
    classify_images()