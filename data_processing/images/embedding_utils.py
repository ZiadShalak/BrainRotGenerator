# embedding_utils.py
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input# type: ignore
from tensorflow.keras.preprocessing import image # type: ignore
from pathlib import Path
import sys
from tqdm import tqdm

def generate_embeddings_batched(model, image_paths, batch_size):
    """
    Generates embeddings for all images using batched processing for high performance.
    """
    print(f"Generating embeddings for {len(image_paths)} images using a batch size of {batch_size}...")
    
    ordered_paths = []
    all_embeddings = []

    for i in tqdm(range(0, len(image_paths), batch_size), desc="Processing Batches"):
        batch_paths = image_paths[i:i + batch_size]
        batch_input = []
        valid_paths_in_batch = []

        for img_path in batch_paths:
            try:
                img = image.load_img(img_path, target_size=(224, 224))
                img_array = image.img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0)
                img_preprocessed = preprocess_input(img_array)
                batch_input.append(img_preprocessed)
                valid_paths_in_batch.append(str(img_path))
            except Exception as e:
                print(f"\nSkipping {img_path.name} due to error: {e}", file=sys.stderr)
        
        if not batch_input:
            continue

        batch_array = np.vstack(batch_input)
        batch_embeddings = model.predict(batch_array, verbose=0)
        
        all_embeddings.extend(batch_embeddings)
        ordered_paths.extend(valid_paths_in_batch)
    
    return ordered_paths, np.array(all_embeddings)