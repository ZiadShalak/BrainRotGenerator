import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2# type: ignore
from sklearn.metrics.pairwise import cosine_similarity# type: ignore
from pathlib import Path
import sys
from tqdm import tqdm

# --- NEW: Import our shared function ---
from embedding_utils import generate_embeddings_batched

# --- Configuration ---
# This line gets the full path to the currently running script (e.g., C:\...\data_processing\images\find_duplicates.py)
# It then goes up three levels (to 'images', to 'data_processing', to the main project root)
# to get a reliable path to our main project folder.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# All other paths are now built from this reliable root path.
IMG_DIR = PROJECT_ROOT / "images"
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg'}

# You can also define an output directory in the root
OUTPUT_DIR = PROJECT_ROOT / "data_outputs"
OUTPUT_DIR.mkdir(exist_ok=True) # Ensure the output directory exists
DUPLICATE_FILE_OUTPUT = OUTPUT_DIR / "image_duplicates.json"


# --- Script-specific settings ---
BATCH_SIZE = 64
SIMILARITY_THRESHOLD = 0.98

# The find_and_group_duplicates function remains exactly the same as before...
def find_and_group_duplicates(image_paths, embeddings):
    # ... (no changes needed in this function)
    print(f"\nCalculating cosine similarity for {len(embeddings)} images...")
    similarity_matrix = cosine_similarity(embeddings)
    
    duplicates_found = 0
    processed_indices = set()
    duplicate_groups = []

    for i in tqdm(range(len(similarity_matrix)), desc="Finding Duplicates"):
        if i in processed_indices:
            continue
        
        similar_indices = np.where(similarity_matrix[i] >= SIMILARITY_THRESHOLD)[0]

        if len(similar_indices) > 1:
            duplicate_group = [image_paths[idx] for idx in similar_indices]
            duplicate_groups.append(duplicate_group)
            duplicates_found += (len(similar_indices) - 1)
            processed_indices.update(similar_indices)
            
    print(f"\nFound {duplicates_found} potential duplicate images across {len(duplicate_groups)} groups.")
    return duplicate_groups


def main():
    """Main execution function"""
    print("Loading MobileNetV2 model for feature extraction...")
    try:
        model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')
    except Exception as e:
        print(f"Error loading model: {e}", file=sys.stderr)
        sys.exit(1)
    print("Model loaded successfully.")

    image_paths = sorted([p for p in IMG_DIR.glob('**/*') if p.suffix.lower() in IMAGE_EXTENSIONS])
    if not image_paths:
        print(f"Error: No images found in {IMG_DIR}", file=sys.stderr)
        sys.exit(1)

    # Use the imported, shared function
    paths, embeddings = generate_embeddings_batched(model, image_paths, BATCH_SIZE)
    duplicate_groups = find_and_group_duplicates(paths, embeddings)
    
    if duplicate_groups:
        print("\n--- Duplicate Image Groups ---")
        for i, group in enumerate(duplicate_groups):
            print(f"\nGroup {i+1}:")
            for file_path in group:
                print(f"  - {file_path}")
    else:
        print("\nNo duplicates found with the current similarity threshold.")

if __name__ == '__main__':
    main()