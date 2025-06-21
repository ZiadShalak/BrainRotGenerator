# prototype_image_clustering.py
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2# type: ignore
from sklearn.cluster import KMeans# type: ignore
from pathlib import Path
import json
import sys

# --- NEW: Import our shared function ---
from embedding_utils import generate_embeddings_batched

# --- Configuration with ROBUST Absolute Paths ---
# This gets the main project root folder (e.g., 'AI Vision/')
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# All other paths are now built from this reliable root path.
IMG_DIR = PROJECT_ROOT / "images"
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg'}

# Create a dedicated output folder in the root if it doesn't exist
OUTPUT_DIR = PROJECT_ROOT / "data_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
OUTPUT_FILE = OUTPUT_DIR / "image_cluster_data.json"

# --- Script-specific settings ---
NUM_CLUSTERS = 50 
BATCH_SIZE = 64

def cluster_embeddings(image_paths, embeddings):
    # This function remains the same...
    if len(embeddings) == 0:
        print("No embeddings were generated. Cannot perform clustering.", file=sys.stderr)
        return None
    print(f"\nStarting K-Means clustering to find {NUM_CLUSTERS} groups...")
    kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=42, n_init='auto')
    kmeans.fit(embeddings)
    print("Clustering complete.")
    cluster_data = dict(zip(image_paths, kmeans.labels_.tolist()))
    return cluster_data

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
    cluster_results = cluster_embeddings(paths, embeddings)
    
    if cluster_results:
        print(f"Saving cluster results to {OUTPUT_FILE}...")
        with open(OUTPUT_FILE, 'w') as f:
            json.dump(cluster_results, f, indent=4)
        print("Done.")

if __name__ == '__main__':
    main()