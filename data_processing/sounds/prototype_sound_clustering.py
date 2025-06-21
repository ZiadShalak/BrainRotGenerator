# data_processing/sounds/prototype_sound_clustering.py (CPU Parallelized Version)
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import librosa
from sklearn.cluster import KMeans
from pathlib import Path
import json
import sys
import multiprocessing
import os
from tqdm import tqdm

# --- Configuration ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

SND_DIR = PROJECT_ROOT / "sounds"
OUTPUT_DIR = PROJECT_ROOT / "data_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
OUTPUT_FILE = OUTPUT_DIR / "sound_cluster_data.json"

SOUND_EXTENSIONS = {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}
NUM_CLUSTERS = 30
TARGET_SAMPLE_RATE = 16000

# --- Worker Initialization ---
# These will be global within each worker process
yamnet_model = None

def init_worker():
    """Initializer for each worker process to load the model once."""
    global yamnet_model
    # Set TensorFlow to use CPU only in the worker processes
    tf.config.set_visible_devices([], 'GPU')
    yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')
    print(f"Worker process {os.getpid()} initialized.")

# --- Main Worker Function ---
def process_single_sound(sound_path):
    """Processes one sound file and returns its path and embedding."""
    try:
        waveform, sr = librosa.load(sound_path, sr=TARGET_SAMPLE_RATE, mono=True)
        _, embeddings, _ = yamnet_model(waveform)
        clip_embedding = np.mean(embeddings, axis=0)
        return (str(sound_path), clip_embedding)
    except Exception as e:
        # We print the error but return None so the main process doesn't crash
        print(f"\nError processing {sound_path.name}: {e}", file=sys.stderr)
        return None

def main():
    """Main execution function"""
    sound_paths = sorted([p for p in SND_DIR.glob('**/*') if p.suffix.lower() in SOUND_EXTENSIONS])
    if not sound_paths:
        print(f"Error: No sound files found in {SND_DIR}", file=sys.stderr)
        sys.exit(1)
    
    print(f"Found {len(sound_paths)} sounds to process.")
    num_processes = multiprocessing.cpu_count()
    print(f"Starting embedding generation with {num_processes} parallel CPU processes...")

    with multiprocessing.Pool(processes=num_processes, initializer=init_worker) as pool:
        # Use imap to process files in parallel and tqdm for a progress bar
        results = list(tqdm(pool.imap(process_single_sound, sound_paths), total=len(sound_paths)))

    # Filter out any files that failed
    successful_results = [r for r in results if r is not None]
    if not successful_results:
        print("No sound files could be processed.", file=sys.stderr)
        sys.exit(1)
        
    # Unzip the results into separate lists
    paths, embeddings = zip(*successful_results)
    embeddings = np.array(embeddings)
    
    # --- Clustering Step ---
    print(f"\nStarting K-Means clustering on {len(embeddings)} sound embeddings...")
    kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=42, n_init='auto')
    kmeans.fit(embeddings)
    print("Clustering complete.")
    
    cluster_data = dict(zip(paths, kmeans.labels_.tolist()))
    
    # --- Save Results ---
    print(f"Saving sound cluster results to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(cluster_data, f, indent=4)
    print("Done.")

if __name__ == '__main__':
    # Set start method for compatibility with Windows/macOS
    multiprocessing.set_start_method('spawn')
    main()