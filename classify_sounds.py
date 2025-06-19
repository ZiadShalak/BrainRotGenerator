# classify_sounds.py
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_io as tfio
import numpy as np
from pathlib import Path
import json
import sys
import csv
import io

# --- Configuration ---
SND_DIR = Path("./sounds")
OUTPUT_FILE = Path("./sound_categories.json")
SOUND_EXTENSIONS = {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}
# YAMNet model expects specific audio properties
TARGET_SAMPLE_RATE = 16000

def load_class_map():
    """
    Loads the YAMNet class map from the model's assets. This maps the model's
    numeric output to human-readable class names.
    """
    # The class map is stored within the model's assets directory on TF Hub
    class_map_path = yamnet_model.class_map_path().numpy().decode('utf-8')
    # The file is a CSV with 'index,mid,display_name'
    with tf.io.gfile.GFile(class_map_path) as csvfile:
        reader = csv.reader(csvfile)
        # Skip the header
        next(reader)
        # We only need the display names
        display_names = [display_name for _, _, display_name in reader]
    return display_names

@tf.function
def load_audio_for_yamnet(file_path):
    """
    Loads an audio file and preprocesses it to the format YAMNet expects:
    - Mono channel
    - 16kHz sample rate
    - Float values between -1.0 and 1.0
    """
    # Read the audio file
    audio_tensor = tfio.audio.AudioIOTensor(file_path, dtype=tf.int16)    # Convert to float and take the first channel if stereo
    audio = tf.cast(audio_tensor.to_tensor(), tf.float32)
    if tf.shape(audio)[-1] > 1:
        audio = tf.reduce_mean(audio, axis=-1)
    else:
        audio = tf.squeeze(audio, axis=[-1])

    # Resample to the target rate
    sample_rate = tf.cast(audio_tensor.rate, tf.int64)
    if sample_rate != TARGET_SAMPLE_RATE:
        audio = tfio.audio.resample(audio, sample_rate, TARGET_SAMPLE_RATE)
        
    return audio

def classify_sounds():
    """
    Scans the sound directory, uses YAMNet to classify each sound,
    and saves the results to a JSON file.
    """
    global yamnet_model # Make model globally accessible for helper function
    
    # --- 1. Load the Pre-trained Model and Class Map ---
    print("Loading YAMNet model from TensorFlow Hub...")
    try:
        yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')
        class_names = load_class_map()
    except Exception as e:
        print(f"Error loading model: {e}", file=sys.stderr)
        print("Please ensure you have an active internet connection the first time you run this.", file=sys.stderr)
        sys.exit(1)
        
    print("Model and class map loaded successfully.")
    
    # --- 2. Find all sounds ---
    sound_paths = [p for p in SND_DIR.glob('**/*') if p.suffix.lower() in SOUND_EXTENSIONS]
    if not sound_paths:
        print(f"Error: No sound files found in {SND_DIR}", file=sys.stderr)
        sys.exit(1)
        
    print(f"Found {len(sound_paths)} sounds to classify. This will take a very long time...")

    # --- 3. Loop through and classify each sound ---
    sound_categories = {}
    for i, sound_path in enumerate(sound_paths):
        try:
            # Load and preprocess the audio
            waveform = load_audio_for_yamnet(str(sound_path))
            
            # Make the prediction
            scores, embeddings, spectrogram = yamnet_model(waveform)
            
            # Get the single best prediction for the entire clip
            # We do this by taking the average score for each class across all frames
            mean_scores = np.mean(scores, axis=0)
            top_class_index = np.argmax(mean_scores)
            label = class_names[top_class_index]
            
            sound_categories[str(sound_path)] = label
            
            print(f"  ({i+1}/{len(sound_paths)}) {sound_path.name} -> Classified as: {label}")
            
        except Exception as e:
            # Using tf.function can sometimes hide the true error, so we print it
            print(f"Error processing {sound_path.name}: {e}", file=sys.stderr)

    # --- 4. Save the results ---
    print(f"\nClassification complete. Saving results to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(sound_categories, f, indent=4)
        
    print("Done.")

if __name__ == '__main__':
    classify_sounds()