# classify_sounds.py (Refactored with Librosa)
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from pathlib import Path
import json
import sys
import csv
import librosa # <--- New import

# --- Configuration ---
SND_DIR = Path("./sounds")
OUTPUT_FILE = Path("./sound_categories.json")
SOUND_EXTENSIONS = {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}
TARGET_SAMPLE_RATE = 16000 # YAMNet model expects 16kHz

# This will be loaded once in the main function
yamnet_model = None

def load_class_map():
    class_map_path = yamnet_model.class_map_path().numpy().decode('utf-8')
    with tf.io.gfile.GFile(class_map_path) as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        display_names = [display_name for _, _, display_name in reader]
    return display_names

def load_audio_with_librosa(file_path):
    """
    Loads an audio file using Librosa and preprocesses it for YAMNet.
    """
    # Load audio file, ensuring it's mono and preserving original sample rate
    waveform, sample_rate = librosa.load(file_path, sr=None, mono=True)

    # Resample if necessary
    if sample_rate != TARGET_SAMPLE_RATE:
        waveform = librosa.resample(y=waveform, orig_sr=sample_rate, target_sr=TARGET_SAMPLE_RATE)

    return waveform

def main():
    global yamnet_model

    print("Loading YAMNet model from TensorFlow Hub...")
    try:
        yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')
        class_names = load_class_map()
    except Exception as e:
        print(f"Error loading model: {e}", file=sys.stderr)
        sys.exit(1)
    print("Model and class map loaded successfully.")

    sound_paths = [p for p in SND_DIR.glob('**/*') if p.suffix.lower() in SOUND_EXTENSIONS]
    if not sound_paths:
        print(f"Error: No sound files found in {SND_DIR}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(sound_paths)} sounds to classify. Using GPU, this should be fast...")

    sound_categories = {}
    for i, sound_path in enumerate(sound_paths):
        try:
            # Use our new Librosa-based loading function
            waveform = load_audio_with_librosa(sound_path)

            # Make the prediction (YAMNet can accept a NumPy array directly)
            scores, _, _ = yamnet_model(waveform)

            mean_scores = np.mean(scores, axis=0)
            top_class_index = np.argmax(mean_scores)
            label = class_names[top_class_index]

            sound_categories[str(sound_path)] = label

            if (i + 1) % 25 == 0 or (i + 1) == len(sound_paths):
                print(f"  Processed {i+1}/{len(sound_paths)} files...")

        except Exception as e:
            print(f"Error processing {sound_path.name}: {e}", file=sys.stderr)

    print(f"\nClassification complete. Saving results to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(sound_categories, f, indent=4)

    print("Done.")

if __name__ == '__main__':
    main()