# create_montages.py
import random
import json
from pathlib import Path

# --- Configuration ---
IMG_DIR = Path("./images")
SND_DIR = Path("./sounds")
OUTPUT_FILE = Path("./montages.json")

# How many unique montage "playlists" to generate
NUM_MONTAGES_TO_CREATE = 200

# The structure of each montage
MAX_IMAGES_PER_MONTAGE = 3
MAX_SOUNDS_PER_MONTAGE = 2

def create_montages():
    """
    Scans asset folders and generates a JSON file containing a list of
    pre-defined, repeatable montage actions.
    """
    print("Scanning asset folders...")
    try:
        image_paths = [str(p) for p in IMG_DIR.glob('*.[jp][pn]g')] + \
                      [str(p) for p in IMG_DIR.glob('*.jpeg')]
        sound_paths = [str(p) for p in SND_DIR.iterdir() if p.is_file()]

        if not image_paths or not sound_paths:
            raise RuntimeError("Could not find sufficient images or sounds.")
        
        print(f"Found {len(image_paths)} images and {len(sound_paths)} sounds.")
    except Exception as e:
        print(f"Error: {e}")
        return

    montage_list = []
    # Use a set to ensure we don't create duplicate montages
    generated_montages = set()

    print(f"Generating {NUM_MONTAGES_TO_CREATE} unique montages...")
    while len(montage_list) < NUM_MONTAGES_TO_CREATE:
        num_images = random.randint(1, MAX_IMAGES_PER_MONTAGE)
        num_sounds = random.randint(1, MAX_SOUNDS_PER_MONTAGE)

        # Randomly sample files for this montage.
        # We use min() to handle cases where we have fewer files than the max setting.
        img_sequence = tuple(sorted(random.sample(image_paths, min(num_images, len(image_paths)))))
        snd_sequence = tuple(sorted(random.sample(sound_paths, min(num_sounds, len(sound_paths)))))

        # The combination of image and sound sequences defines the unique action
        montage_tuple = (img_sequence, snd_sequence)

        # If we've already created this exact montage, skip and try again
        if montage_tuple in generated_montages:
            continue
        
        generated_montages.add(montage_tuple)
        
        montage_list.append({
            "images": img_sequence,
            "sounds": snd_sequence
        })

        if len(montage_list) % 20 == 0:
            print(f"  ...created {len(montage_list)} montages.")

    print(f"\nSaving {len(montage_list)} montages to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(montage_list, f, indent=2)
    
    print("Done!")

if __name__ == "__main__":
    create_montages()