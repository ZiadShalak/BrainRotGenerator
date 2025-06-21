# validate_dataset.py
import cv2
import librosa
from pathlib import Path
from tqdm import tqdm

# --- Configuration ---
IMG_DIR = Path("./images")
SND_DIR = Path("./sounds")
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
SOUND_EXTENSIONS = {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}

def validate_data():
    """
    Scans all image and sound files to identify any that cannot be loaded
    by their respective libraries (OpenCV and Librosa).
    """
    print("--- Starting Dataset Health Check ---")
    
    # --- 1. Validate Images ---
    print(f"\nScanning images in: {IMG_DIR}...")
    image_paths = [p for p in IMG_DIR.glob('**/*') if p.suffix.lower() in IMAGE_EXTENSIONS]
    bad_images = []
    
    for img_path in tqdm(image_paths, desc="Checking Images"):
        try:
            img = cv2.imread(str(img_path))
            if img is None:
                # If imread returns None, the file is problematic
                bad_images.append(img_path)
        except Exception as e:
            # Catch any other unexpected errors during loading
            print(f"Unexpected error on {img_path}: {e}")
            bad_images.append(img_path)
            
    # --- 2. Validate Sounds ---
    print(f"\nScanning sounds in: {SND_DIR}...")
    sound_paths = [p for p in SND_DIR.glob('**/*') if p.suffix.lower() in SOUND_EXTENSIONS]
    bad_sounds = []

    for sound_path in tqdm(sound_paths, desc="Checking Sounds"):
        try:
            # We don't need the data, just to see if it loads without error
            librosa.load(sound_path, sr=None)
        except Exception:
            # Any exception during loading means the file is problematic
            bad_sounds.append(sound_path)
            
    # --- 3. Print Final Report ---
    print("\n\n--- Validation Report ---")
    
    print(f"\nImages Checked: {len(image_paths)}")
    if bad_images:
        print(f"❌ Found {len(bad_images)} problematic image files:")
        for path in bad_images:
            print(f"  - {path}")
    else:
        print("✅ All image files loaded successfully.")
        
    print(f"\nSounds Checked: {len(sound_paths)}")
    if bad_sounds:
        print(f"❌ Found {len(bad_sounds)} problematic sound files:")
        for path in bad_sounds:
            print(f"  - {path}")
    else:
        print("✅ All sound files loaded successfully.")
        
    print("\n--- End of Report ---")
    print("Recommendation: Please convert or delete the problematic files listed above.")

if __name__ == '__main__':
    validate_data()