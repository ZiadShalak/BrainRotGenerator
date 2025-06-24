# C:/AIVisionTest/debug_loader.py
import cv2 # type: ignore
import librosa # type: ignore
from pathlib import Path

# --- IMPORTANT: Update these two lines to match the files you copied ---
#                  Replace 'your_problem_image.jpg' with the actual filename.
TEST_IMAGE_PATH = Path("./images/meme_063.jpg") 
TEST_SOUND_PATH = Path("./sounds/486_frio_ByvnCVE.mp3")
# --------------------------------------------------------------------

print(f"--- Testing Image: {TEST_IMAGE_PATH} ---")
if not TEST_IMAGE_PATH.exists():
    print("❌ FAILED: Image file does not exist at this path.")
else:
    try:
        img = cv2.imread(str(TEST_IMAGE_PATH))
        if img is None:
            raise ValueError("OpenCV returned None. File is likely corrupt or in an unsupported format.")
        print(f"✅ SUCCESS: Image loaded successfully. Shape: {img.shape}")
    except Exception as e:
        print(f"❌ FAILED to load image. Error: {e}")

print(f"\n--- Testing Sound: {TEST_SOUND_PATH} ---")
if not TEST_SOUND_PATH.exists():
    print("❌ FAILED: Sound file does not exist at this path.")
else:
    try:
        waveform, sr = librosa.load(TEST_SOUND_PATH, sr=None)
        print(f"✅ SUCCESS: Sound loaded successfully. Duration: {len(waveform)/sr:.2f}s, Sample Rate: {sr}")
    except Exception as e:
        print(f"❌ FAILED to load sound. Error: {e}")