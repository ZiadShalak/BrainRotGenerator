# emotion_detector.py
import cv2
from deepface import DeepFace
import time
import numpy as np

# --- Configuration ---
FACE_DETECTOR_BACKEND = 'opencv'
DETECTION_INTERVAL = 5 
SMOOTH_ALPHA = 0.2

class EmotionDetector:
    """
    Detects "happy" probability using the DeepFace library, with optimizations
    for performance and signal smoothing.
    """
    def __init__(self, neutral_score, smile_score):
        """Initializes the detector with calibration values."""
        if smile_score <= neutral_score:
            smile_score = neutral_score + 0.1 # Fallback
        self.neutral_score = neutral_score
        self.smile_score = smile_score
        
        self.frame_count = 0
        self.last_face_region = None
        self.smoothed_intensity = 0.0
        
        print("Pre-loading emotion recognition model (this may take a moment on first run)...")
        try:
            dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
            _ = DeepFace.analyze(dummy_frame, actions=['emotion'], enforce_detection=False)
            print("Model pre-loaded successfully.")
        except Exception as e:
            print(f"Could not pre-load model: {e}")

    def get_happy_score(self, frame):
        """
        Processes a frame and returns a tuple: 
        (smoothed_happy_score, face_is_detected_flag)
        """
        run_detection = (self.frame_count % DETECTION_INTERVAL == 0)
        calibrated_intensity = 0.0
        emotions = None
        face_found_this_frame = False

        try:
            face_crop = frame # Default to full frame
            use_last_region = not run_detection and self.last_face_region is not None

            if use_last_region:
                x, y, w, h = self.last_face_region['x'], self.last_face_region['y'], self.last_face_region['w'], self.last_face_region['h']
                padding = 20
                face_crop = frame[max(0, y-padding):y+h+padding, max(0, x-padding):x+w+padding]

            if face_crop.size > 0:
                # If we are not running full detection, enforce_detection must be False
                enforce = run_detection and not use_last_region
                analysis = DeepFace.analyze(
                    face_crop, actions=['emotion'], enforce_detection=enforce, detector_backend=FACE_DETECTOR_BACKEND, silent=True
                )
                if isinstance(analysis, list) and len(analysis) > 0:
                    emotions = analysis[0]['emotion']
                    # If we ran a full detection, update the last known region
                    if run_detection:
                        self.last_face_region = analysis[0]['region']
                    face_found_this_frame = True
        
        except Exception:
            # If any error occurs (e.g., no face found on a detection frame), reset tracking
            self.last_face_region = None
            face_found_this_frame = False

        if emotions:
            raw_happy_score = emotions.get('happy', 0.0) / 100.0
            normalized_intensity = (raw_happy_score - self.neutral_score) / (self.smile_score - self.neutral_score)
            calibrated_intensity = max(0.0, min(1.0, normalized_intensity))

        # If no face was found on this frame, decay the score towards zero
        if not face_found_this_frame:
            calibrated_intensity = 0.0

        self.smoothed_intensity = (SMOOTH_ALPHA * calibrated_intensity) + ((1 - SMOOTH_ALPHA) * self.smoothed_intensity)
        self.frame_count += 1
        
        # Return both the score and whether a face is currently being tracked
        face_is_detected = self.last_face_region is not None
        return self.smoothed_intensity, face_is_detected

# Replace the entire function in emotion_detector.py
def run_interactive_calibration(camera_capture):
    """Runs an interactive calibration to find neutral and smile scores for DeepFace."""
    WAIT_TIME = 3.0
    SAMPLE_FRAMES = 30
    
    print("--- Starting Emotion Detector Calibration ---")
    # --- NEW: Correct way to pre-load/warm-up the model ---
    print("Pre-loading emotion recognition model (this may take a moment on first run)...")
    try:
        dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        _ = DeepFace.analyze(dummy_frame, actions=['emotion'], enforce_detection=False)
        print("Model pre-loaded successfully.")
    except Exception as e:
        print(f"Could not pre-load model: {e}")
    # --- End of new code ---

    def sample_scores(prompt):
        print(f"\n{prompt}")
        print(f"Please hold your expression for {int(WAIT_TIME)} seconds...")
        # (The rest of this inner function is unchanged)
        time.sleep(WAIT_TIME)
        scores = []
        print("Sampling...")
        for _ in range(SAMPLE_FRAMES):
            ret, frame = camera_capture.read()
            if not ret: continue
            try:
                analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False, silent=True)
                if isinstance(analysis, list) and len(analysis) > 0:
                    scores.append(analysis[0]['emotion'].get('happy', 0.0) / 100.0)
            except Exception:
                pass
            time.sleep(0.05)
        return float(np.mean(scores)) if scores else 0.0

    neutral = sample_scores("Step 1: Please hold a NEUTRAL face.")
    print(f"Calibrated neutral value: {neutral:.4f}")

    smile = sample_scores("Step 2: Please give your BEST, biggest smile.")
    print(f"Calibrated smile value: {smile:.4f}")
    
    print("--- Calibration Complete ---")
    return neutral, smile