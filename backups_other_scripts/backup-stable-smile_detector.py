# smile_detector.py (Refactored Version)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Suppress TensorFlow INFO and WARNING messages
import time
import math
import cv2
import numpy as np
import mediapipe as mp
import time
import math
import textwrap

# --- Configuration (Constants) ---
# These are safe to define at the module level as they don't execute code.
MAX_FACES = 1
MIN_DET_CONF = 0.5
MIN_TRK_CONF = 0.5
SMOOTH_ALPHA = 0.2  # Smoothing factor for the smile intensity

# --- Landmark Indices (Constants) ---
LE_IDX = 33;   RE_IDX = 263  # Left/Right Eye outer corners
ML_IDX = 61;   MR_IDX = 291  # Left/Right Mouth corners
MT_IDX = 13;   MB_IDX = 14   # Middle Top/Bottom lips

def _dist(p1, p2):
    """Helper function to calculate euclidean distance between two landmarks."""
    return math.hypot(p1.x - p2.x, p1.y - p2.y)

class SmileDetector:
    """
    A class to detect smile intensity from a video frame.
    It encapsulates the MediaPipe model and the calibration data, and it does
    not manage the camera or GUI itself, making it a reusable component.
    """
    def __init__(self, neutral_frac, smile_frac):
        """
        Initializes the detector with pre-calculated calibration values.

        Args:
            neutral_frac (float): The mouth ratio for a neutral expression.
            smile_frac (float): The mouth ratio for a full smile.
        """
        if smile_frac <= neutral_frac:
            print(f"Warning: smile_frac ({smile_frac:.4f}) <= neutral_frac ({neutral_frac:.4f}). Adjusting smile_frac to a fallback value.")
            smile_frac = neutral_frac * 1.25 + 0.05 # Ensure smile_frac is always greater

        self.neutral_frac = neutral_frac
        self.smile_frac = smile_frac
        self.prev_intensity = 0.0  # Used for EMA smoothing

        # Initialize the MediaPipe Face Mesh model instance
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=MAX_FACES,
            refine_landmarks=True,
            min_detection_confidence=MIN_DET_CONF,
            min_tracking_confidence=MIN_TRK_CONF
        )

    def _get_mouth_fraction(self, landmarks):
        """Calculates the combined mouth-to-eye ratio from landmarks."""
        eye_dist = _dist(landmarks[LE_IDX], landmarks[RE_IDX])
        mouth_width = _dist(landmarks[ML_IDX], landmarks[MR_IDX])
        mouth_height = _dist(landmarks[MT_IDX], landmarks[MB_IDX])

        # Avoid division by zero
        if eye_dist == 0 or mouth_width == 0:
            return 0.0

        # Ratio of mouth width to the distance between eyes (for normalization)
        width_fraction = mouth_width / eye_dist
        # Mouth aspect ratio (how open vs. wide it is)
        mouth_aspect_ratio = mouth_height / mouth_width
        
        # Combine the two metrics. You can experiment with these weights.
        return 0.5 * width_fraction + 0.5 * mouth_aspect_ratio

    def get_intensity(self, frame_bgr):
        """
        Processes a single BGR camera frame and returns the normalized, 
        smoothed smile intensity on a scale of [0, 1].
        """
        # MediaPipe expects RGB images
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame_rgb)

        # If a face is detected, calculate intensity. Otherwise, use the last known value.
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            combined_frac = self._get_mouth_fraction(landmarks)
            
            # Normalize the fraction based on calibration data to get a value from 0.0 to 1.0+
            raw_intensity = (combined_frac - self.neutral_frac) / (self.smile_frac - self.neutral_frac)
            raw_intensity = max(0.0, min(1.0, raw_intensity))  # Clamp to the [0, 1] range

            # Apply Exponential Moving Average (EMA) for smoothing
            self.prev_intensity = SMOOTH_ALPHA * raw_intensity + (1 - SMOOTH_ALPHA) * self.prev_intensity
        
        return self.prev_intensity

# --- Standalone Calibration Function ---
# This function is separate from the class. It is designed to be called ONCE
# by a main application to get the calibration values.

def _overlay_text_for_calibration(frame, text, countdown=None):
    """Helper to draw text on the calibration window."""
    h, w = frame.shape[:2]
    lines = textwrap.wrap(text, width=30)
    block_h = 20 + 25 * (len(lines) + (1 if countdown else 0))
    overlay = frame.copy()
    cv2.rectangle(overlay, (0,0), (w, block_h), (0,0,0), -1)
    frame[:] = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
    y = 20
    for line in lines:
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
        y += 25
    if countdown is not None:
        cv2.putText(frame, f"{int(countdown)}s", (w-60,25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2, cv2.LINE_AA)

def run_interactive_calibration(camera_capture):
    """
    Runs an interactive calibration routine using a provided OpenCV camera object.
    This function WILL create a temporary GUI window for user feedback.
    
    Args:
        camera_capture: An already opened `cv2.VideoCapture` object.

    Returns:
        A tuple of (neutral_frac, smile_frac).
    """
    WAIT_TIME = 2.0
    SAMPLE_FRAMES = 15
    
    # We need a temporary face mesh processor just for calibration
    face_mesh_processor = mp.solutions.face_mesh.FaceMesh(max_num_faces=1)

    def sample_fracs():
        fracs = []
        for _ in range(SAMPLE_FRAMES):
            ret, frame = camera_capture.read()
            if not ret: continue
            
            results = face_mesh_processor.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if results.multi_face_landmarks:
                detector = SmileDetector(0, 1) # Dummy instance to access the calculation method
                fracs.append(detector._get_mouth_fraction(results.multi_face_landmarks[0].landmark))
            
            _overlay_text_for_calibration(frame, "Sampling facial data…")
            cv2.imshow("Calibration", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
            time.sleep(0.05)
        return float(np.mean(fracs)) if fracs else 0.0

    # Neutral face calibration
    start_time = time.time()
    while time.time() - start_time < WAIT_TIME:
        ret, frame = camera_capture.read()
        if not ret: continue
        _overlay_text_for_calibration(frame, "Step 1: Hold a NEUTRAL face", WAIT_TIME - (time.time() - start_time))
        cv2.imshow("Calibration", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): raise KeyboardInterrupt
    neutral = sample_fracs()
    print(f"Calibrated neutral value: {neutral:.4f}")

    # Smile calibration
    start_time = time.time()
    while time.time() - start_time < WAIT_TIME:
        ret, frame = camera_capture.read()
        if not ret: continue
        _overlay_text_for_calibration(frame, "Step 2: Give your BEST smile", WAIT_TIME - (time.time() - start_time))
        cv2.imshow("Calibration", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): raise KeyboardInterrupt
    smile = sample_fracs()
    print(f"Calibrated smile value: {smile:.4f}")

    cv2.destroyWindow("Calibration")
    face_mesh_processor.close() # Clean up the temporary processor
    return neutral, smile