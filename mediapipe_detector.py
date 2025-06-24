# mediapipe_detector.py (Corrected for AttributeError)
import cv2 # type: ignore
import mediapipe as mp # type: ignore
import numpy as np # type: ignore
import time
import math

# --- Configuration ---
SMOOTH_ALPHA = 0.1  # Smoothing factor for the final intensity score
SMILE_SENSITIVITY = 1.0 # Increase this to make smiles easier to register

# --- Landmark constants for MediaPipe Face Mesh ---
LEFT_LIP_CORNER = 61
RIGHT_LIP_CORNER = 291
LEFT_EYE_CORNER = 130
RIGHT_EYE_CORNER = 359

def calculate_distance(p1, p2):
    """Helper function to calculate Euclidean distance between two points."""
    return math.sqrt(((p1.x - p2.x) ** 2) + ((p1.y - p2.y) ** 2))

class SmileDetector:
    """
    Detects smile intensity using MediaPipe Face Mesh for high-performance,
    stable, real-time results.
    """
    def __init__(self, neutral_ratio, smile_ratio):
        """Initializes the detector with calibration values."""
        if smile_ratio <= neutral_ratio:
            smile_ratio = neutral_ratio + 0.1 # Fallback
            
        self.neutral_ratio = neutral_ratio
        self.smile_ratio = smile_ratio
        self.smoothed_intensity = 0.0
        self.face_is_currently_detected = False

        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)

    def get_smile_intensity(self, frame):
        """
        Processes a frame and returns a tuple:
        (smoothed_smile_intensity, face_is_detected_flag)
        """
        # --- FIX: Use the correct syntax to set the writeable flag ---
        # For performance, mark the image as not writeable to pass by reference.
        frame.flags.writeable = False
        # Convert the BGR image to RGB.
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame and find face landmarks
        results = self.face_mesh.process(rgb_frame)
        
        # --- FIX: Set the flag back to True ---
        frame.flags.writeable = True # Re-enable writing
        
        face_found = False
        calibrated_intensity = 0.0

        if results.multi_face_landmarks:
            face_found = True
            landmarks = results.multi_face_landmarks[0].landmark

            p_left_lip = landmarks[LEFT_LIP_CORNER]
            p_right_lip = landmarks[RIGHT_LIP_CORNER]
            p_left_eye = landmarks[LEFT_EYE_CORNER]
            p_right_eye = landmarks[RIGHT_EYE_CORNER]

            mouth_width = calculate_distance(p_left_lip, p_right_lip)
            eye_distance = calculate_distance(p_left_eye, p_right_eye)

            if eye_distance > 0:
                current_ratio = mouth_width / eye_distance
                normalized = (current_ratio - self.neutral_ratio) / (self.smile_ratio - self.neutral_ratio)
                calibrated_intensity = max(0.0, min(1.0, normalized))

        if not face_found:
            calibrated_intensity = self.smoothed_intensity * 0.95

        self.smoothed_intensity = (SMOOTH_ALPHA * calibrated_intensity) + ((1 - SMOOTH_ALPHA) * self.smoothed_intensity)
        
        self.face_is_currently_detected = face_found
        
        return self.smoothed_intensity, self.face_is_currently_detected

    def close(self):
        """Clean up the MediaPipe instance."""
        self.face_mesh.close()


def run_interactive_calibration(camera_capture):
    """
    Runs an interactive calibration to find neutral and smile landmark ratios.
    This function is now much faster and smoother.
    """
    WAIT_TIME = 2.0
    SAMPLE_FRAMES = 45
    
    print("--- Starting MediaPipe Smile Detector Calibration ---")
    
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.5)

    def sample_ratios(prompt):
        print(f"\n{prompt}")
        print(f"Get ready... Hold your expression for the next {int(SAMPLE_FRAMES / 15)} seconds.")
        time.sleep(WAIT_TIME)
        
        ratios = []
        print("Sampling...")
        for _ in range(SAMPLE_FRAMES):
            ret, frame = camera_capture.read()
            if not ret: continue
            
            # This logic does not need the writeable flag optimization
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)

            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                mouth_width = calculate_distance(landmarks[LEFT_LIP_CORNER], landmarks[RIGHT_LIP_CORNER])
                eye_distance = calculate_distance(landmarks[LEFT_EYE_CORNER], landmarks[RIGHT_EYE_CORNER])
                if eye_distance > 0:
                    ratios.append(mouth_width / eye_distance)
            
            cv2.imshow("Calibration", frame)
            cv2.waitKey(1) 
            time.sleep(1/30)

        return float(np.median(ratios)) if ratios else 0.0

    neutral = sample_ratios("Step 1: Please hold a completely NEUTRAL face (no smile).")
    print(f"Calibrated NEUTRAL ratio: {neutral:.4f}")

    smile = sample_ratios("Step 2: Please hold your BIGGEST, most genuine smile.")
    smile = smile * SMILE_SENSITIVITY
    print(f"Calibrated SMILE ratio: {smile:.4f}")
    
    cv2.destroyWindow("Calibration")

    if not neutral or not smile or smile <= neutral:
        print("\n--- WARNING: Calibration failed! ---")
        print("Could not get a reliable reading. Using default fallback values.")
        return 0.4, 0.6
        
    print("\n--- Calibration Complete ---")
    return neutral, smile