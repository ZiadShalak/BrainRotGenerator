#!/usr/bin/env python3
import cv2
import numpy as np
import random
from pathlib import Path
import sounddevice as sd
import soundfile as sf
import time
import json
import threading
import csv
from collections import defaultdict

# We use our stable, refactored smile_detector library
from smile_detector import SmileDetector, run_interactive_calibration

# --- 1. Configuration ---
LOG_DIR = Path("./logs")
LOG_FILE = LOG_DIR / "feedback.csv"
WIN_THRESHOLD = 0.5
DEFAULT_STEP_DURATION = 2.0
COOLDOWN_DURATION = 3.0
MIN_MONTAGE_DURATION = 2.0
EXPLICIT_REWARD_BONUS = 0.75
SKIP_PENALTY = -0.25
GRADING_DURATION = 3.0
LOAD_RETRY_ATTEMPTS = 3 # Number of times to try loading a file before giving up

# --- Layout Configuration ---
CAM_W, CAM_H = 640, 480
MEME_W, MEME_H = 640, 480
WINDOW_W = CAM_W + MEME_W
WINDOW_H = MEME_H + 80
WINDOW_NAME = "Meme Montage RL"

# --- Meter and Text Configuration ---
METER_X, METER_Y = 10, CAM_H + 40
METER_W, METER_H = 200, 25
STATUS_X, STATUS_Y = CAM_W + 20, CAM_H + 45

# --- 1a. Load and Process Clustered Content ---
# <-- UPDATED to load cluster data
IMG_CLUSTER_FILE = Path("./data_outputs/image_cluster_data.json")
SND_CLUSTER_FILE = Path("./data_outputs/sound_cluster_data.json")

try:
    print("Loading clustered content data...")
    with open(IMG_CLUSTER_FILE, 'r') as f:
        image_data = json.load(f)
    with open(SND_CLUSTER_FILE, 'r') as f:
        sound_data = json.load(f)

    image_clusters = defaultdict(list)
    for path, cluster_id in image_data.items():
        image_clusters[f"img_cluster_{cluster_id}"].append(path)

    sound_clusters = defaultdict(list)
    for path, cluster_id in sound_data.items():
        sound_clusters[f"snd_cluster_{cluster_id}"].append(path)
        
    image_cluster_ids = list(image_clusters.keys())
    sound_cluster_ids = list(sound_clusters.keys())

    print(f"Loaded {len(image_cluster_ids)} image clusters and {len(sound_cluster_ids)} sound clusters.")
    if not image_cluster_ids or not sound_cluster_ids:
        raise RuntimeError("One of the cluster data files is empty or invalid.")

except Exception as e:
    print(f"FATAL ERROR: Could not load or process cluster JSON files: {e}")
    exit()

# --- 2. Reinforcement Learning Agent Class (UCB Version) ---
class RLAgent:
    """ 
    Manages the learning process for decoupled actions using the UCB1 algorithm
    to intelligently balance exploration and exploitation.
    """
    def __init__(self, image_actions, sound_actions, c_param=2.0):
        self.image_actions = image_actions
        self.sound_actions = sound_actions
        self.c_param = c_param  # Exploration parameter

        self.total_trials = 0
        
        self.image_category_rewards = defaultdict(float)
        self.image_category_counts = defaultdict(int)
        self.sound_category_rewards = defaultdict(float)
        self.sound_category_counts = defaultdict(int)

    def select_action(self):
        """ Selects an image and sound cluster ID using the UCB1 algorithm. """
        # --- Select Image Cluster ---
        untried_images = [cat for cat in self.image_actions if self.image_category_counts[cat] == 0]
        if untried_images:
            selected_image_cat = random.choice(untried_images)
        else:
            ucb_scores = {}
            # We add 1 to total_trials in the log to avoid log(0) issues on the first real UCB step
            log_total = np.log(self.total_trials + 1)
            for cat in self.image_actions:
                avg_reward = self.image_category_rewards[cat] / self.image_category_counts[cat]
                exploration_bonus = self.c_param * np.sqrt(log_total / self.image_category_counts[cat])
                ucb_scores[cat] = avg_reward + exploration_bonus
            selected_image_cat = max(ucb_scores, key=ucb_scores.get)

        # --- Select Sound Cluster ---
        untried_sounds = [cat for cat in self.sound_actions if self.sound_category_counts[cat] == 0]
        if untried_sounds:
            selected_sound_cat = random.choice(untried_sounds)
        else:
            ucb_scores = {}
            log_total = np.log(self.total_trials + 1)
            for cat in self.sound_actions:
                avg_reward = self.sound_category_rewards[cat] / self.sound_category_counts[cat]
                exploration_bonus = self.c_param * np.sqrt(log_total / self.sound_category_counts[cat])
                ucb_scores[cat] = avg_reward + exploration_bonus
            selected_sound_cat = max(ucb_scores, key=ucb_scores.get)
            
        return selected_image_cat, selected_sound_cat

    def update_q(self, image_category, sound_category, reward):
        """ Updates the Q-values for both the image and sound cluster. """
        self.total_trials += 1
        
        self.image_category_counts[image_category] += 1
        self.image_category_rewards[image_category] += reward

        self.sound_category_counts[sound_category] += 1
        self.sound_category_rewards[sound_category] += reward

# --- 3. Pre-flight Calibration & Agent Initialization ---
print("Initializing camera for calibration...")
cap = cv2.VideoCapture(0)
if not cap.isOpened(): raise IOError("Cannot open webcam")
neutral_val, smile_val = run_interactive_calibration(cap)
smile_detector = SmileDetector(neutral_frac=neutral_val, smile_frac=smile_val)
print("--- Calibration Complete ---")

# Initialize agent with the CLUSTER IDs as actions
agent = RLAgent(image_actions=image_cluster_ids, sound_actions=sound_cluster_ids)

# Setup CSV Logger
LOG_DIR.mkdir(exist_ok=True)
log_file_exists = LOG_FILE.exists()
csv_log_file = open(LOG_FILE, 'a', newline='', encoding='utf-8')
csv_writer = csv.writer(csv_log_file)


# --- 4. Main Application Setup (Audio) ---
# This section is unchanged
app_state = {'volume': 1.0, 'audio_data': None, 'audio_position': 0, 'stream_active': False}
audio_lock = threading.Lock()
def audio_callback(outdata, frames, time, status):
    if status: print(status, flush=True)
    with audio_lock:
        if app_state['stream_active'] and app_state['audio_data'] is not None:
            pos = app_state['audio_position']
            remaining = len(app_state['audio_data']) - pos
            chunk_size = min(remaining, frames)
            chunk = app_state['audio_data'][pos:pos + chunk_size]
            outdata[:chunk_size] = chunk * app_state['volume']
            if chunk_size < frames: outdata[chunk_size:] = 0
            app_state['audio_position'] += chunk_size
        else: outdata.fill(0)
stream = sd.OutputStream(callback=audio_callback, channels=2, dtype='float32'); stream.start()
cv2.namedWindow(WINDOW_NAME)
def volume_callback(val):
    with audio_lock: app_state['volume'] = val / 100.0
cv2.createTrackbar('Volume', WINDOW_NAME, 100, 100, volume_callback)


# --- 5. Main Application Loop ---
# This section contains all our recent UX improvements
app_mode = 'READY'
smiles_this_montage = []
explicit_feedback_this_montage = 0.0
feedback_indicator_end_time = 0

active_image_cluster_id = None # <-- UPDATED
active_sound_cluster_id = None # <-- UPDATED
active_image_path = None
active_sound_path = None

montage_start_time = 0
current_montage_duration = DEFAULT_STEP_DURATION
cooldown_start_time = 0
grading_start_time = 0
current_meme_img = np.zeros((MEME_H, MEME_W, 3), dtype=np.uint8)

def start_new_montage():
    """Helper function to reset state and start a new cluster-based montage."""
    global smiles_this_montage, active_image_cluster_id, active_sound_cluster_id
    global active_image_path, active_sound_path, current_montage_duration
    global montage_start_time, app_mode, current_meme_img, explicit_feedback_this_montage

    print("\n--- Agent selecting new clusters ---")
    smiles_this_montage = []
    explicit_feedback_this_montage = 0.0

    active_image_cluster_id, active_sound_cluster_id = agent.select_action()
    print(f"Agent chose clusters: [Image: {active_image_cluster_id}] - [Sound: {active_sound_cluster_id}]")

    try:
        image_path_list = image_clusters[active_image_cluster_id]
        active_image_path = random.choice(image_path_list)

        sound_path_list = sound_clusters[active_sound_cluster_id]
        active_sound_path = random.choice(sound_path_list)
    except (KeyError, IndexError) as e:
        print(f"Error sampling from clusters: {e}. Skipping trial.")
        app_mode = 'COOLDOWN'
        cooldown_start_time = time.time()
        return

    # --- NEW: Retry loop for loading the image ---
    image_loaded = False
    for attempt in range(LOAD_RETRY_ATTEMPTS):
        try:
            current_meme_img = cv2.imread(str(active_image_path))
            if current_meme_img is None:
                raise IOError("OpenCV could not decode the image file.")
            print(f"🖼️  Displaying: {Path(active_image_path).name}")
            image_loaded = True
            break # Success, exit the retry loop
        except Exception as e:
            print(f"Attempt {attempt + 1} failed to load image '{active_image_path}': {e}")
            if attempt < LOAD_RETRY_ATTEMPTS - 1:
                time.sleep(0.1) # Wait briefly before retrying
    if not image_loaded:
        current_meme_img = np.zeros((MEME_H, MEME_W, 3), dtype=np.uint8)

    # --- NEW: Retry loop for loading the sound ---
    sound_loaded = False
    for attempt in range(LOAD_RETRY_ATTEMPTS):
        try:
            data, fs = sf.read(active_sound_path, dtype='float32')
            current_montage_duration = len(data) / fs
            if data.ndim == 1: data = np.column_stack((data, data))
            with audio_lock:
                app_state['audio_data'] = data
                app_state['audio_position'] = 0
                app_state['stream_active'] = True
            print(f"🔊 Playing: {Path(active_sound_path).name}")
            sound_loaded = True
            break # Success, exit the retry loop
        except Exception as e:
            print(f"Attempt {attempt + 1} failed to play sound '{active_sound_path}': {e}")
            if attempt < LOAD_RETRY_ATTEMPTS - 1:
                time.sleep(0.1)
    if not sound_loaded:
        current_montage_duration = MIN_MONTAGE_DURATION

    current_montage_duration = max(current_montage_duration, MIN_MONTAGE_DURATION)
    montage_start_time = time.time()
    app_mode = 'PLAYING'

# Update CSV logging to handle new cluster format
if not log_file_exists:
    csv_writer.writerow([
        "timestamp", "image_cluster_id", "sound_cluster_id", # <-- UPDATED
        "image_path", "sound_path", "outcome", "shaped_reward", 
        "avg_smile", "peak_smile"
    ])

print("--- Starting Application ---")
print("Press SPACE to begin the automated loop. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret: break
    smile_intensity = smile_detector.get_intensity(frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): break

    # State Machine Logic
    if app_mode == 'READY':
        if key == ord(' '):
            start_new_montage()

    elif app_mode == 'PLAYING':
        smiles_this_montage.append(smile_intensity)
        
        if key == ord('f'):
            explicit_feedback_this_montage = EXPLICIT_REWARD_BONUS
            feedback_indicator_end_time = time.time() + 1.5
            print("--- [FEEDBACK] GOOD reaction registered! ---")
        elif key == ord('b'):
            explicit_feedback_this_montage = -EXPLICIT_REWARD_BONUS
            feedback_indicator_end_time = time.time() + 1.5
            print("--- [FEEDBACK] BAD reaction registered! ---")
        elif key == ord('s'):
            print("--- [ACTION] Trial Skipped by User ---")
            with audio_lock: app_state['stream_active'] = False
            agent.update_q(active_image_cluster_id, active_sound_cluster_id, SKIP_PENALTY)
            csv_writer.writerow([
                time.time(), active_image_cluster_id, active_sound_cluster_id,
                active_image_path, active_sound_path, "skipped", f"{SKIP_PENALTY:.4f}", 0.0, 0.0
            ])
            csv_log_file.flush()
            print(f"  Final Reward: {SKIP_PENALTY:.3f} (Skip Penalty)"); print("---")
            app_mode = 'COOLDOWN'
            cooldown_start_time = time.time()

        if app_mode == 'PLAYING' and time.time() - montage_start_time > current_montage_duration:
            print("--- Montage Finished: Entering Grading Period ---")
            with audio_lock: app_state['stream_active'] = False
            app_mode = 'GRADING'
            grading_start_time = time.time()

    elif app_mode == 'GRADING':
        if key == ord('f'):
            explicit_feedback_this_montage = EXPLICIT_REWARD_BONUS
            feedback_indicator_end_time = time.time() + 1.5
            print("--- [FEEDBACK] GOOD reaction registered! ---")
        elif key == ord('b'):
            explicit_feedback_this_montage = -EXPLICIT_REWARD_BONUS
            feedback_indicator_end_time = time.time() + 1.5
            print("--- [FEEDBACK] BAD reaction registered! ---")

        if time.time() - grading_start_time > GRADING_DURATION:
            print("--- Grading Period Finished: Calculating Reward ---")
            avg_smile = np.mean(smiles_this_montage) if smiles_this_montage else 0.0
            peak_smile = np.max(smiles_this_montage) if smiles_this_montage else 0.0
            raw_reward = 0.7 * avg_smile + 0.3 * peak_smile
            shaped_reward = raw_reward - WIN_THRESHOLD
            
            final_reward = shaped_reward + explicit_feedback_this_montage
            print(f"  Implicit Reward: {shaped_reward:.3f}, Explicit Bonus: {explicit_feedback_this_montage:.2f}")

            agent.update_q(active_image_cluster_id, active_sound_cluster_id, final_reward) # <-- UPDATED
            
            csv_writer.writerow([
                time.time(), active_image_cluster_id, active_sound_cluster_id, # <-- UPDATED
                active_image_path, active_sound_path, "completed", f"{final_reward:.4f}", 
                f"{avg_smile:.4f}", f"{peak_smile:.4f}"
            ])
            csv_log_file.flush()
            
            print(f"  Avg Smile:    {avg_smile:.3f}"); print(f"  Peak Smile:   {peak_smile:.3f}")
            print(f"  Final Reward: {final_reward:.3f} (Raw - Threshold + Bonus)")
            print(f"  Total Trials: {agent.total_trials}"); print("---") # <-- UPDATED

            app_mode = 'COOLDOWN'
            cooldown_start_time = time.time()

    elif app_mode == 'COOLDOWN':
        if time.time() - cooldown_start_time > COOLDOWN_DURATION:
            start_new_montage()

    # --- Drawing Logic ---
    main_canvas = np.full((WINDOW_H, WINDOW_W, 3), 40, dtype=np.uint8)
    cam_display = cv2.resize(frame, (CAM_W, CAM_H))
    
    if app_mode == 'PLAYING' or app_mode == 'GRADING':
        if current_meme_img is not None and current_meme_img.shape[0] > 0:
             meme_display = cv2.resize(current_meme_img, (MEME_W, MEME_H))
        else:
             meme_display = np.zeros((MEME_H, MEME_W, 3), dtype=np.uint8)
    else:
        meme_display = np.zeros((MEME_H, MEME_W, 3), dtype=np.uint8)

    if app_mode == 'READY':
        cv2.putText(main_canvas, "Press SPACE to begin", (STATUS_X, STATUS_Y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    elif app_mode == 'GRADING':
        time_left = GRADING_DURATION - (time.time() - grading_start_time)
        cv2.putText(main_canvas, f"GRADE NOW (F/B)... {int(time_left)+1}s", (STATUS_X, STATUS_Y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    elif app_mode == 'COOLDOWN':
        time_left = COOLDOWN_DURATION - (time.time() - cooldown_start_time)
        cv2.putText(main_canvas, f"Next trial in {int(time_left)+1}s...", (STATUS_X, STATUS_Y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    main_canvas[0:CAM_H, 0:CAM_W] = cam_display
    main_canvas[0:MEME_H, CAM_W:CAM_W + MEME_W] = meme_display
    
    if time.time() < feedback_indicator_end_time:
        cv2.putText(main_canvas, "FEEDBACK REGISTERED", (STATUS_X, STATUS_Y - 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Draw the smile meter
    cv2.rectangle(main_canvas, (METER_X, METER_Y), (METER_X + METER_W, METER_Y + METER_H), (50, 50, 50), 2)
    fill_width = int(METER_W * smile_intensity)
    cv2.rectangle(main_canvas, (METER_X, METER_Y), (METER_X + fill_width, METER_Y + METER_H), (0, 255, 0), -1)
    cv2.putText(main_canvas, "Smile Meter", (METER_X, METER_Y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow(WINDOW_NAME, main_canvas)


# --- 6. Cleanup ---
print("\nCleaning up resources...")
stream.stop()
stream.close()
cap.release()
cv2.destroyAllWindows()
csv_log_file.close()