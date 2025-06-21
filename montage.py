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

# We use our stable, refactored emotion_detector module
from emotion_detector import EmotionDetector, run_interactive_calibration

# --- 1. Configuration ---
LOG_DIR = Path("./logs")
LOG_FILE = LOG_DIR / "feedback.csv"
AGENT_MEMORY_FILE = LOG_DIR / "agent_memory.json"

WIN_THRESHOLD = 0.5
DEFAULT_STEP_DURATION = 2.0
COOLDOWN_DURATION = 3.0
MIN_MONTAGE_DURATION = 2.0
EXPLICIT_REWARD_BONUS = 0.75
SKIP_PENALTY = -0.25
GRADING_DURATION = 3.0
LOAD_RETRY_ATTEMPTS = 3

# --- NEW: Terminal Formatting ---
SEPARATOR_MAIN = "=" * 60
SEPARATOR_SUB = "-" * 25

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

# --- 2. Reinforcement Learning Agent Class ---
class RLAgent:
    """ 
    Manages the learning process for decoupled actions using the UCB1 algorithm
    and supports saving/loading its state to persist learning across sessions.
    """
    def __init__(self, image_actions, sound_actions, c_param=2.0):
        self.image_actions = image_actions
        self.sound_actions = sound_actions
        self.c_param = c_param
        self.total_trials = 0
        self.image_category_rewards = defaultdict(float)
        self.image_category_counts = defaultdict(int)
        self.sound_category_rewards = defaultdict(float)
        self.sound_category_counts = defaultdict(int)

    def select_action(self):
        # --- Select Image Cluster ---
        untried_images = [cat for cat in self.image_actions if self.image_category_counts[cat] == 0]
        if untried_images:
            selected_image_cat = random.choice(untried_images)
        else:
            ucb_scores = {}
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
        self.total_trials += 1
        self.image_category_counts[image_category] += 1
        self.image_category_rewards[image_category] += reward
        self.sound_category_counts[sound_category] += 1
        self.sound_category_rewards[sound_category] += reward

    def save_state(self, file_path):
        print(f"--- Saving agent's memory to {file_path} ---")
        state = {
            'total_trials': self.total_trials,
            'image_category_rewards': dict(self.image_category_rewards),
            'image_category_counts': dict(self.image_category_counts),
            'sound_category_rewards': dict(self.sound_category_rewards),
            'sound_category_counts': dict(self.sound_category_counts),
        }
        try:
            with open(file_path, 'w') as f:
                json.dump(state, f, indent=4)
            print("Agent memory saved successfully.")
        except Exception as e:
            print(f"Error saving agent state: {e}")

    def load_state(self, file_path):
        if not file_path.exists():
            print("No saved agent memory found. Starting with a blank slate.")
            return
        print(f"--- Loading agent's memory from {file_path} ---")
        try:
            with open(file_path, 'r') as f:
                state = json.load(f)
            self.total_trials = state.get('total_trials', 0)
            self.image_category_rewards = defaultdict(float, state.get('image_category_rewards', {}))
            self.image_category_counts = defaultdict(int, state.get('image_category_counts', {}))
            self.sound_category_rewards = defaultdict(float, state.get('sound_category_rewards', {}))
            self.sound_category_counts = defaultdict(int, state.get('sound_category_counts', {}))
            print(f"Agent memory loaded. Resuming from {self.total_trials} previous trials.")
        except Exception as e:
            print(f"Error loading agent state: {e}. Starting fresh.")

# --- 3. Pre-flight Calibration & Agent Initialization ---
print(SEPARATOR_MAIN)
print("INITIALIZING APPLICATION")
print(SEPARATOR_MAIN)
print("Initializing camera for calibration...")
cap = cv2.VideoCapture(0)
if not cap.isOpened(): raise IOError("Cannot open webcam")
neutral_val, smile_val = run_interactive_calibration(cap)
emotion_detector = EmotionDetector(neutral_score=neutral_val, smile_score=smile_val)
print("--- Calibration Complete ---")
agent = RLAgent(image_actions=image_cluster_ids, sound_actions=sound_cluster_ids)
agent.load_state(AGENT_MEMORY_FILE)
LOG_DIR.mkdir(exist_ok=True)
log_file_exists = LOG_FILE.exists()
csv_log_file = open(LOG_FILE, 'a', newline='', encoding='utf-8')
csv_writer = csv.writer(csv_log_file)
if not log_file_exists:
    csv_writer.writerow([
        "timestamp", "image_cluster_id", "sound_cluster_id", 
        "image_path", "sound_path", "outcome", "final_reward", 
        "avg_smile", "peak_smile", "explicit_rating"
    ])


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
app_mode = 'READY'
smiles_this_montage = []
explicit_rating_this_montage = None
feedback_indicator_end_time = 0
active_image_cluster_id, active_sound_cluster_id = None, None
active_image_path, active_sound_path = None, None
montage_start_time, current_montage_duration = 0, DEFAULT_STEP_DURATION
cooldown_start_time, grading_start_time = 0, 0
current_meme_img = np.zeros((MEME_H, MEME_W, 3), dtype=np.uint8)

# --- NEW: Session Tracking Variables ---
session_start_time = time.time()
session_graded_montages = 0

def start_new_montage():
    """Helper function to reset state and start a new cluster-based montage."""
    global smiles_this_montage, active_image_cluster_id, active_sound_cluster_id
    global active_image_path, active_sound_path, current_montage_duration
    global montage_start_time, app_mode, current_meme_img, explicit_rating_this_montage

    print(f"\n{SEPARATOR_MAIN}")
    smiles_this_montage = []
    explicit_rating_this_montage = None
    
    active_image_cluster_id, active_sound_cluster_id = agent.select_action()
    print(f"AGENT ACTION (Trial {agent.total_trials + 1})")
    print(f"{SEPARATOR_SUB}")
    print(f"Image Cluster: {active_image_cluster_id}")
    print(f"Sound Cluster: {active_sound_cluster_id}")
    print(SEPARATOR_SUB)

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

    image_loaded = False
    for attempt in range(LOAD_RETRY_ATTEMPTS):
        try:
            current_meme_img = cv2.imread(str(active_image_path))
            if current_meme_img is None: raise IOError("OpenCV could not decode image.")
            print(f"-> Displaying: {Path(active_image_path).name}")
            image_loaded = True
            break
        except Exception as e:
            print(f"Attempt {attempt + 1} failed for image '{active_image_path}': {e}")
            if attempt < LOAD_RETRY_ATTEMPTS - 1: time.sleep(0.1)
    if not image_loaded: current_meme_img = np.zeros((MEME_H, MEME_W, 3), dtype=np.uint8)

    sound_loaded = False
    for attempt in range(LOAD_RETRY_ATTEMPTS):
        try:
            data, fs = sf.read(active_sound_path, dtype='float32')
            current_montage_duration = len(data) / fs
            if data.ndim == 1: data = np.column_stack((data, data))
            with audio_lock:
                app_state['audio_data'] = data; app_state['audio_position'] = 0; app_state['stream_active'] = True
            print(f"-> Playing:    {Path(active_sound_path).name}")
            sound_loaded = True
            break
        except Exception as e:
            print(f"Attempt {attempt + 1} failed for sound '{active_sound_path}': {e}")
            if attempt < LOAD_RETRY_ATTEMPTS - 1: time.sleep(0.1)
    if not sound_loaded: current_montage_duration = MIN_MONTAGE_DURATION

    current_montage_duration = max(current_montage_duration, MIN_MONTAGE_DURATION)
    montage_start_time = time.time()
    app_mode = 'PLAYING'

print(SEPARATOR_MAIN)
print("APPLICATION READY")
print("Press SPACE to begin the automated loop. Press 'q' to quit.")
print(SEPARATOR_MAIN)

while True:
    ret, frame = cap.read()
    if not ret: break
    smile_intensity, face_detected = emotion_detector.get_happy_score(frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): break

    if app_mode == 'READY':
        if key == ord(' '):
            start_new_montage()

    elif app_mode == 'PLAYING':
        smiles_this_montage.append(smile_intensity)
        
        rating_keys = {'1': 1, '2': 2, '3': 3, '4': 4, '5': 5}
        pressed_key_char = chr(key) if key != 255 else None
        if pressed_key_char in rating_keys:
            explicit_rating_this_montage = rating_keys[pressed_key_char]
            feedback_indicator_end_time = time.time() + 1.5
            print(f"--- [FEEDBACK] Rating ({explicit_rating_this_montage}) registered! ---")
        elif key == ord('s'):
            print(f"\n{SEPARATOR_SUB}\n--- TRIAL SKIPPED ---\n{SEPARATOR_SUB}")
            session_graded_montages += 1
            with audio_lock: app_state['stream_active'] = False
            agent.update_q(active_image_cluster_id, active_sound_cluster_id, SKIP_PENALTY)
            csv_writer.writerow([
                time.time(), active_image_cluster_id, active_sound_cluster_id,
                active_image_path, active_sound_path, "skipped", f"{SKIP_PENALTY:.4f}", 
                0.0, 0.0, -1 # -1 for rating indicates a skip
            ])
            csv_log_file.flush()
            app_mode = 'COOLDOWN'
            cooldown_start_time = time.time()

        if app_mode == 'PLAYING' and time.time() - montage_start_time > current_montage_duration:
            print("--- Montage Finished: Entering Grading Period ---")
            with audio_lock: app_state['stream_active'] = False
            app_mode = 'GRADING'
            grading_start_time = time.time()

    elif app_mode == 'GRADING':
        rating_keys = {'1': 1, '2': 2, '3': 3, '4': 4, '5': 5}
        pressed_key_char = chr(key) if key != 255 else None
        if pressed_key_char in rating_keys:
            explicit_rating_this_montage = rating_keys[pressed_key_char]
            feedback_indicator_end_time = time.time() + 1.5
            print(f"--- [FEEDBACK] Rating ({explicit_rating_this_montage}) registered! ---")

        if time.time() - grading_start_time > GRADING_DURATION:
            print(f"\n{SEPARATOR_SUB}\n--- TRIAL COMPLETE ---")
            session_graded_montages += 1
            
            final_reward = 0.0
            avg_smile = np.mean(smiles_this_montage) if smiles_this_montage else 0.0
            peak_smile = np.max(smiles_this_montage) if smiles_this_montage else 0.0

            if explicit_rating_this_montage is not None:
                reward_map = {1: -0.75, 2: -0.25, 3: 0.0, 4: 0.25, 5: 0.75}
                final_reward = reward_map.get(explicit_rating_this_montage, 0.0)
                print("Feedback Source: Manual Rating")
            else:
                raw_reward = 0.7 * avg_smile + 0.3 * peak_smile
                final_reward = raw_reward - WIN_THRESHOLD
                print("Feedback Source: Smile Detector")
            
            agent.update_q(active_image_cluster_id, active_sound_cluster_id, final_reward)
            
            csv_writer.writerow([
                time.time(), active_image_cluster_id, active_sound_cluster_id,
                active_image_path, active_sound_path, "completed", f"{final_reward:.4f}", 
                f"{avg_smile:.4f}", f"{peak_smile:.4f}", explicit_rating_this_montage or 0
            ])
            csv_log_file.flush()
            
            print(f"Image Cluster  : {active_image_cluster_id}")
            print(f"Sound Cluster  : {active_sound_cluster_id}")
            print(f"Smile Score    : Avg={avg_smile:.3f}, Peak={peak_smile:.3f}")
            print(f"Manual Rating  : {explicit_rating_this_montage if explicit_rating_this_montage is not None else 'N/A'}")
            print(f"Final Reward   : {final_reward:.3f}")
            print(f"Total Trials   : {agent.total_trials}")
            print(f"{SEPARATOR_SUB}")
            
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
        cv2.putText(main_canvas, f"GRADE NOW (1-5)... {int(time_left)+1}s", (STATUS_X, STATUS_Y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    elif app_mode == 'COOLDOWN':
        time_left = COOLDOWN_DURATION - (time.time() - cooldown_start_time)
        cv2.putText(main_canvas, f"Next trial in {int(time_left)+1}s...", (STATUS_X, STATUS_Y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    main_canvas[0:CAM_H, 0:CAM_W] = cam_display
    main_canvas[0:MEME_H, CAM_W:CAM_W + MEME_W] = meme_display
    
    if time.time() < feedback_indicator_end_time:
        cv2.putText(main_canvas, "FEEDBACK REGISTERED", (STATUS_X, STATUS_Y - 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    if not emotion_detector.last_face_region: # Assuming emotion_detector has this attribute
         text = "NO FACE DETECTED"
         (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
         cv2.rectangle(main_canvas, (5, 5), (15 + text_w, 15 + text_h), (0,0,0), -1)
         cv2.putText(main_canvas, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Draw the smile meter
    cv2.rectangle(main_canvas, (METER_X, METER_Y), (METER_X + METER_W, METER_Y + METER_H), (50, 50, 50), 2)
    fill_width = int(METER_W * smile_intensity)
    cv2.rectangle(main_canvas, (METER_X, METER_Y), (METER_X + fill_width, METER_Y + METER_H), (0, 255, 0), -1)
    cv2.putText(main_canvas, "Smile Meter", (METER_X, METER_Y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow(WINDOW_NAME, main_canvas)


# --- 6. Cleanup ---
print("\n" + SEPARATOR_MAIN)
print("SESSION SUMMARY")
print(SEPARATOR_MAIN)
session_duration = time.time() - session_start_time
minutes, seconds = divmod(session_duration, 60)
print(f"Total Session Time: {int(minutes)}m {int(seconds)}s")
print(f"Total Montages Graded: {session_graded_montages}")
print(SEPARATOR_MAIN)

agent.save_state(AGENT_MEMORY_FILE)
stream.stop()
stream.close()
cap.release()
cv2.destroyAllWindows()
csv_log_file.close()