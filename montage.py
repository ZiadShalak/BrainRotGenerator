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
IMG_DIR = Path("./images")
SND_DIR = Path("./sounds")
MONTAGE_FILE = Path("./montages.json")
LOG_DIR = Path("./logs")
LOG_FILE = LOG_DIR / "feedback.csv"
WIN_THRESHOLD = 0.5 # The baseline smile intensity to be considered a "good" reaction
DEFAULT_STEP_DURATION = 2.0 # Seconds to display an image if there's no sound
COOLDOWN_DURATION = 3.0 # Seconds to wait between automated trials

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

# --- 2. Reinforcement Learning Agent Class ---
# [This class is unchanged from the previous version]
class RLAgent:
    """ Manages the learning process using an epsilon-greedy strategy. """
    def __init__(self, actions, epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.995, chaos_prob=0.02):
        self.actions = actions
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.chaos_prob = chaos_prob
        self.action_rewards = defaultdict(float)
        self.action_counts = defaultdict(int)

    def load_history(self, log_file):
        """ Loads historical performance from the log file to bootstrap the agent. """
        if not log_file.exists(): return
        print(f"Loading learning history from {log_file}...")
        with open(log_file, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    r = float(row['shaped_reward'])
                    action_key = (row['images'], row['sounds'])
                    self.action_counts[action_key] += 1
                    n = self.action_counts[action_key]
                    Q = self.action_rewards[action_key]
                    self.action_rewards[action_key] = Q + (r - Q) / n
                except (ValueError, KeyError, IndexError):
                    continue
        print(f"History loaded. {len(self.action_rewards)} unique actions seen.")

    def select_action(self):
        """ Selects an action (a montage) using an epsilon-greedy strategy. """
        action_tuples = [ (json.dumps(a['images']), json.dumps(a['sounds'])) for a in self.actions]

        if random.random() < self.chaos_prob:
            action_index = random.randrange(len(self.actions))
            return self.actions[action_index]

        if random.random() < self.epsilon:
            action_index = random.randrange(len(self.actions))
            return self.actions[action_index]
        else:
            best_action_tuple = max(self.action_rewards, key=self.action_rewards.get, default=None)
            if best_action_tuple is None:
                return random.choice(self.actions)

            for action in self.actions:
                if (json.dumps(action['images']), json.dumps(action['sounds'])) == best_action_tuple:
                    return action
            return random.choice(self.actions)

    def update_q(self, action, reward):
        """ Updates the Q-value for a given action and decays epsilon. """
        action_key = (json.dumps(action['images']), json.dumps(action['sounds']))
        self.action_counts[action_key] += 1
        n = self.action_counts[action_key]
        Q = self.action_rewards[action_key]
        self.action_rewards[action_key] = Q + (reward - Q) / n
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

# --- 3. Pre-flight Calibration & Asset Loading ---
# [This section is unchanged]
print("Initializing camera for calibration...")
cap = cv2.VideoCapture(0)
if not cap.isOpened(): raise IOError("Cannot open webcam")
neutral_val, smile_val = run_interactive_calibration(cap)
smile_detector = SmileDetector(neutral_frac=neutral_val, smile_frac=smile_val)
print("--- Calibration Complete ---")

try:
    print(f"Loading montage playlist from {MONTAGE_FILE}...")
    with open(MONTAGE_FILE, 'r') as f:
        montage_playlists = json.load(f)
    if not montage_playlists: raise RuntimeError("Montage file is empty.")
    print(f"Successfully loaded {len(montage_playlists)} montages.")
except Exception as e:
    print(f"Error loading assets: {e}"); cap.release(); exit()

agent = RLAgent(actions=montage_playlists)
agent.load_history(LOG_FILE)

LOG_DIR.mkdir(exist_ok=True)
log_file_exists = LOG_FILE.exists()
csv_log_file = open(LOG_FILE, 'a', newline='', encoding='utf-8')
csv_writer = csv.writer(csv_log_file)
if not log_file_exists:
    csv_writer.writerow(["timestamp", "images", "sounds", "shaped_reward", "avg_smile", "peak_smile"])

# --- 4. Main Application Setup ---
# [Audio setup is unchanged]
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
# --- NEW: State variables for the automated loop ---
app_mode = 'READY' # Can be 'READY', 'PLAYING', 'COOLDOWN'
smiles_this_montage = []
active_montage_action = None
montage_step_index = 0
montage_step_start_time = 0
current_step_duration = 0
cooldown_start_time = 0
current_meme_img = np.zeros((MEME_H, MEME_W, 3), dtype=np.uint8)

def start_new_montage():
    """Helper function to reset state and start a new montage."""
    global smiles_this_montage, montage_step_index, active_montage_action
    global current_step_duration, montage_step_start_time, app_mode

    print("\n--- Agent selecting new montage ---")
    smiles_this_montage = []
    montage_step_index = 0
    active_montage_action = agent.select_action()
    
    # Load content for the FIRST step
    image_sequence = active_montage_action["images"]
    sound_sequence = active_montage_action["sounds"]

    if image_sequence:
        try:
            current_meme_img = cv2.imread(image_sequence[0])
            print(f"🖼️ Displaying: {Path(image_sequence[0]).name}")
        except Exception as e:
            print(f"Error loading image: {e}"); current_meme_img.fill(0)
    
    if sound_sequence:
        try:
            data, fs = sf.read(sound_sequence[0], dtype='float32')
            current_step_duration = len(data) / fs
            if data.ndim == 1: data = np.column_stack((data, data))
            with audio_lock:
                app_state['audio_data'] = data; app_state['audio_position'] = 0
                app_state['stream_active'] = True
            print(f"🔊 Playing: {Path(sound_sequence[0]).name}")
        except Exception as e:
            print(f"Error playing sound: {e}")
            current_step_duration = DEFAULT_STEP_DURATION
    else:
        current_step_duration = DEFAULT_STEP_DURATION

    montage_step_start_time = time.time()
    app_mode = 'PLAYING'

print("--- Starting Application ---")
print("Press SPACE to begin the automated loop. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret: break
    smile_intensity = smile_detector.get_intensity(frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): break

    # --- NEW: State Machine Logic ---
    if app_mode == 'READY':
        if key == ord(' '):
            start_new_montage()

    elif app_mode == 'PLAYING':
        smiles_this_montage.append(smile_intensity)
        # Check if the current step (image/sound) is finished
        if time.time() - montage_step_start_time > current_step_duration:
            montage_step_index += 1 # Move to the next step
            
            image_sequence = active_montage_action["images"]
            sound_sequence = active_montage_action["sounds"]
            
            # Check if there are more steps in the montage
            if montage_step_index < max(len(image_sequence), len(sound_sequence)):
                print(f"  -> Next step ({montage_step_index})...")
                # Load content for the new step
                if montage_step_index < len(image_sequence):
                    try: current_meme_img = cv2.imread(image_sequence[montage_step_index])
                    except Exception as e: print(f"Error loading image: {e}"); current_meme_img.fill(0)
                
                if montage_step_index < len(sound_sequence):
                    try:
                        data, fs = sf.read(sound_sequence[montage_step_index], dtype='float32')
                        current_step_duration = len(data) / fs
                        if data.ndim == 1: data = np.column_stack((data, data))
                        with audio_lock:
                            app_state['audio_data'] = data; app_state['audio_position'] = 0; app_state['stream_active'] = True
                        print(f"    🔊 Playing: {Path(sound_sequence[montage_step_index]).name}")
                    except Exception as e:
                        print(f"Error playing sound: {e}"); current_step_duration = DEFAULT_STEP_DURATION
                else:
                    current_step_duration = DEFAULT_STEP_DURATION
                montage_step_start_time = time.time() # Reset timer for the new step
            else:
                # --- End of Montage: Calculate Reward & Start Cooldown ---
                with audio_lock: app_state['stream_active'] = False
                
                print("--- Montage Finished: Calculating Reward ---")
                avg_smile = np.mean(smiles_this_montage) if smiles_this_montage else 0.0
                peak_smile = np.max(smiles_this_montage) if smiles_this_montage else 0.0
                raw_reward = 0.7 * avg_smile + 0.3 * peak_smile
                shaped_reward = raw_reward - WIN_THRESHOLD
                agent.update_q(active_montage_action, shaped_reward)
                img_str = json.dumps(active_montage_action['images'])
                snd_str = json.dumps(active_montage_action['sounds'])
                csv_writer.writerow([time.time(), img_str, snd_str, f"{shaped_reward:.4f}", f"{avg_smile:.4f}", f"{peak_smile:.4f}"])
                csv_log_file.flush()
                print(f"  Avg Smile:    {avg_smile:.3f}"); print(f"  Peak Smile:   {peak_smile:.3f}")
                print(f"  Final Reward: {shaped_reward:.3f} (Raw - {WIN_THRESHOLD} threshold)")
                print(f"  Epsilon:      {agent.epsilon:.2f}"); print("---")

                app_mode = 'COOLDOWN'
                cooldown_start_time = time.time()
    
    elif app_mode == 'COOLDOWN':
        # Check if cooldown period is over
        if time.time() - cooldown_start_time > COOLDOWN_DURATION:
            start_new_montage()

    # --- Drawing Logic ---
    main_canvas = np.full((WINDOW_H, WINDOW_W, 3), 40, dtype=np.uint8)
    cam_display = cv2.resize(frame, (CAM_W, CAM_H))
    
    if app_mode == 'PLAYING':
        meme_display = cv2.resize(current_meme_img, (MEME_W, MEME_H))
    else:
        meme_display = np.zeros((MEME_H, MEME_W, 3), dtype=np.uint8)
        if app_mode == 'READY':
            cv2.putText(main_canvas, "Press SPACE to begin", (STATUS_X, STATUS_Y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        elif app_mode == 'COOLDOWN':
            time_left = COOLDOWN_DURATION - (time.time() - cooldown_start_time)
            cv2.putText(main_canvas, f"Next trial in {int(time_left)+1}s...", (STATUS_X, STATUS_Y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    main_canvas[0:CAM_H, 0:CAM_W] = cam_display
    main_canvas[0:MEME_H, CAM_W:CAM_W + MEME_W] = meme_display

    # Draw the smile meter
    cv2.rectangle(main_canvas, (METER_X, METER_Y), (METER_X + METER_W, METER_Y + METER_H), (50, 50, 50), 2)
    fill_width = int(METER_W * smile_intensity)
    cv2.rectangle(main_canvas, (METER_X, METER_Y), (METER_X + fill_width, METER_Y + METER_H), (0, 255, 0), -1)
    cv2.putText(main_canvas, "Smile Meter", (METER_X, METER_Y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow(WINDOW_NAME, main_canvas)

# --- 6. Cleanup ---
print("\nCleaning up resources...")
stream.stop(); stream.close()
cap.release(); cv2.destroyAllWindows()
csv_log_file.close()