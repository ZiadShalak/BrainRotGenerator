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
LOG_DIR = Path("./logs")
LOG_FILE = LOG_DIR / "feedback.csv"
WIN_THRESHOLD = 0.5 # The baseline smile intensity to be considered a "good" reaction
DEFAULT_STEP_DURATION = 2.0 # Seconds to display an image if there's no sound
COOLDOWN_DURATION = 2.0 # Seconds to wait between automated trials
EXPLICIT_REWARD_BONUS = 0.75 # A large bonus/penalty for explicit feedback
MIN_MONTAGE_DURATION = 2.0 # Enforce a minimum display time for each trial
SKIP_PENALTY = -0.25       # The negative reward applied when a trial is skipped
GRADING_DURATION = 3.0 # Seconds to wait for user feedback after a trial


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


# --- 1a. Load and Process Categorized Content ---
IMG_CATS_FILE = Path("./image_categories.json")
SND_CATS_FILE = Path("./sound_categories.json")

try:
    print("Loading categorized content data...")
    with open(IMG_CATS_FILE, 'r') as f:
        image_data = json.load(f)
    with open(SND_CATS_FILE, 'r') as f:
        sound_data = json.load(f)

    # Create an "inverted index" to easily find all files in a category
    image_cats_to_files = defaultdict(list)
    for path, category in image_data.items():
        image_cats_to_files[category].append(path)

    sound_cats_to_files = defaultdict(list)
    for path, category in sound_data.items():
        sound_cats_to_files[category].append(path)
        
    # Get the lists of unique categories, which will be our new "actions"
    image_actions = list(image_cats_to_files.keys())
    sound_actions = list(sound_cats_to_files.keys())

    print(f"Loaded {len(image_actions)} image categories and {len(sound_actions)} sound categories.")
    if not image_actions or not sound_actions:
        raise RuntimeError("One of the category files is empty or invalid.")

except Exception as e:
    print(f"FATAL ERROR: Could not load or process category JSON files: {e}")
    # We can't continue without this data, so we exit.
    exit()


# --- 2. Reinforcement Learning Agent Class ---
# --- [REVISED] 2. Reinforcement Learning Agent Class ---
class RLAgent:
    """ 
    Manages the learning process for decoupled actions using the UCB1 algorithm
    to intelligently balance exploration and exploitation.
    """
    def __init__(self, image_actions, sound_actions, c_param=2.0):
        self.image_actions = image_actions
        self.sound_actions = sound_actions
        self.c_param = c_param  # Exploration parameter. Higher c = more exploration.

        self.total_trials = 0
        
        # Decoupled Q-tables for image and sound categories
        self.image_category_rewards = defaultdict(float)
        self.image_category_counts = defaultdict(int)
        self.sound_category_rewards = defaultdict(float)
        self.sound_category_counts = defaultdict(int)

    def select_action(self):
        """ 
        Selects an image and sound category using the UCB1 algorithm.
        """
        # --- Select Image Category ---
        # First, try any untried image categories to initialize them
        untried_images = [cat for cat in self.image_actions if self.image_category_counts[cat] == 0]
        if untried_images:
            selected_image_cat = random.choice(untried_images)
        else:
            # If all have been tried, calculate UCB scores for all image categories
            ucb_scores = {}
            for cat in self.image_actions:
                avg_reward = self.image_category_rewards[cat]
                exploration_bonus = self.c_param * np.sqrt(np.log(self.total_trials) / self.image_category_counts[cat])
                ucb_scores[cat] = avg_reward + exploration_bonus
            selected_image_cat = max(ucb_scores, key=ucb_scores.get)

        # --- Select Sound Category (using the same logic) ---
        untried_sounds = [cat for cat in self.sound_actions if self.sound_category_counts[cat] == 0]
        if untried_sounds:
            selected_sound_cat = random.choice(untried_sounds)
        else:
            ucb_scores = {}
            for cat in self.sound_actions:
                avg_reward = self.sound_category_rewards[cat]
                exploration_bonus = self.c_param * np.sqrt(np.log(self.total_trials) / self.sound_category_counts[cat])
                ucb_scores[cat] = avg_reward + exploration_bonus
            selected_sound_cat = max(ucb_scores, key=ucb_scores.get)
            
        return selected_image_cat, selected_sound_cat

    def update_q(self, image_category, sound_category, reward):
        """ 
        Updates the Q-values (average rewards) for both the image and sound category
        and increments the total trial count.
        """
        # This trial is now officially complete
        self.total_trials += 1
        
        # Update stats for the image category
        self.image_category_counts[image_category] += 1
        n_img = self.image_category_counts[image_category]
        Q_img = self.image_category_rewards[image_category]
        # Note: We are now storing the SUM of rewards, not the average, to simplify UCB calculation
        self.image_category_rewards[image_category] += reward

        # Update stats for the sound category
        self.sound_category_counts[sound_category] += 1
        n_snd = self.sound_category_counts[sound_category]
        Q_snd = self.sound_category_rewards[sound_category]
        self.sound_category_rewards[sound_category] += reward


# --- 3. Pre-flight Calibration & Agent Initialization ---
print("Initializing camera for calibration...")
cap = cv2.VideoCapture(0)
if not cap.isOpened(): raise IOError("Cannot open webcam")
neutral_val, smile_val = run_interactive_calibration(cap)
smile_detector = SmileDetector(neutral_frac=neutral_val, smile_frac=smile_val)
print("--- Calibration Complete ---")

# Initialize our new, redesigned agent with the category lists
agent = RLAgent(image_actions=image_actions, sound_actions=sound_actions)

# Setup CSV Logger
LOG_DIR.mkdir(exist_ok=True)
log_file_exists = LOG_FILE.exists()
csv_log_file = open(LOG_FILE, 'a', newline='', encoding='utf-8')
csv_writer = csv.writer(csv_log_file)


# --- 4. Main Application Setup (Audio) ---
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
# State variables for the automated loop and our new category-based system
app_mode = 'READY'  # Can be 'READY', 'PLAYING', 'COOLDOWN'
smiles_this_montage = []
explicit_feedback_this_montage = 0.0 # Will be + or - the bonus
feedback_indicator_end_time = 0 # To show text on screen temporarily

# Variables to track the chosen categories and files for the current trial
active_image_category = None
active_sound_category = None
active_image_path = None
active_sound_path = None

montage_start_time = 0
current_montage_duration = DEFAULT_STEP_DURATION
cooldown_start_time = 0
grading_start_time = 0
current_meme_img = np.zeros((MEME_H, MEME_W, 3), dtype=np.uint8)

def start_new_montage():
    """Helper function to reset state and start a new category-based montage."""
    global smiles_this_montage, active_image_category, active_sound_category
    global active_image_path, active_sound_path, current_montage_duration
    global montage_start_time, app_mode, current_meme_img

    print("\n--- Agent selecting new categories ---")
    smiles_this_montage = []
    explicit_feedback_this_montage = 0.0
    
    # 1. Agent selects CATEGORIES, not specific files
    active_image_category, active_sound_category = agent.select_action()
    print(f"Agent chose categories: [Image: {active_image_category}] - [Sound: {active_sound_category}]")

    # 2. Randomly sample one file from each chosen category
    try:
        image_path_list = image_cats_to_files[active_image_category]
        active_image_path = random.choice(image_path_list)
        
        sound_path_list = sound_cats_to_files[active_sound_category]
        active_sound_path = random.choice(sound_path_list)
    except (KeyError, IndexError) as e:
        print(f"Error sampling from categories: {e}. Skipping trial.")
        app_mode = 'COOLDOWN' # Skip to cooldown if we can't find files
        cooldown_start_time = time.time()
        return

    # 3. Load the randomly selected image
    try:
        current_meme_img = cv2.imread(active_image_path)
        print(f"🖼️  Displaying: {Path(active_image_path).name}")
    except Exception as e:
        print(f"Error loading image: {e}"); current_meme_img.fill(0)
    
    # 4. Load and play the randomly selected sound
    try:
        data, fs = sf.read(active_sound_path, dtype='float32')
        current_montage_duration = len(data) / fs
        if data.ndim == 1: data = np.column_stack((data, data))
        with audio_lock:
            app_state['audio_data'] = data
            app_state['audio_position'] = 0
            app_state['stream_active'] = True
        print(f"🔊 Playing: {Path(active_sound_path).name}")
    except Exception as e:
        print(f"Error playing sound: {e}")
        current_montage_duration = MIN_MONTAGE_DURATION # Use our new constant

    montage_start_time = time.time()
    app_mode = 'PLAYING'

# Update CSV logging to handle new category format
if not log_file_exists:
    csv_writer.writerow([
        "timestamp", "image_category", "sound_category", 
        "image_path", "sound_path", "outcome", "shaped_reward", 
        "avg_smile", "peak_smile"
    ])

print("--- Starting Application ---")
print("Press SPACE to begin the automated loop. Press 'q' to quit.")

# --- REVISED Main Application and Drawing Loop ---
while True:
    ret, frame = cap.read()
    if not ret: break
    smile_intensity = smile_detector.get_intensity(frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): break

    # --- State Machine Logic ---
    if app_mode == 'READY':
        if key == ord(' '):
            start_new_montage()

    elif app_mode == 'PLAYING':
        smiles_this_montage.append(smile_intensity)
        
        # Check for explicit user feedback during the montage
        if key == ord('f'):
            explicit_feedback_this_montage = EXPLICIT_REWARD_BONUS
            feedback_indicator_end_time = time.time() + 1.5
            print("--- [FEEDBACK] GOOD reaction registered! ---")
        elif key == ord('b'):
            explicit_feedback_this_montage = -EXPLICIT_REWARD_BONUS
            feedback_indicator_end_time = time.time() + 1.5
            print("--- [FEEDBACK] BAD reaction registered! ---")
        elif key == ord('s'):
            # Skip functionality
            print("--- [ACTION] Trial Skipped by User ---")
            with audio_lock: app_state['stream_active'] = False
            agent.update_q(active_image_category, active_sound_category, SKIP_PENALTY)
            csv_writer.writerow([
                time.time(), active_image_category, active_sound_category,
                active_image_path, active_sound_path, "skipped", f"{SKIP_PENALTY:.4f}", 0.0, 0.0
            ])
            csv_log_file.flush()
            print(f"  Final Reward: {SKIP_PENALTY:.3f} (Skip Penalty)"); print("---")
            app_mode = 'COOLDOWN'
            cooldown_start_time = time.time()

        # Check if the montage duration is over
        if app_mode == 'PLAYING' and time.time() - montage_start_time > current_montage_duration:
            # Transition to GRADING state
            print("--- Montage Finished: Entering Grading Period ---")
            with audio_lock: app_state['stream_active'] = False
            app_mode = 'GRADING'
            grading_start_time = time.time()

    elif app_mode == 'GRADING':
        # During the grading period, still listen for 'f' or 'b'
        if key == ord('f'):
            explicit_feedback_this_montage = EXPLICIT_REWARD_BONUS
            feedback_indicator_end_time = time.time() + 1.5
            print("--- [FEEDBACK] GOOD reaction registered! ---")
        elif key == ord('b'):
            explicit_feedback_this_montage = -EXPLICIT_REWARD_BONUS
            feedback_indicator_end_time = time.time() + 1.5
            print("--- [FEEDBACK] BAD reaction registered! ---")

        # Check if the grading period is over
        if time.time() - grading_start_time > GRADING_DURATION:
            # End of Grading: Calculate Reward & Start Cooldown
            print("--- Grading Period Finished: Calculating Reward ---")
            avg_smile = np.mean(smiles_this_montage) if smiles_this_montage else 0.0
            peak_smile = np.max(smiles_this_montage) if smiles_this_montage else 0.0
            raw_reward = 0.7 * avg_smile + 0.3 * peak_smile
            shaped_reward = raw_reward - WIN_THRESHOLD
            
            final_reward = shaped_reward + explicit_feedback_this_montage
            print(f"  Implicit Reward: {shaped_reward:.3f}, Explicit Bonus: {explicit_feedback_this_montage:.2f}")

            agent.update_q(active_image_category, active_sound_category, final_reward)
            
            csv_writer.writerow([
                time.time(), active_image_category, active_sound_category,
                active_image_path, active_sound_path, "completed", f"{final_reward:.4f}", 
                f"{avg_smile:.4f}", f"{peak_smile:.4f}"
            ])
            csv_log_file.flush()
            
            print(f"  Avg Smile:    {avg_smile:.3f}"); print(f"  Peak Smile:   {peak_smile:.3f}")
            print(f"  Final Reward: {final_reward:.3f} (Raw - Threshold + Bonus)")
            print(f"  Total Trials: {agent.total_trials}"); print("---")

            app_mode = 'COOLDOWN'
            cooldown_start_time = time.time()

    elif app_mode == 'COOLDOWN':
        if time.time() - cooldown_start_time > COOLDOWN_DURATION:
            start_new_montage()

    # --- Drawing Logic ---
    main_canvas = np.full((WINDOW_H, WINDOW_W, 3), 40, dtype=np.uint8)
    cam_display = cv2.resize(frame, (CAM_W, CAM_H))
    
    # Keep meme visible during PLAYING and GRADING states
    if app_mode == 'PLAYING' or app_mode == 'GRADING':
        if current_meme_img is not None and current_meme_img.shape[0] > 0:
             meme_display = cv2.resize(current_meme_img, (MEME_W, MEME_H))
        else:
             meme_display = np.zeros((MEME_H, MEME_W, 3), dtype=np.uint8)
    else:
        meme_display = np.zeros((MEME_H, MEME_W, 3), dtype=np.uint8)

    # Update status text based on the current mode
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
    
    # Show feedback indicator text
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