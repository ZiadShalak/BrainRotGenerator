# data_processing/sounds/visualize_sounds.py
import json
from pathlib import Path
import shutil
from tqdm import tqdm

# --- Configuration with ROBUST Absolute Paths ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Input file from our new data_outputs folder
CLUSTER_DATA_FILE = PROJECT_ROOT / "data_outputs" / "sound_cluster_data.json"

# Output directory in the project root
VISUALIZATION_DIR = PROJECT_ROOT / "sound_cluster_visualization"

def visualize():
    """
    Reads the sound cluster data and copies each sound file into a folder
    corresponding to its assigned cluster ID.
    """
    # --- 1. Load the cluster data ---
    if not CLUSTER_DATA_FILE.exists():
        print(f"Error: Cluster data file not found at '{CLUSTER_DATA_FILE}'")
        print("Please run prototype_sound_clustering.py first.")
        return

    print(f"Loading cluster data from {CLUSTER_DATA_FILE}...")
    with open(CLUSTER_DATA_FILE, 'r') as f:
        sound_to_cluster_map = json.load(f)

    # --- 2. Setup the output directory ---
    # Delete the old output directory if it exists for a clean run
    if VISUALIZATION_DIR.exists():
        print(f"Removing old visualization directory: {VISUALIZATION_DIR}")
        shutil.rmtree(VISUALIZATION_DIR)
    
    print(f"Creating new visualization directory: {VISUALIZATION_DIR}")
    VISUALIZATION_DIR.mkdir()

    # --- 3. Copy files into their respective cluster folders ---
    print("Copying sounds into cluster folders...")
    
    # Use tqdm for a progress bar
    for sound_path_str, cluster_id in tqdm(sound_to_cluster_map.items(), desc="Organizing Sounds"):
        try:
            sound_path = Path(sound_path_str)
            
            # Create the path for the new cluster-specific folder
            cluster_folder = VISUALIZATION_DIR / f"cluster_{cluster_id}"
            
            # Create the folder if it's the first time we've seen this cluster ID
            cluster_folder.mkdir(exist_ok=True)
            
            # Copy the original sound into its new cluster folder
            shutil.copy(sound_path, cluster_folder)
        except FileNotFoundError:
            print(f"\nWarning: Source sound not found, skipping: {sound_path_str}")
        except Exception as e:
            print(f"\nAn error occurred processing {sound_path_str}: {e}")

    print(f"\nVisualization complete. Sounds have been organized into folders inside '{VISUALIZATION_DIR}'.")

if __name__ == '__main__':
    visualize()