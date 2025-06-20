# visualize_clusters.py
import json
from pathlib import Path
import shutil
from tqdm import tqdm

# --- Configuration with ROBUST Absolute Paths ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Input file from our new data_outputs folder
CLUSTER_DATA_FILE = PROJECT_ROOT / "data_outputs" / "image_cluster_data.json"

# Output directory in the project root
VISUALIZATION_DIR = PROJECT_ROOT / "image_cluster_visualization"


def visualize():
    """
    Reads the cluster data and copies each image into a folder
    corresponding to its assigned cluster ID.
    """
    # --- 1. Load the cluster data ---
    if not CLUSTER_DATA_FILE.exists():
        print(f"Error: Cluster data file not found at '{CLUSTER_DATA_FILE}'")
        print("Please run prototype_image_clustering.py first.")
        return

    print(f"Loading cluster data from {CLUSTER_DATA_FILE}...")
    with open(CLUSTER_DATA_FILE, 'r') as f:
        image_to_cluster_map = json.load(f)

    # --- 2. Setup the output directory ---
    # Delete the old output directory if it exists for a clean run
    if VISUALIZATION_DIR.exists():
        print(f"Removing old visualization directory: {VISUALIZATION_DIR}")
        shutil.rmtree(VISUALIZATION_DIR)
    
    print(f"Creating new visualization directory: {VISUALIZATION_DIR}")
    VISUALIZATION_DIR.mkdir()

    # --- 3. Copy files into their respective cluster folders ---
    print("Copying images into cluster folders...")
    
    # Use tqdm for a progress bar
    for image_path_str, cluster_id in tqdm(image_to_cluster_map.items(), desc="Organizing Images"):
        try:
            image_path = Path(image_path_str)
            
            # Create the path for the new cluster-specific folder
            cluster_folder = VISUALIZATION_DIR / f"cluster_{cluster_id}"
            
            # Create the folder if it's the first time we've seen this cluster ID
            cluster_folder.mkdir(exist_ok=True)
            
            # Copy the original image into its new cluster folder
            shutil.copy(image_path, cluster_folder)
        except FileNotFoundError:
            print(f"\nWarning: Source image not found, skipping: {image_path_str}")
        except Exception as e:
            print(f"\nAn error occurred processing {image_path_str}: {e}")

    print(f"\nVisualization complete. Images have been organized into folders inside '{VISUALIZATION_DIR}'.")

if __name__ == '__main__':
    visualize()