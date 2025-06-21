# Meme Montage Reinforcement Learning

This project is a sophisticated AI system that learns a user's unique sense of humor. It uses a Reinforcement Learning agent to intelligently experiment with combinations of images and sounds, observes the user's reactions via an advanced emotion detector, and improves its content choices over time.

## Key Features

* **AI-Powered Content Analysis:** Before the main experiment, a data processing pipeline uses pre-trained neural networks (**MobileNet** for images, **YAMNet** for sounds) to analyze the entire content library. It generates numerical "embeddings" (fingerprints) for each file and uses **K-Means clustering** to automatically group visually or acoustically similar content. This gives the agent a high-level, abstract understanding of its content.
* **Advanced Reinforcement Learning:** The core of the application is an RL agent powered by the **Upper Confidence Bound (UCB)** algorithm. This allows the agent to intelligently balance exploiting content it knows is effective with exploring new combinations to avoid user fatigue.
* **Persistent Memory:** The agent's learned knowledge is saved to `agent_memory.json` upon exit and reloaded on startup, allowing for true long-term learning across multiple sessions.
* **Sophisticated User Feedback System:** The system uses a multi-layered feedback approach:
    * **Emotion Detection:** A real-time, calibrated emotion detector, built with the `deepface` library, provides a constant "happiness score".
    * **Explicit Graded Ratings:** The user can provide a nuanced `1-5` rating after each trial, which overrides the emotion score to provide a clean, "ground truth" reward signal.
    * **User Experience Controls:** Includes a "skip" function ('s' key), a post-trial grading period, and a minimum trial duration for a better user experience.
* **Robust & Optimized:** The data processing scripts are optimized for performance, and the main application contains error handling and retry mechanisms to deal with problematic files gracefully.

## Project Structure

* `montage.py`: The main, feature-complete application that runs the interactive experiment.
* `emotion_detector.py`: The reusable module for detecting happiness/smile intensity.
* `data_processing/`: A folder containing all utility scripts for analyzing and processing your content library.
    * `images/`: Scripts for image clustering, visualization, and duplicate finding.
    * `sounds/`: Scripts for sound clustering and visualization.
* `data_outputs/`: The folder where the generated `.json` cluster data is stored.
* `images/` & `sounds/`: The folders where you place your source image and sound files.
* `logs/`: Contains the `feedback.csv` log and the `agent_memory.json` file.
* `requirements.txt`: A list of all required Python packages for the project.
* `run_montage.bat` & `run_analysis.bat`: Convenience scripts for launching the application and analysis tool on Windows.

## Setup and Installation

1.  **Prerequisites:** Ensure you have Python 3.11 installed and added to your system's PATH.
2.  **Clone Repository:** `git clone <your-repository-url>`
3.  **Install Dependencies:** Navigate to the project root and run:
    ```bash
    pip install -r requirements.txt
    ```
4.  **Add Content:** Place your image files in the `/images` folder and your sound files in the `/sounds` folder.
5.  **Run Data Processing:**
    * Navigate to `data_processing/images/` and run `python prototype_image_clustering.py`.
    * Navigate to `data_processing/sounds/` and run `python prototype_sound_clustering.py`.
    * This only needs to be done once, or whenever you add significant new content.

## How to Use

1.  From the main project directory, double-click **`run_montage.bat`**.
2.  The application will start and guide you through a one-time calibration for the emotion detector.
3.  Once the main window appears, press **SPACE** to begin the first trial.
4.  Use the number keys **1-5** during the "GRADING" period to provide feedback.
5.  Press **'s'** at any time during a trial to skip it.
6.  Press **'q'** to quit the application and save the agent's memory.