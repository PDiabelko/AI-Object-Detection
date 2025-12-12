# AI Object Detection Project

This project provides a suite of tools for object detection using YOLO (You Only Look Once) models. It includes interactive applications built with Streamlit and Tkinter for real-time inference on images and camera feeds, as well as scripts for training and validating custom models.

## Features

- **Multiple Interfaces**:
  - **Streamlit Web App**: A browser-based interface for easy access and visualization.
  - **Tkinter Desktop App**: A standalone desktop application for local usage.
  - **Crosswalk Safety Demo**: A specialized Streamlit app demonstrating crosswalk safety features with simulated LED indicators.
- **Real-time Detection**: Support for live camera feed detection.
- **Image Processing**: Upload and process static images with adjustable confidence thresholds.
- **Model Training**: Scripts included to download datasets and train your own YOLO models.
- **Visualization**: Bounding boxes, class labels, and confidence scores displayed directly on images.

## Installation

1.  **Clone the repository**:
    ```bash
    git clone <repository-url>
    cd AI-Object-Detection
    ```

2.  **Install dependencies**:
    Ensure you have Python installed. Install the required packages using pip:
    ```bash
    pip install streamlit ultralytics pillow numpy opencv-python pandas tk
    ```

## Usage

### Running the Applications

#### 1. Streamlit Web App
This is the main web interface for object detection.
```bash
streamlit run app_streamlit.py
```

#### 2. Tkinter Desktop App
A desktop GUI for object detection.
```bash
python app.py
```

#### 3. Crosswalk Safety Demo
A specialized version of the Streamlit app focused on crosswalk scenarios.
```bash
streamlit run crosswalk.py
```

### Training & Validation

The `Training_Model_Scripts/` directory contains scripts to help you train your own models.

1.  **Download Dataset**:
    Downloads the COCO128 dataset (or modify for your own).
    ```bash
    python Training_Model_Scripts/dataset.py
    ```

2.  **Train Model**:
    Trains a YOLOv11 model on the dataset.
    ```bash
    python Training_Model_Scripts/training.py
    ```

3.  **Validate Model**:
    Validates the performance of a trained model.
    ```bash
    python Training_Model_Scripts/validate.py
    ```

## Project Structure

- **`app_streamlit.py`**: Main Streamlit application for object detection.
- **`app.py`**: Tkinter-based desktop application.
- **`crosswalk.py`**: Streamlit application with specific features for crosswalk detection (e.g., LED indicators).
- **`Training_Model_Scripts/`**:
  - `dataset.py`: Script to download datasets (uses KaggleHub).
  - `training.py`: Script to configure and run YOLO model training.
  - `validate.py`: Script to validate trained model performance.
- **`runs/`**: Directory where training results (weights, logs) are saved by Ultralytics YOLO.
- **`images/`**: Directory for storing input/output images.
- **Model Files**:
  - `best.pt`: The primary trained model used by the applications.
  - `yolo11n.pt`, `yolo11s.pt`: Pre-trained YOLOv11 base models.
  - `best-S.pt`, `best-S10.pt`: Checkpoints from various training runs.

## Models

The applications are configured to use `best.pt` by default. If you train a new model, you can update the `MODEL_PATH` variable in the respective application files to point to your new weights (e.g., `runs/detect/train/weights/best.pt`).
