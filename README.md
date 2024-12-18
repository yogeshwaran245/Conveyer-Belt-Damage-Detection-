# Conveyor Belt Damage Detection Using YOLOv8

This project demonstrates the implementation of a damage detection system for conveyor belts using YOLOv8. It uses a custom-trained model for real-time and static image-based detection. The dataset for this project was augmented and labeled using Roboflow.

---

## Features
- **Real-Time Damage Detection**: Uses a webcam feed to detect damages on conveyor belts.
- **Static Image Analysis**: Detect damages in uploaded images.
- **Custom Dataset**: Trained on a dataset specific to conveyor belt damages with augmentation applied.
- **YOLOv8 Model**: Utilized the advanced YOLOv8 for high accuracy and performance.

---

## Installation

### Requirements
Ensure you have the following installed on your system:
- Python 3.7+
- `ultralytics` library for YOLO
- OpenCV
- NumPy
- Roboflow (for dataset preparation)

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/yogeshwaran245/Conveyor-Belt-Damage-Detection.git
   cd conveyor-belt-damage-detection
   ```
2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download the trained YOLOv8 model (`best922.pt`) and place it in the project directory.

---

## Usage

### 1. Real-Time Detection
Run the `cameratest.py` script to initiate real-time damage detection via webcam:
```bash
python cameratest.py
```
Press `q` to exit the live detection.

### 2. Static Image Analysis
Use the `test.py` script for testing the model on static images:
```bash
python test.py
```
Modify the `image_paths` list in `test.py` to include the paths to your test images.

---

## File Structure
```
conveyor-belt-damage-detection/
├── model.ipynb        # Model training using YOLOv8
├── cameratest.py      # Real-time detection script
├── test.py            # Static image analysis script
├── best922.pt         # YOLOv8 trained model file
├── requirements.txt   # List of required libraries

```

---

## Dataset
The dataset was created using Roboflow, where images were labeled and augmented for better model performance. The dataset includes various types of conveyor belt damages such as:

---

## Model
- YOLOv8 was chosen for its state-of-the-art object detection capabilities.
- The model was trained on a custom dataset for 100 epochs with a learning rate of 0.01.
- Validation achieved an mAP (mean Average Precision) of 90%.

---

## Results
### Real-Time Detection
The model effectively detects and highlights damages on a conveyor belt in real-time using a webcam feed.

### Static Image Analysis
Given test images, the model accurately identifies and localizes the damages.

---

## Acknowledgments
- [YOLOv8 by Ultralytics](https://github.com/ultralytics/ultralytics)
- [Roboflow](https://roboflow.com) for dataset preparation and augmentation
