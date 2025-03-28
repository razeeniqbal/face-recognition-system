# Face Recognition System

A comprehensive web application for face detection and recognition built with Python, OpenCV, and Streamlit, meeting all the requirements for the technical assessment. This system uses lightweight OpenCV-based methods for face detection and alignment without dependencies on dlib, making it more portable and easier to deploy.

## Features

- **Face Detection**: Detect faces in images using either OpenCV's Haar Cascade or DNN-based detectors
- **Face Alignment**: Align faces using eye detection for better recognition accuracy
- **Face Recognition**: Identify people in images using a trained SVM classifier with HOG features
- **User-friendly Interface**: Clean, intuitive web UI with upload and camera capture options
- **Visualizations**: Show bounding boxes and recognized face names in results
- **Detailed Results**: Display recognition results with expandable face details
- **Model Evaluation**: Display accuracy, precision, recall, and F1-score metrics
- **Data Augmentation**: Enhance training data with flips, rotations, and brightness adjustments

## Technology Stack

- **Python**: Core programming language
- **OpenCV**: Image processing, face detection, and face alignment
- **scikit-learn**: Machine learning for face classification (SVM)
- **Streamlit**: Web application framework
- **NumPy & Pillow**: Image manipulation libraries

## Project Structure

```
face-recognition-system/
├── app.py                         # Main Streamlit application
├── utils.py                       # Utility functions for face processing
├── train.py                       # Model training script
├── deploy.py                      # Deployment helper script
├── download_models.py             # Script to download datasets and models
├── download_caffe_model.py        # Script to download Caffe DNN models
├── setup.py                       # Setup script to install dependencies
├── verify_model.py                # Model verification and diagnostics
├── requirements.txt               # Python dependencies
├── README.md                      # Project documentation
├── models/                        # Directory for storing trained models
│   ├── face_classifier.pkl        # Trained classifier (generated after training)
│   ├── res10_300x300_ssd_iter_140000.caffemodel  # OpenCV DNN face detector (downloaded)
│   └── deploy.prototxt            # OpenCV DNN config (downloaded)
└── lfw/                           # LFW dataset directory (downloaded)
```

> **Note:** The `models/` directory and `lfw/` dataset are not included in the GitHub repository due to size constraints. These will be generated/downloaded during the setup process as explained below.

## Installation

### Prerequisites

- Python 3.12
- pip package manager

### Setup Instructions

1. Clone the repository:
```bash
git clone https://github.com/yourusername/face-recognition-system.git
cd face-recognition-system
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Run the setup script to install dependencies and configure the system:
```bash
python setup.py
```

4. Download the LFW dataset (if you want to train the model yourself):
```bash
python download_models.py
```
This script will download the LFW (Labeled Faces in the Wild) dataset for training.

5. Download the Caffe models for better face detection (optional but recommended):
```bash
python download_caffe_model.py
```

6. Train the model to generate the face_classifier.pkl file:
```bash
python train.py --data_dir lfw --min_images 3 --augmentation
```
This step will create the `models/face_classifier.pkl` file necessary for face recognition.

7. Run the application:
```bash
streamlit run app.py
```

8. Access the application in your browser at `http://localhost:8501`

> **Important:** The models and dataset files will be automatically downloaded and generated during the setup process. You do not need to manually download any files that are not included in the repository.

## Face Detection Options

This system offers multiple methods for face detection, automatically using the best available option:

1. **DNN-based Detector (Caffe Model)**: Most accurate detection method (primary choice if available)
   - Great for various face angles and lighting conditions
   - Slower but more robust

2. **DNN-based Detector (TensorFlow Model)**: Alternative DNN method (secondary choice)
   - Similar performance to Caffe model
   - Used as fallback if Caffe model fails

3. **Haar Cascade Classifier**: Basic method using OpenCV's pre-trained Haar Cascade
   - Faster but less accurate in challenging scenarios
   - Used as fallback if DNN models are unavailable

The system automatically tries each method in the order listed above, using the best available option.

## Training the Model

### Dataset Organization

The LFW dataset will be automatically organized in the correct format when downloaded:

```
lfw/
├── Aaron_Eckhart/
│   ├── Aaron_Eckhart_0001.jpg
│   └── ...
├── Aaron_Guiel/
│   ├── Aaron_Guiel_0001.jpg
│   └── ...
└── ...
```

### Training Options

Standard training:
```bash
python train.py --data_dir lfw --min_images 3 --augmentation
```

For memory-constrained systems, use incremental training:
```bash
python train.py --mode incremental --data_dir lfw --start_people 5 --max_people 20 --augmentation
```

### Training Parameters

- `--data_dir`: Path to your dataset directory
- `--min_images`: Minimum number of images required per person (default: 3)
- `--mode`: Training mode - 'direct' or 'incremental'
- `--augmentation`: Enable data augmentation (recommended)
- `--start_people`, `--increment`, `--max_people`: Parameters for incremental training

## Model Verification

To verify your trained model and test the recognition pipeline:

```bash
python verify_model.py
```

This will:
- Check model integrity and structure
- Display support vectors per class
- Visualize model with PCA (saved as an image)
- Test the recognition pipeline with a sample image (if available)

## Technical Implementation

### Face Detection and Processing Pipeline

1. **Face Detection**: 
   - First attempts to use OpenCV DNN (Caffe model)
   - Falls back to OpenCV DNN (TensorFlow model) if Caffe fails
   - Uses Haar Cascade if both DNN models are unavailable

2. **Face Alignment**:
   - Detects eyes using OpenCV's eye cascade
   - Calculates angle between eyes and rotates face to align
   - Crops and scales the face based on eye positions

3. **Feature Extraction**:
   - Converts aligned face to grayscale
   - Extracts HOG (Histogram of Oriented Gradients) features
   - Creates standardized feature vectors for recognition

4. **Recognition**:
   - Uses trained SVM classifier to identify faces
   - Applies confidence boosting to enhance recognition reliability
   - Returns identified person name and confidence score

### Data Augmentation

When enabled, the system applies these augmentations to training data:
- Horizontal flips
- Small rotations (±5 degrees)
- Brightness and contrast adjustments

This increases the effective training dataset size and improves model robustness.

## Deployment

### Streamlit Cloud Deployment

The application is designed to work seamlessly on Streamlit Cloud:

1. Push your code to GitHub (without the models/ and lfw/ directories)
2. Create an account on [Streamlit Cloud](https://streamlit.io/cloud)
3. Connect your GitHub repository
4. Set the main file path to `app.py`
5. When the app starts for the first time, it will:
   - Inform users that detection works initially (using Haar Cascade)
   - Provide instructions for downloading the DNN models for better detection
   - Guide users to train a model for recognition functionality

### Render Deployment

1. Push your code to GitHub
2. Create an account on [Render](https://render.com)
3. Create a new Web Service
4. Connect your GitHub repository
5. Use the following settings:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `streamlit run app.py`

For detailed deployment instructions, run:
```bash
python deploy.py
```

## Troubleshooting

### Common Issues and Solutions

1. **Face detection not working properly**:
   - Run `python download_caffe_model.py` to download the DNN models
   - Ensure adequate lighting in images
   - Try with a frontal face image first

2. **Memory issues during training**:
   - Use incremental training with fewer people: `--mode incremental --start_people 5`
   - Reduce minimum images per person: `--min_images 3`
   - Train without augmentation initially (remove `--augmentation` flag)

3. **Recognition accuracy issues**:
   - Ensure training with varied images (different angles, lighting)
   - Try using data augmentation: `--augmentation`
   - Verify model performance: `python verify_model.py`

4. **Application not starting**:
   - Check that all dependencies are installed: `pip install -r requirements.txt`
   - Verify Python version (3.7+): `python --version`
   - Check port availability (8501 is default for Streamlit)

## Screenshots

Here are some screenshots showing the application in action:

![Face Recognition Interface](/result_screenshots/screenshot1.png)

*Main application interface with uploaded image and recognition results*

![Multiple Face Detection](/result_screenshots/screenshot2.png)

*Detection of faces by uploading the photo*

![Multiple Face Detection](/result_screenshots/screenshot3.png)

*Detection of faces by uploading take photo using web camera*

## Author

Razeen Iqbal
