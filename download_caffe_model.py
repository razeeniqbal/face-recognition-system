import os
import requests
from tqdm import tqdm

def download_file(url, destination):
    """Download a file with progress bar"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024
    
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    
    progress_bar = tqdm(
        total=total_size, 
        unit='iB', 
        unit_scale=True,
        desc=f"Downloading {os.path.basename(destination)}"
    )
    
    with open(destination, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    
    progress_bar.close()

def download_caffe_models():
    """Download Caffe SSD face detector models (more compatible with OpenCV)"""
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    
    # Caffe model files (more compatible with recent OpenCV versions)
    caffe_model_files = {
        "res10_300x300_ssd_iter_140000.caffemodel": "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel",
        "deploy.prototxt": "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
    }
    
    print("Downloading OpenCV DNN Caffe face detector models (recommended for compatibility)...")
    for filename, url in caffe_model_files.items():
        filepath = os.path.join(models_dir, filename)
        
        if os.path.exists(filepath):
            print(f"{filename} already exists at {filepath}")
            continue
        
        try:
            download_file(url, filepath)
            print(f"Downloaded {filename} successfully!")
        except Exception as e:
            print(f"Error downloading {filename}: {str(e)}")
    
    print("\nCaffe models downloaded. These models are generally more compatible with OpenCV.")
    print("You can now run the app with `streamlit run app.py`")

if __name__ == "__main__":
    download_caffe_models()