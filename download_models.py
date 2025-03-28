import os
import requests
from tqdm import tqdm
import sys
import zipfile
import tarfile
import gzip

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

def download_lfw_dataset(output_dir="lfw"):
    """Download the LFW dataset"""
    lfw_url = "http://vis-www.cs.umass.edu/lfw/lfw.tgz"
    temp_file = "lfw.tgz"
    
    print("Downloading LFW dataset...")
    download_file(lfw_url, temp_file)
    
    print("Extracting dataset...")
    with tarfile.open(temp_file, 'r:gz') as tar:
        tar.extractall()
    
    # Remove temporary file
    os.remove(temp_file)
    
    print(f"LFW dataset extracted to {output_dir}")

def download_opencv_dnn_models():
    """Download OpenCV DNN face detector models"""
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    
    # Caffe model files (more compatible with recent OpenCV versions)
    caffe_model_files = {
        "res10_300x300_ssd_iter_140000.caffemodel": "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel",
        "deploy.prototxt": "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
    }
    
    # TensorFlow model files (as alternative)
    tf_model_files = {
        "opencv_face_detector_uint8.pb": "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20180205_uint8/opencv_face_detector_uint8.pb",
        "opencv_face_detector.pbtxt": "https://github.com/opencv/opencv/raw/master/samples/dnn/face_detector/opencv_face_detector.pbtxt"
    }
    
    print("Downloading OpenCV DNN Caffe face detector models (recommended)...")
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
    
    print("\nDownloading OpenCV DNN TensorFlow face detector models (alternative)...")
    for filename, url in tf_model_files.items():
        filepath = os.path.join(models_dir, filename)
        
        if os.path.exists(filepath):
            print(f"{filename} already exists at {filepath}")
            continue
        
        try:
            download_file(url, filepath)
            print(f"Downloaded {filename} successfully!")
        except Exception as e:
            print(f"Error downloading {filename}: {str(e)}")
    
    print("\nNote: The system will try to use the Caffe model first, then the TensorFlow model,")
    print("and fall back to Haar Cascade if neither DNN model works.")

def print_celeba_instructions():
    """Print instructions for downloading CelebA dataset"""
    print("\n" + "="*80)
    print("CelebA DATASET DOWNLOAD INSTRUCTIONS")
    print("="*80)
    print("\nTo download the CelebA dataset:")
    print("1. Visit: https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html")
    print("2. Download the 'Align&Cropped Images' dataset")
    print("3. Extract it to your project directory")
    print("4. Create a directory named 'celeba' in your project")
    print("5. Organize the images into subdirectories by identity using the identity labels file")
    print("\nAfter organizing, you can train your model with:")
    print("python train.py --data_dir celeba --min_images 5")
    print("="*80)

def print_lfw_instructions():
    """Print instructions for manually downloading LFW dataset"""
    print("\n" + "="*80)
    print("LFW DATASET DOWNLOAD INSTRUCTIONS")
    print("="*80)
    print("\nTo download the LFW dataset:")
    print("1. Visit: https://www.kaggle.com/datasets/jessicali9530/lfw-dataset")
    print("2. Sign in to Kaggle (or create an account)")
    print("3. Download the dataset ZIP file")
    print("4. Extract it to your project directory")
    print("5. Ensure the extracted folder is named 'lfw' (rename it if necessary)")
    print("\nAfter downloading, you can train your model with:")
    print("python train.py --data_dir lfw --min_images 5")
    print("="*80)

def main():
    """Main function to download required models and datasets"""
    # Create directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    
    print("Face Recognition System - Download Resources")
    print("============================================")
    print("\nThis script will help you download required resources:")
    print("1. LFW Dataset (Labeled Faces in the Wild)")
    print("2. OpenCV DNN face detector models (for better face detection)")
    print("3. Instructions for CelebA dataset (alternative dataset)")
    
    # Ask user what to download
    try_download_lfw = input("Would you like to download the LFW dataset? (y/n): ").lower() == 'y'
    try_download_dnn = input("Would you like to download OpenCV DNN models for better face detection? (y/n): ").lower() == 'y'
    
    if try_download_lfw:
        try:
            download_lfw_dataset()
            print("Dataset download complete!")
        except Exception as e:
            print(f"Error downloading dataset: {str(e)}")
            print_lfw_instructions()
    else:
        print_lfw_instructions()
    
    if try_download_dnn:
        try:
            download_opencv_dnn_models()
            print("DNN models download complete!")
        except Exception as e:
            print(f"Error downloading DNN models: {str(e)}")
            print("You can still use the system with Haar Cascade detection")
    
    # Always show CelebA instructions as an alternative dataset
    print_celeba_instructions()
    
    print("\nNext steps:")
    print("1. Train the model with: python train.py --data_dir lfw --min_images 5 --augmentation")
    print("2. For memory-constrained systems, use incremental training:")
    print("   python train.py --mode incremental --data_dir lfw --start_people 5 --max_people 20 --augmentation")
    print("3. Run the app with: streamlit run app.py")

if __name__ == "__main__":
    main()