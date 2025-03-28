import os
import subprocess
import sys
import time

def print_header(message):
    """Print a formatted header message"""
    print("\n" + "=" * 80)
    print(f" {message}")
    print("=" * 80)

def run_command(command, description=None):
    """Run a command with progress indication"""
    if description:
        print(f"\n> {description}...")
    
    try:
        result = subprocess.run(command, shell=True, check=False, 
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              universal_newlines=True)
        
        if result.returncode == 0:
            return True, result.stdout
        else:
            print(f"Command failed with error code {result.returncode}")
            print(f"Error output: {result.stderr}")
            return False, result.stderr
    except Exception as e:
        print(f"Error executing command: {str(e)}")
        return False, str(e)

def create_directories():
    """Create required directories"""
    directories = ["models", "data", "test_data"]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")

def verify_python_version():
    """Verify that Python version is compatible"""
    current_version = sys.version_info[:2]
    
    if current_version < (3, 7):
        print(f"Warning: Python 3.7 or higher is recommended.")
        print(f"Current version: Python {current_version[0]}.{current_version[1]}")
        return False
    
    print(f"Python version {current_version[0]}.{current_version[1]} is compatible.")
    return True

def install_requirements():
    """Install required packages"""
    print_header("Installing Required Packages")
    
    # Check if pip is available using python -m pip
    success, output = run_command("python -m pip --version", "Checking pip installation")
    if not success:
        print("Error: pip is not installed or not in PATH.")
        print("Try installing pip: python -m ensurepip --upgrade")
        return False
    else:
        print(f"Found pip: {output.strip()}")
    
    # Install core requirements
    packages = [
        "opencv-python",
        "streamlit",
        "scikit-learn",
        "numpy",
        "pillow",
        "tqdm",
        "requests"
    ]
    
    # Install packages one by one
    overall_success = True
    for package in packages:
        print(f"\nInstalling {package}...")
        cmd = f"python -m pip install {package}"
        success, _ = run_command(cmd, f"Installing {package}")
        if not success:
            print(f"Failed to install {package}.")
            overall_success = False
    
    return overall_success

def download_opencv_dnn_models():
    """Download OpenCV DNN face detector models if the user wants them"""
    print_header("Optional: Download OpenCV DNN Models")
    
    download_models = input("Would you like to download OpenCV DNN models for better face detection? (y/n): ").lower() == 'y'
    
    if not download_models:
        print("Skipping DNN model download. The system will use Haar Cascades for face detection.")
        return False
    
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    
    model_files = {
        "opencv_face_detector_uint8.pb": "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20180205_uint8/opencv_face_detector_uint8.pb",
        "opencv_face_detector.pbtxt": "https://github.com/opencv/opencv/raw/master/samples/dnn/face_detector/opencv_face_detector.pbtxt"
    }
    
    for filename, url in model_files.items():
        filepath = os.path.join(models_dir, filename)
        
        if os.path.exists(filepath):
            print(f"{filename} already exists at {filepath}")
            continue
        
        print(f"Downloading {filename}...")
        
        try:
            import requests
            response = requests.get(url)
            
            if response.status_code == 200:
                with open(filepath, 'wb') as f:
                    f.write(response.content)
                print(f"Downloaded {filename} successfully!")
            else:
                print(f"Failed to download {filename}. Status code: {response.status_code}")
                return False
        except Exception as e:
            print(f"Error downloading {filename}: {str(e)}")
            return False
    
    print("OpenCV DNN models downloaded successfully!")
    print("The system will now use DNN-based face detection for better accuracy.")
    return True

def test_basic_functionality():
    """Test if basic OpenCV functionality works"""
    print_header("Testing Basic Functionality")
    
    try:
        import cv2
        print("OpenCV is installed correctly!")
        
        # Test face detection
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        if face_cascade.empty():
            print("Warning: Could not load face cascade classifier!")
        else:
            print("Face detection is available!")
        
        # Check if OpenCV DNN models are available
        model_path = os.path.join("models", "opencv_face_detector_uint8.pb")
        config_path = os.path.join("models", "opencv_face_detector.pbtxt")
        
        if os.path.exists(model_path) and os.path.exists(config_path):
            try:
                net = cv2.dnn.readNetFromTensorflow(model_path, config_path)
                print("OpenCV DNN face detector is available!")
            except Exception as e:
                print(f"Warning: OpenCV DNN models exist but could not be loaded: {str(e)}")
        
        return True
    except ImportError:
        print("Error: OpenCV is not installed correctly!")
        return False
    except Exception as e:
        print(f"Error testing OpenCV: {str(e)}")
        return False

def main():
    """Main setup function"""
    print_header("Face Recognition System Setup (OpenCV-only)")
    
    print("This setup script will help you install and configure the face recognition system.")
    print("It uses OpenCV for face detection and recognition without any need for dlib.")
    
    # Create directories
    create_directories()
    
    # Verify Python version
    verify_python_version()
    
    # Install requirements
    install_success = install_requirements()
    if not install_success:
        print("Warning: Some packages could not be installed.")
        print("The system may still work with limited functionality.")
    
    # Download OpenCV DNN models (optional)
    download_opencv_dnn_models()
    
    # Test basic functionality
    test_basic_functionality()
    
    print_header("Setup Complete")
    print("The OpenCV-based face recognition system has been set up!")
    
    print("\nNext steps:")
    print("1. Download a dataset: python download_models.py")
    print("2. Train the model: python train.py --augmentation")
    print("3. Run the application: python -m streamlit run app.py")
    
    print("\nTroubleshooting tips:")
    print("- If face detection is not accurate enough, try running the setup again")
    print("  and download the OpenCV DNN models for better detection.")
    print("- For memory issues during training, use incremental training:")
    print("  python train.py --mode incremental --data_dir your_dataset --start_people 5 --max_people 20")
    
    print("\nGood luck with your project!")

if __name__ == "__main__":
    main()