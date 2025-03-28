import os
import shutil
import argparse
import subprocess
import sys

def check_dependencies():
    """Check if required packages are installed"""
    try:
        import streamlit
        import cv2
        import sklearn
        print("All required packages are installed!")
        return True
    except ImportError as e:
        print(f"Missing dependency: {str(e)}")
        print("Please install all dependencies using: pip install -r requirements.txt")
        return False

def check_models():
    """Check if trained model is present"""
    model_path = 'models/face_classifier.pkl'
    
    if not os.path.exists(model_path):
        print(f"Warning: Face recognition model not found at {model_path}")
        print("The system will still work for face detection, but not recognition.")
        print("To train the model, run: python train.py")
        return False
    
    print("Face recognition model is present!")
    return True

def prepare_for_deployment(target_dir="deployment"):
    """Prepare files for deployment"""
    # Create deployment directory
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    os.makedirs(target_dir)
    
    # Files to copy
    files_to_copy = [
        "app.py",
        "utils.py",
        "requirements.txt",
        "README.md"
    ]
    
    # Create models directory
    os.makedirs(os.path.join(target_dir, "models"), exist_ok=True)
    
    # Copy files
    for file_path in files_to_copy:
        if os.path.exists(file_path):
            dest_path = os.path.join(target_dir, file_path)
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            shutil.copy2(file_path, dest_path)
            print(f"Copied: {file_path}")
        else:
            print(f"Warning: Could not find {file_path}")
    
    # Copy model if it exists
    model_path = "models/face_classifier.pkl"
    if os.path.exists(model_path):
        dest_path = os.path.join(target_dir, model_path)
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        shutil.copy2(model_path, dest_path)
        print(f"Copied: {model_path}")
    else:
        print(f"Warning: Model file {model_path} not found")
    
    print(f"\nDeployment files prepared in '{target_dir}' directory!")
    print("You can now upload these files to your deployment platform.")

def deploy_to_streamlit_cloud():
    """Instructions for deploying to Streamlit Cloud"""
    print("\n=== Deploying to Streamlit Cloud ===")
    print("To deploy to Streamlit Cloud:")
    print("1. Create a GitHub repository and push your code")
    print("2. Visit https://streamlit.io/cloud")
    print("3. Sign in and click 'New app'")
    print("4. Connect to your GitHub repository")
    print("5. Set the main file path to 'app.py'")
    print("6. Click 'Deploy'")
    
    print("\nIMPORTANT: Make sure your GitHub repository includes:")
    print("- app.py")
    print("- utils.py")
    print("- requirements.txt")
    print("- models/face_classifier.pkl (optional, for recognition)")

def deploy_to_render():
    """Instructions for deploying to Render"""
    print("\n=== Deploying to Render ===")
    print("To deploy to Render:")
    print("1. Create a GitHub repository and push your code")
    print("2. Visit https://render.com")
    print("3. Sign up and create a new Web Service")
    print("4. Connect to your GitHub repository")
    print("5. Use the following settings:")
    print("   - Build Command: pip install -r requirements.txt")
    print("   - Start Command: streamlit run app.py")
    print("6. Click 'Create Web Service'")

def run_local():
    """Run the app locally"""
    if not check_dependencies():
        return
    
    check_models()
    
    print("\nStarting Streamlit app...")
    subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])

def main():
    parser = argparse.ArgumentParser(description="Face Recognition App Deployment Helper")
    parser.add_argument("--prepare", action="store_true", help="Prepare files for deployment")
    parser.add_argument("--run", action="store_true", help="Run the app locally")
    parser.add_argument("--target", type=str, default="deployment", help="Target directory for deployment files")
    
    args = parser.parse_args()
    
    if args.prepare:
        prepare_for_deployment(args.target)
    elif args.run:
        run_local()
    else:
        # No arguments provided, show menu
        print("Face Recognition App Deployment Helper")
        print("1. Check dependencies and models")
        print("2. Prepare files for deployment")
        print("3. Show Streamlit Cloud deployment instructions")
        print("4. Show Render deployment instructions") 
        print("5. Run app locally")
        print("6. Exit")
        
        choice = input("Enter your choice (1-6): ")
        
        if choice == "1":
            check_dependencies()
            check_models()
        elif choice == "2":
            target = input("Enter target directory [deployment]: ") or "deployment"
            prepare_for_deployment(target)
        elif choice == "3":
            deploy_to_streamlit_cloud()
        elif choice == "4":
            deploy_to_render()
        elif choice == "5":
            run_local()
        elif choice == "6":
            print("Exiting...")
        else:
            print("Invalid choice!")

if __name__ == "__main__":
    main()