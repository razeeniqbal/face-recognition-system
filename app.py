import streamlit as st
import cv2
import numpy as np
import os
import pickle
from PIL import Image
import tempfile

# Import utility functions
from utils import detect_faces, process_image, recognize_face, verify_model_integrity

# Page configuration
st.set_page_config(
    page_title="Face Recognition System",
    page_icon="👤",
    layout="wide"
)

def show_metrics():
    """Show model metrics as a markdown table in rows"""
    if os.path.exists('models/face_classifier.pkl'):
        try:
            with open('models/face_classifier.pkl', 'rb') as f:
                model_data = pickle.load(f)
                metrics = model_data.get('metrics', {})
                
            if metrics:
                st.write("**Model Performance Metrics:**")
                
                # Create a markdown table
                table = """
                | Metric | Value |
                | ------ | ----- |
                | Accuracy | {:.4f} |
                | Precision | {:.4f} |
                | Recall | {:.4f} |
                | F1 Score | {:.4f} |
                """.format(
                    metrics.get('accuracy', 0),
                    metrics.get('precision', 0),
                    metrics.get('recall', 0),
                    metrics.get('f1', 0)
                )
                
                st.markdown(table)
            else:
                st.warning("No metrics available")
        except Exception as e:
            st.error(f"Error loading metrics: {str(e)}")
    else:
        st.info("No model file found")

def detect_face_mode():
    """Show which face detection mode is currently active"""
    # Check if OpenCV DNN models are available
    caffe_model = "models/res10_300x300_ssd_iter_140000.caffemodel"
    caffe_config = "models/deploy.prototxt"
    
    if os.path.exists(caffe_model) and os.path.exists(caffe_config):
        try:
            # Try to load the model to confirm it works
            net = cv2.dnn.readNetFromCaffe(caffe_config, caffe_model)
            return "DNN (Caffe)"
        except:
            pass
    
    dnn_pb_file = "models/opencv_face_detector_uint8.pb"
    dnn_pbtxt_file = "models/opencv_face_detector.pbtxt"
    
    if os.path.exists(dnn_pb_file) and os.path.exists(dnn_pbtxt_file):
        try:
            # Try to load the model to confirm it works
            net = cv2.dnn.readNetFromTensorflow(dnn_pb_file, dnn_pbtxt_file)
            return "DNN (TensorFlow)"
        except:
            pass
    
    return "Haar Cascade"

def main():
    st.title("Face Recognition System")
    
    # Add a "How it works" expander
    with st.expander("How it works"):
        st.write("""
        This face recognition system works in four steps:
        1. **Face Detection**: Locates faces in the uploaded image using OpenCV
        2. **Face Alignment**: Aligns detected faces based on eye positions for better recognition
        3. **Feature Extraction**: Extracts facial features using HOG (Histogram of Oriented Gradients)
        4. **Recognition**: Matches facial features to known identities using an SVM classifier
        
        Try uploading a photo or taking one with your camera to see it in action!
        """)
    
    # Check if models exist
    has_recognition = os.path.exists('models/face_classifier.pkl')
    
    # Add debug mode in sidebar
    with st.sidebar:
        st.title("About")
        st.write("This face recognition system uses:")
        
        # Display face detection mode
        detection_mode = detect_face_mode()
        st.write(f"- OpenCV {detection_mode} for face detection")
        
        # Check what models are available
        if os.path.exists('models/face_classifier.pkl'):
            st.write("- SVM classifier for face recognition")
            
            # Show how many people can be recognized
            with open('models/face_classifier.pkl', 'rb') as f:
                model_data = pickle.load(f)
                person_names = model_data.get('person_names', [])
            st.write(f"- Can recognize {len(person_names)} different people")
        else:
            st.write("- Face detection only (recognition model not trained)")
            
        st.write("- OpenCV for face alignment")
        st.write("- HOG feature extraction")
        
        st.write("---")
        
        # Model status
        st.subheader("System Status")
        
        if os.path.exists('models/face_classifier.pkl'):
            st.success("Face recognition model: Available")
            
            # Model metrics moved to sidebar
            st.subheader("Model Performance")
            show_metrics()
        else:
            st.warning("Face recognition model: Not trained")
            st.write("Run train.py to train the recognition model")
            
        # Check if Caffe DNN models exist
        caffe_model = "models/res10_300x300_ssd_iter_140000.caffemodel"
        if os.path.exists(caffe_model):
            st.success("OpenCV DNN (Caffe) detector: Available")
        else:
            # Check if TensorFlow DNN models exist
            dnn_model = "models/opencv_face_detector_uint8.pb"
            if os.path.exists(dnn_model):
                st.success("OpenCV DNN (TensorFlow) detector: Available")
            else:
                st.info("OpenCV DNN detector: Not available")
                st.write("Run download_caffe_model.py to download the DNN face detector for better accuracy")
        
        # Debug options
        st.subheader("Debug Options")
        debug_mode = st.checkbox("Enable Debug Mode", value=False)
        
        if debug_mode:
            st.write("Debug mode enabled - check console for detailed logs")
            verify_model = st.button("Verify Model Integrity")
            
            if verify_model:
                if verify_model_integrity():
                    st.success("Model verification passed")
                else:
                    st.error("Model verification failed. See console for details.")
        
        # Add deployment instructions
        st.subheader("Deployment")
        st.write("This app can be deployed to:")
        st.write("- [Streamlit Cloud](https://streamlit.io/cloud)")
        st.write("- [Render](https://render.com)")
        st.write("Run `python deploy.py` for deployment instructions")
    
    # Add tabs for different input methods
    tab1, tab2 = st.tabs(["Upload Image", "Camera"])
    
    # Process uploaded image
    with tab1:
        uploaded_file = st.file_uploader("Choose an image file...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            # Read image
            image = Image.open(uploaded_file)
            image_np = np.array(image)
            
            col1, col2 = st.columns(2)
            
            # Display original image in first column
            with col1:
                st.subheader("Original Image")
                st.image(image, use_container_width=True)
            
            # Process image
            with st.spinner("Processing image..."):
                result_image, face_results = process_image(image_np, debug_mode)
            
            # Display result in second column
            with col2:
                st.subheader("Recognition Result")
                st.image(result_image, use_container_width=True)
            
            # Display face details
            if face_results:
                st.subheader(f"Detected {len(face_results)} faces:")
                for i, face in enumerate(face_results):
                    with st.expander(f"Face #{i+1}: {face['label']}"):
                        st.write(f"**Name:** {face['name']}")
                        # Removed confidence display
                        st.write(f"**Location (x, y, w, h):** {face['location']}")
            else:
                st.info("No faces detected in the image!")
    
    # Add camera input option
    with tab2:
        camera_image = st.camera_input("Take a picture")
        
        if camera_image is not None:
            # Read image
            image = Image.open(camera_image)
            image_np = np.array(image)
            
            # Process image
            with st.spinner("Processing image..."):
                result_image, face_results = process_image(image_np, debug_mode)
            
            # Display result
            st.subheader("Recognition Result")
            st.image(result_image, use_container_width=True)
            
            # Display face details
            if face_results:
                st.subheader(f"Detected {len(face_results)} faces:")
                for i, face in enumerate(face_results):
                    with st.expander(f"Face #{i+1}: {face['label']}"):
                        st.write(f"**Name:** {face['name']}")
                        # Removed confidence display
                        st.write(f"**Location (x, y, w, h):** {face['location']}")
            else:
                st.info("No faces detected in the image!")

if __name__ == "__main__":
    main()