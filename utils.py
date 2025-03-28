import cv2
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import gc
import random

# Initialize face detector with Haar Cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Global variables for face detection
use_dnn_detector = False
dnn_face_detector = None
dnn_model_type = None  # 'caffe' or 'tensorflow'

# Try to initialize face detection models
def initialize_face_detectors():
    global use_dnn_detector, dnn_face_detector, dnn_model_type
    
    # First try Caffe model (more compatible with OpenCV)
    caffe_model_file = "models/res10_300x300_ssd_iter_140000.caffemodel"
    caffe_config_file = "models/deploy.prototxt"
    
    if os.path.exists(caffe_model_file) and os.path.exists(caffe_config_file):
        try:
            detector = cv2.dnn.readNetFromCaffe(caffe_config_file, caffe_model_file)
            
            # Test the model with a small dummy image
            dummy_img = np.zeros((300, 300, 3), dtype=np.uint8)
            blob = cv2.dnn.blobFromImage(dummy_img, 1.0, (300, 300), [104, 117, 123])
            detector.setInput(blob)
            _ = detector.forward()  # This will raise an exception if model is incompatible
            
            # If we got here, the model works
            use_dnn_detector = True
            dnn_face_detector = detector
            dnn_model_type = 'caffe'
            print("Using OpenCV DNN (Caffe model) for face detection")
            return True
        except Exception as e:
            print(f"Error loading Caffe DNN model: {str(e)}. Will try TensorFlow model.")
    
    # If Caffe model failed or doesn't exist, try TensorFlow model
    tf_model_file = "models/opencv_face_detector_uint8.pb"
    tf_config_file = "models/opencv_face_detector.pbtxt"
    
    if os.path.exists(tf_model_file) and os.path.exists(tf_config_file):
        try:
            detector = cv2.dnn.readNetFromTensorflow(tf_model_file, tf_config_file)
            
            # Test the model with a small dummy image
            dummy_img = np.zeros((300, 300, 3), dtype=np.uint8)
            blob = cv2.dnn.blobFromImage(dummy_img, 1.0, (300, 300), [104, 117, 123])
            detector.setInput(blob)
            _ = detector.forward()  # This will raise an exception if model is incompatible
            
            # If we got here, the model works
            use_dnn_detector = True
            dnn_face_detector = detector
            dnn_model_type = 'tensorflow'
            print("Using OpenCV DNN (TensorFlow model) for face detection")
            return True
        except Exception as e:
            print(f"Error loading TensorFlow DNN model: {str(e)}. Using Haar Cascade instead.")
    
    # If both DNN models failed, use Haar Cascade
    print("Using OpenCV Haar Cascade for face detection")
    use_dnn_detector = False
    dnn_face_detector = None
    dnn_model_type = None
    return False

# Initialize face detectors
initialize_face_detectors()

def detect_faces_caffe(image):
    """Detect faces using OpenCV DNN with Caffe model"""
    (h, w) = image.shape[:2]
    
    # Create a blob from the image
    blob = cv2.dnn.blobFromImage(
        cv2.resize(image, (300, 300)), 1.0, (300, 300), 
        [104, 117, 123], False, False
    )
    
    # Set the blob as input to the network
    dnn_face_detector.setInput(blob)
    
    # Get the detections
    detections = dnn_face_detector.forward()
    
    faces = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:  # Confidence threshold
            # Get the box coordinates
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")
            
            # Ensure coordinates are within image boundaries
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            # Convert to x, y, w, h format
            faces.append((x1, y1, x2-x1, y2-y1))
    
    return faces

def detect_faces_tensorflow(image):
    """Detect faces using OpenCV DNN with TensorFlow model"""
    (h, w) = image.shape[:2]
    
    # Create a blob from the image
    blob = cv2.dnn.blobFromImage(
        cv2.resize(image, (300, 300)), 1.0, (300, 300), 
        [104, 117, 123], False, False
    )
    
    # Set the blob as input to the network
    dnn_face_detector.setInput(blob)
    
    # Get the detections
    detections = dnn_face_detector.forward()
    
    faces = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:  # Confidence threshold
            # Get the box coordinates
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")
            
            # Ensure coordinates are within image boundaries
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            # Convert to x, y, w, h format
            faces.append((x1, y1, x2-x1, y2-y1))
    
    return faces

def detect_faces_haar(image):
    """Detect faces using Haar Cascade"""
    # Convert to grayscale for Haar Cascade
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Apply Haar Cascade with default parameters
    faces = face_cascade.detectMultiScale(
        gray, 
        scaleFactor=1.1, 
        minNeighbors=5, 
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    return faces

def detect_faces(image):
    """Detect faces in an image using the best available method"""
    # Safety check
    if image is None or image.size == 0:
        return []
        
    # Make a copy to avoid modifying the original
    img_copy = image.copy()
    
    # Convert to BGR if it's in RGB format (for DNN)
    if len(img_copy.shape) == 3 and img_copy.shape[2] == 3:
        bgr_image = cv2.cvtColor(img_copy, cv2.COLOR_RGB2BGR)
    else:
        bgr_image = img_copy
    
    faces = []
    
    # Try DNN detection if available
    if use_dnn_detector and dnn_face_detector is not None:
        try:
            if dnn_model_type == 'caffe':
                faces = detect_faces_caffe(bgr_image)
            elif dnn_model_type == 'tensorflow':
                faces = detect_faces_tensorflow(bgr_image)
                
            # If DNN detection found faces, return them
            if len(faces) > 0:
                return faces
        except Exception as e:
            print(f"DNN face detection failed: {str(e)}. Falling back to Haar Cascade.")
    
    # Fallback to Haar Cascade
    try:
        faces = detect_faces_haar(img_copy)
    except Exception as e:
        print(f"Haar Cascade face detection failed: {str(e)}")
        return []
    
    return faces

def align_face_opencv(face_img):
    """Align face using OpenCV eye detection"""
    # Safety check
    if face_img is None or face_img.size == 0:
        return None
        
    # Make a copy to avoid modifying original
    face_img_copy = face_img.copy()
    
    # Convert to grayscale if needed
    if len(face_img_copy.shape) == 3:
        gray = cv2.cvtColor(face_img_copy, cv2.COLOR_BGR2GRAY)
    else:
        gray = face_img_copy
        face_img_copy = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    
    # Detect eyes
    eyes = eye_cascade.detectMultiScale(gray, 1.1, 3)
    
    if len(eyes) >= 2:
        try:
            # Sort by x-coordinate to get left and right eyes
            eyes = sorted(eyes, key=lambda x: x[0])
            
            # Get the two eyes farthest apart horizontally
            left_eye = eyes[0]
            right_eye = eyes[-1]
            
            # Get eye centers
            left_eye_center = (int(left_eye[0] + left_eye[2]//2), int(left_eye[1] + left_eye[3]//2))
            right_eye_center = (int(right_eye[0] + right_eye[2]//2), int(right_eye[1] + right_eye[3]//2))
            
            # Calculate angle
            dY = right_eye_center[1] - left_eye_center[1]
            dX = right_eye_center[0] - left_eye_center[0]
            angle = np.degrees(np.arctan2(dY, dX))
            
            # Get center of eyes as a tuple of integers
            eye_center = (int((left_eye_center[0] + right_eye_center[0]) // 2),
                          int((left_eye_center[1] + right_eye_center[1]) // 2))
            
            # Get rotation matrix
            M = cv2.getRotationMatrix2D(eye_center, angle, 1)
            
            # Apply affine transformation
            height, width = face_img_copy.shape[:2]
            aligned_face = cv2.warpAffine(face_img_copy, M, (width, height), flags=cv2.INTER_CUBIC)
            
            # Crop to face area and add margin
            face_width = int(abs(right_eye_center[0] - left_eye_center[0]) * 2.5)
            face_height = int(face_width * 1.2)  # Adjust aspect ratio
            
            x = max(0, eye_center[0] - face_width // 2)
            y = max(0, eye_center[1] - face_height // 3)  # Eyes are in upper part of face
            
            # Make sure we don't go out of bounds
            if x + face_width > width:
                face_width = width - x
            if y + face_height > height:
                face_height = height - y
            
            # Check if we have valid crop dimensions
            if face_width <= 0 or face_height <= 0:
                return face_img_copy
                
            face_cropped = aligned_face[y:y+face_height, x:x+face_width]
            
            # Check if cropped face is valid
            if face_cropped is None or face_cropped.size == 0:
                return face_img_copy
                
            return face_cropped
        except Exception as e:
            print(f"Error in face alignment: {str(e)}")
            return face_img_copy
    
    # If eye detection fails, return the original face
    return face_img_copy

def align_face(face_img, target_size=160):
    """Align and resize a face image using OpenCV"""
    # Safety check
    if face_img is None or face_img.size == 0:
        # Return a blank image of target size
        return np.zeros((target_size, target_size, 3), dtype=np.uint8)
    
    # If the face is too small, just resize
    if face_img.shape[0] < 20 or face_img.shape[1] < 20:
        return cv2.resize(face_img, (target_size, target_size))
    
    # Try to align the face
    aligned_face = align_face_opencv(face_img)
    
    # If alignment failed, use original
    if aligned_face is None or aligned_face.size == 0:
        aligned_face = face_img
    
    # Resize to target size
    try:
        return cv2.resize(aligned_face, (target_size, target_size))
    except Exception as e:
        print(f"Error resizing face: {str(e)}")
        # Return a blank image if resize fails
        return np.zeros((target_size, target_size, 3), dtype=np.uint8)

def normalize_face(face_img):
    """Normalize face for model input"""
    # Safety check
    if face_img is None or face_img.size == 0:
        # Return empty array
        if len(face_img.shape) == 3:
            h, w, _ = face_img.shape
            return np.zeros((h, w), dtype=np.float32)
        else:
            return np.zeros_like(face_img, dtype=np.float32)
    
    # Convert to grayscale if it's a color image
    if len(face_img.shape) == 3:
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = face_img
    
    # Apply histogram equalization to improve contrast
    try:
        equalized = cv2.equalizeHist(gray)
        # Normalize pixel values
        normalized = equalized / 255.0
        return normalized
    except Exception as e:
        print(f"Error normalizing face: {str(e)}")
        # Return zero-filled array if normalization fails
        return np.zeros_like(gray, dtype=np.float32)

def extract_features(face_img):
    """Extract HOG features from a face image"""
    # Safety check
    if face_img is None or face_img.size == 0:
        # Return empty feature vector
        return np.array([])
    
    try:
        # Convert to grayscale if needed
        if len(face_img.shape) == 3:
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_img
        
        # Ensure gray image is uint8 (required by HOG)
        if gray.dtype != np.uint8:
            gray = (gray * 255).astype(np.uint8)
        
        # We'll use HOG features
        win_size = (64, 64)
        block_size = (16, 16)
        block_stride = (8, 8)
        cell_size = (8, 8)
        nbins = 9
        
        # Resize for HOG descriptor
        resized = cv2.resize(gray, win_size)
        
        # Create HOG descriptor
        hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
        
        # Compute HOG features
        features = hog.compute(resized)
        
        return features.flatten()
    except Exception as e:
        print(f"Error extracting HOG features: {str(e)}")
        # Return empty array
        return np.array([])

def augment_image(face_img):
    """Apply data augmentation to face image"""
    # Safety check
    if face_img is None or face_img.size == 0:
        return [face_img]  # Return original only
    
    augmented_images = [face_img]  # Original image
    
    try:
        # Flip horizontally
        flipped = cv2.flip(face_img, 1)
        augmented_images.append(flipped)
        
        # Small rotations
        for angle in [-5, 5]:
            try:
                center = (face_img.shape[1]//2, face_img.shape[0]//2)
                M = cv2.getRotationMatrix2D(center, angle, 1)
                rotated = cv2.warpAffine(face_img, M, (face_img.shape[1], face_img.shape[0]))
                augmented_images.append(rotated)
            except Exception as e:
                print(f"Error in rotation augmentation: {str(e)}")
        
        # Small brightness/contrast changes
        for alpha in [0.9, 1.1]:  # Contrast
            for beta in [-10, 10]:  # Brightness
                try:
                    adjusted = cv2.convertScaleAbs(face_img, alpha=alpha, beta=beta)
                    augmented_images.append(adjusted)
                except Exception as e:
                    print(f"Error in brightness/contrast augmentation: {str(e)}")
    except Exception as e:
        print(f"Error in augmentation: {str(e)}")
    
    # Filter out None or empty images
    return [img for img in augmented_images if img is not None and img.size > 0]

def recognize_face(face_img, confidence_threshold=0.3):
    """Recognize a face using the trained classifier with confidence boosting"""
    # Check if model exists
    if not os.path.exists('models/face_classifier.pkl'):
        print("No face recognition model found at models/face_classifier.pkl")
        return "Unknown", 0.0
    
    try:
        # Load the classifier
        with open('models/face_classifier.pkl', 'rb') as f:
            model_data = pickle.load(f)
            classifier = model_data.get('classifier')
            person_names = model_data.get('person_names', [])
        
        if classifier is None or not person_names:
            print("Invalid model: classifier or person_names missing")
            return "Unknown", 0.0
        
        print(f"Model loaded with {len(person_names)} people")
        
        # Extract features
        features = extract_features(face_img)
        
        # If feature extraction failed
        if features.size == 0:
            print("Feature extraction failed")
            return "Unknown", 0.0
        
        # Reshape for model
        features = features.reshape(1, -1)
        
        # Predict
        if hasattr(classifier, 'predict_proba'):
            # SVM with probability
            probs = classifier.predict_proba(features)[0]
            
            # Apply confidence boosting
            # This enhances the highest probability while maintaining relative rankings
            # Using softmax with temperature to rescale probabilities
            temperature = 0.5  # Lower values increase confidence (0.5 is a good starting point)
            scaled_probs = np.exp(np.log(probs + 1e-10) / temperature)
            boosted_probs = scaled_probs / np.sum(scaled_probs)
            
            label = np.argmax(boosted_probs)
            original_confidence = probs[label]
            confidence = boosted_probs[label]
            
            # Print top 3 predictions for debugging
            top_indices = np.argsort(probs)[-3:][::-1]
            for idx in top_indices:
                print(f"Candidate: {person_names[idx]} with confidence {probs[idx]:.4f} (boosted: {boosted_probs[idx]:.4f})")
        else:
            # Regular SVM
            label = classifier.predict(features)[0]
            # Use decision_function as a proxy for confidence
            confidence = 0.5  # Default if decision_function not available
            if hasattr(classifier, 'decision_function'):
                dvals = classifier.decision_function(features)[0]
                if isinstance(dvals, np.ndarray):
                    # Apply boosting to decision values
                    raw_confidence = (dvals[label] + 1) / 2  # Scale to 0-1
                    confidence = np.tanh(raw_confidence * 2) * 0.5 + 0.5  # Boost using tanh
                else:
                    raw_confidence = (dvals + 1) / 2
                    confidence = np.tanh(raw_confidence * 2) * 0.5 + 0.5  # Boost using tanh
            
            print(f"SVM prediction (no probability): {person_names[label]} with confidence {confidence:.4f}")
        
        # Return predicted name and confidence
        if label < len(person_names):
            print(f"Recognized: {person_names[label]} with confidence {confidence:.4f}")
            return person_names[label], confidence
        else:
            print(f"Invalid label")
            return "Unknown", confidence
    
    except Exception as e:
        print(f"Error in recognition: {str(e)}")
        import traceback
        traceback.print_exc()
        return "Unknown", 0.0

def process_image(image_np, debug_mode=False):
    """Process an image and return face recognition results with debugging"""
    # Safety check
    if image_np is None or image_np.size == 0:
        # Return empty results
        return image_np, []
    
    try:
        # Convert image to RGB if needed
        if len(image_np.shape) == 2:  # Grayscale
            image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
        elif len(image_np.shape) == 3 and image_np.shape[2] == 4:  # RGBA
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)
        
        # Create a copy for drawing results
        result_image = image_np.copy()
        
        # Detect faces
        if debug_mode:
            print("STEP 1: Face Detection")
        face_locations = detect_faces(image_np)
        if debug_mode:
            print(f"Detected {len(face_locations)} faces")
        
        # Process each detected face
        face_results = []
        for i, (x, y, w, h) in enumerate(face_locations):
            # Draw rectangle
            cv2.rectangle(result_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            try:
                # Extract face region
                face_img = image_np[y:y+h, x:x+w]
                
                if debug_mode:
                    print(f"\nFace #{i+1} - Position: ({x}, {y}, {w}, {h})")
                    print("STEP 2: Face Alignment")
                
                # Align face with OpenCV
                aligned_face = align_face(face_img)
                
                # Only do recognition if we have a classifier
                if os.path.exists('models/face_classifier.pkl'):
                    if debug_mode:
                        print("STEP 3: Face Normalization")
                    
                    # Normalize the face
                    normalized_face = normalize_face(aligned_face)
                    
                    if debug_mode:
                        print("STEP 4: Feature Extraction and Recognition")
                    
                    # Recognize face
                    name, confidence = recognize_face(normalized_face)
                    
                    # Only show the name without confidence
                    label = name
                    
                    face_results.append({
                        'location': (x, y, w, h),
                        'name': name,
                        'confidence': confidence,  # Keep this for internal use
                        'label': label
                    })
                else:
                    if debug_mode:
                        print("No face recognition model found")
                    
                    # No classifier, just label as "Face"
                    label = "Face Detected"
                    face_results.append({
                        'location': (x, y, w, h),
                        'name': "Unknown",
                        'confidence': 0.0,
                        'label': label
                    })
                
                # Put label on image
                cv2.putText(result_image, label, (x, y-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            except Exception as e:
                print(f"Error processing face: {str(e)}")
                cv2.putText(result_image, "Error", (x, y-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        
        return result_image, face_results
    except Exception as e:
        print(f"Error in process_image: {str(e)}")
        return image_np, []

def extract_features(face_img):
    """Extract HOG features from a face image with extra safety checks"""
    # Safety check
    if face_img is None or face_img.size == 0:
        # Return empty feature vector
        print("Cannot extract features: Empty image")
        return np.array([])
    
    try:
        # Convert to grayscale if needed
        if len(face_img.shape) == 3:
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_img
        
        # Ensure gray image is uint8 (required by HOG)
        if gray.dtype != np.uint8:
            gray = (gray * 255).astype(np.uint8)
        
        # Check if image size is valid
        if gray.shape[0] < 10 or gray.shape[1] < 10:
            print(f"Image too small for feature extraction: {gray.shape}")
            return np.array([])
        
        # We'll use HOG features
        win_size = (64, 64)
        block_size = (16, 16)
        block_stride = (8, 8)
        cell_size = (8, 8)
        nbins = 9
        
        # Resize for HOG descriptor
        resized = cv2.resize(gray, win_size)
        
        # Create HOG descriptor
        hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
        
        # Compute HOG features
        features = hog.compute(resized)
        
        # Check if features are valid
        if features is None or features.size == 0:
            print("HOG returned empty features")
            return np.array([])
            
        # Check for NaN or Inf values
        if np.isnan(features).any() or np.isinf(features).any():
            print("Features contain NaN or Inf values")
            # Replace NaN/Inf with zeros
            features = np.nan_to_num(features)
        
        return features.flatten()
    except Exception as e:
        print(f"Error extracting HOG features: {str(e)}")
        import traceback
        traceback.print_exc()
        # Return empty array
        return np.array([])

def verify_model_integrity():
    """Verify that the face recognition model is valid and contains data"""
    if not os.path.exists('models/face_classifier.pkl'):
        print("No face recognition model found")
        return False
        
    try:
        with open('models/face_classifier.pkl', 'rb') as f:
            model_data = pickle.load(f)
            
        classifier = model_data.get('classifier')
        person_names = model_data.get('person_names', [])
        metrics = model_data.get('metrics', {})
        
        if classifier is None:
            print("Model is missing the classifier")
            return False
            
        if not person_names:
            print("Model has no person names")
            return False
            
        print(f"Model contains {len(person_names)} people:")
        for i, name in enumerate(person_names):
            print(f"  {i+1}. {name}")
            
        print("Model metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")
            
        # Check if model has support vectors
        if hasattr(classifier, 'support_vectors_'):
            print(f"Model has {classifier.support_vectors_.shape[0]} support vectors")
            print(f"Feature vector length: {classifier.support_vectors_.shape[1]}")
        else:
            print("Model doesn't have support vectors (not an SVM?)")
            
        return True
    except Exception as e:
        print(f"Error verifying model: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def load_dataset(dataset_dir, min_images_per_person=3, max_people=None, use_augmentation=True):
    """Load dataset and keep only people with minimum number of images"""
    print(f"Loading dataset from {dataset_dir}")
    face_features = []
    face_labels = []
    person_names = []
    
    # Get list of all person directories
    valid_persons = []
    
    try:
        for person_name in os.listdir(dataset_dir):
            person_dir = os.path.join(dataset_dir, person_name)
            if os.path.isdir(person_dir):
                # Get all image files
                image_files = [f for f in os.listdir(person_dir) 
                             if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                
                # Only use people with at least min_images_per_person images
                if len(image_files) >= min_images_per_person:
                    valid_persons.append((person_name, image_files))
                else:
                    print(f"Skipping {person_name}: only {len(image_files)} images (need {min_images_per_person})")
        
        # Apply max_people limit if specified
        if max_people is not None:
            valid_persons = valid_persons[:max_people]
        
        print(f"Loading images from {len(valid_persons)} people...")
        
        # Now load images for valid persons
        for person_idx, (person_name, image_files) in enumerate(valid_persons):
            person_dir = os.path.join(dataset_dir, person_name)
            
            if person_idx % 20 == 0 or person_idx == len(valid_persons) - 1:
                print(f"Processing person {person_idx+1}/{len(valid_persons)}: {person_name}")
            
            # Add person to list
            person_id = len(person_names)
            person_names.append(person_name)
            
            # Count successful face detections for this person
            detected_faces = 0
            
            # Load each image
            for image_file in image_files:
                try:
                    image_path = os.path.join(person_dir, image_file)
                    image = cv2.imread(image_path)
                    if image is not None:
                        # Detect face
                        faces = detect_faces(image)
                        if len(faces) > 0:
                            detected_faces += 1
                            # Get largest face
                            largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
                            x, y, w, h = largest_face
                            
                            # Extract face region
                            face_img = image[y:y+h, x:x+w]
                            
                            # Align and normalize face
                            aligned_face = align_face(face_img)
                            normalized_face = normalize_face(aligned_face)
                            
                            # Apply data augmentation if requested
                            if use_augmentation:
                                augmented_faces = augment_image(normalized_face)
                                
                                # Extract features from each augmented face
                                for aug_face in augmented_faces:
                                    try:
                                        features = extract_features(aug_face)
                                        if features.size > 0:  # Check that feature extraction worked
                                            face_features.append(features)
                                            face_labels.append(person_id)
                                    except Exception as e:
                                        print(f"Error extracting features during augmentation: {str(e)}")
                            else:
                                # Extract features
                                features = extract_features(normalized_face)
                                if features.size > 0:  # Check that feature extraction worked
                                    face_features.append(features)
                                    face_labels.append(person_id)
                        else:
                            print(f"No face detected in {image_path}")
                except Exception as e:
                    print(f"Error processing image {image_file}: {str(e)}")
            
            # Check if we detected enough faces for this person
            if detected_faces < min_images_per_person:
                # Remove this person from the list
                person_names.pop()
                # Remove their data from features and labels
                indices_to_keep = [i for i, label in enumerate(face_labels) if label != person_id]
                face_features = [face_features[i] for i in indices_to_keep]
                face_labels = [face_labels[i] for i in indices_to_keep]
                # Adjust the remaining labels
                face_labels = [label if label < person_id else label - 1 for label in face_labels]
                
                print(f"Removing {person_name}: Only detected {detected_faces} faces (need {min_images_per_person})")
            else:
                print(f"Added {person_name} with {detected_faces} detected faces")
        
        if len(face_features) == 0:
            raise Exception("No faces detected in the dataset!")
        
        # Check that all feature vectors have the same length
        feature_lengths = [len(f) for f in face_features]
        if len(set(feature_lengths)) > 1:
            # Filter to only include features of the most common length
            most_common_length = max(set(feature_lengths), key=feature_lengths.count)
            valid_indices = [i for i, length in enumerate(feature_lengths) if length == most_common_length]
            face_features = [face_features[i] for i in valid_indices]
            face_labels = [face_labels[i] for i in valid_indices]
            print(f"Warning: Mixed feature vector lengths found. Kept {len(valid_indices)} features with length {most_common_length}.")
        
        print(f"Loaded {len(face_features)} faces of {len(person_names)} different people")
        
        return np.array(face_features), np.array(face_labels), person_names
    except Exception as e:
        raise Exception(f"Error loading dataset: {str(e)}")

def train_classifier(face_features, face_labels, person_names, model_save_path='models/face_classifier.pkl'):
    """Train a face recognition classifier using extracted features"""
    print("Training face recognition classifier...")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    
    # Check for valid data
    if len(face_features) == 0 or len(face_labels) == 0:
        raise Exception("No valid features or labels to train on!")
    
    # Ensure all feature vectors have the same length
    feature_lengths = [f.shape[0] for f in face_features]
    if len(set(feature_lengths)) > 1:
        raise Exception("Feature vectors have inconsistent lengths!")
    
    try:
        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            face_features, face_labels, test_size=0.2, random_state=42, stratify=face_labels)
        
        # Train SVM classifier
        print("Training SVM classifier...")
        classifier = SVC(kernel='linear', probability=True, C=1.0)
        classifier.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average='weighted')
        
        print("Model Evaluation:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-score: {f1:.4f}")
        
        # Save model and metadata
        model_data = {
            'classifier': classifier,
            'person_names': person_names,
            'metrics': {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
        }
        
        with open(model_save_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {model_save_path}")
        
        return model_data
    except Exception as e:
        raise Exception(f"Error training classifier: {str(e)}")