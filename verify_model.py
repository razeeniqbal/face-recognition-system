import os
import sys
import pickle
import numpy as np
import cv2
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def verify_model():
    """Verify the face recognition model and perform diagnostics"""
    model_path = 'models/face_classifier.pkl'
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return False
    
    try:
        # Load the model
        print(f"Loading model from {model_path}...")
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        # Check model components
        classifier = model_data.get('classifier')
        person_names = model_data.get('person_names', [])
        metrics = model_data.get('metrics', {})
        
        if classifier is None:
            print("Error: Model is missing the classifier")
            return False
        
        if not person_names:
            print("Error: Model has no person names")
            return False
        
        # Print basic model information
        print("\n=== MODEL INFORMATION ===")
        print(f"Number of people: {len(person_names)}")
        print(f"People in the model: {person_names}")
        
        print("\n=== MODEL METRICS ===")
        for key, value in metrics.items():
            print(f"{key}: {value:.4f}")
        
        # Print classifier details
        print("\n=== CLASSIFIER DETAILS ===")
        if hasattr(classifier, 'support_vectors_'):
            print(f"Support vectors: {classifier.support_vectors_.shape[0]}")
            print(f"Feature vector length: {classifier.support_vectors_.shape[1]}")
            
            # Check for NaN or Inf values
            if np.isnan(classifier.support_vectors_).any():
                print("Warning: Support vectors contain NaN values")
            if np.isinf(classifier.support_vectors_).any():
                print("Warning: Support vectors contain Inf values")
        else:
            print("Note: Classifier does not have support vectors (not an SVM?)")
        
        if hasattr(classifier, 'n_support_'):
            print("\n=== SUPPORT VECTORS PER CLASS ===")
            for i, count in enumerate(classifier.n_support_):
                if i < len(person_names):
                    print(f"{person_names[i]}: {count} support vectors")
                else:
                    print(f"Class {i}: {count} support vectors")
        
        # Check if the model has probability estimation enabled
        if hasattr(classifier, 'probability') and classifier.probability:
            print("\nModel supports probability estimation")
        else:
            print("\nWarning: Model does not support probability estimation")
            print("This may affect confidence scores during recognition")
        
        # Try to visualize the model (if it's an SVM)
        try:
            if hasattr(classifier, 'support_vectors_'):
                # Get support vectors
                support_vectors = classifier.support_vectors_
                
                # Get the corresponding labels if available
                if hasattr(classifier, 'support_'):
                    support_labels = classifier.support_
                else:
                    # We don't know the labels
                    support_labels = np.zeros(len(support_vectors))
                
                # Apply PCA to visualize in 2D
                pca = PCA(n_components=2)
                vectors_2d = pca.fit_transform(support_vectors)
                
                # Create a simple visualization
                plt.figure(figsize=(10, 8))
                
                # Plot points
                unique_labels = set(support_labels)
                for label in unique_labels:
                    mask = support_labels == label
                    label_name = person_names[int(label)] if int(label) < len(person_names) else f"Class {int(label)}"
                    plt.scatter(vectors_2d[mask, 0], vectors_2d[mask, 1], label=label_name, alpha=0.7)
                
                plt.legend()
                plt.title('PCA of Face Recognition Support Vectors')
                plt.xlabel('Principal Component 1')
                plt.ylabel('Principal Component 2')
                
                # Save the plot
                plot_path = 'model_visualization.png'
                plt.savefig(plot_path)
                print(f"\nModel visualization saved to {plot_path}")
                plt.close()
        except Exception as e:
            print(f"\nCould not visualize model: {str(e)}")
        
        print("\n=== VERIFICATION RESULT ===")
        print("Model appears to be valid")
        return True
        
    except Exception as e:
        print(f"Error verifying model: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_recognition_pipeline():
    """Test the entire recognition pipeline with a sample image"""
    from utils import detect_faces, align_face, normalize_face, extract_features, recognize_face
    
    # Check if we have a test image
    test_image_path = 'test_face.jpg'
    if not os.path.exists(test_image_path):
        print(f"No test image found at {test_image_path}")
        print("Please provide a test image to verify the recognition pipeline")
        return
    
    print(f"\n=== TESTING RECOGNITION PIPELINE WITH {test_image_path} ===")
    
    try:
        # Load test image
        image = cv2.imread(test_image_path)
        if image is None:
            print(f"Could not load image from {test_image_path}")
            return
        
        print(f"Image loaded: {image.shape}")
        
        # Step 1: Detect faces
        print("\nStep 1: Face Detection")
        faces = detect_faces(image)
        print(f"Detected {len(faces)} faces")
        
        if len(faces) == 0:
            print("No faces detected. Cannot proceed with recognition test.")
            return
        
        # Process the first face
        x, y, w, h = faces[0]
        print(f"Processing face at ({x}, {y}, {w}, {h})")
        
        # Extract face region
        face_img = image[y:y+h, x:x+w]
        
        # Step 2: Align face
        print("\nStep 2: Face Alignment")
        aligned_face = align_face(face_img)
        print(f"Aligned face shape: {aligned_face.shape}")
        
        # Save aligned face for inspection
        cv2.imwrite('test_aligned.jpg', aligned_face)
        print("Aligned face saved to test_aligned.jpg")
        
        # Step 3: Normalize face
        print("\nStep 3: Face Normalization")
        normalized_face = normalize_face(aligned_face)
        print(f"Normalized face shape: {normalized_face.shape}")
        print(f"Normalized face value range: [{normalized_face.min()}, {normalized_face.max()}]")
        
        # Save normalized face for inspection
        cv2.imwrite('test_normalized.jpg', (normalized_face * 255).astype(np.uint8))
        print("Normalized face saved to test_normalized.jpg")
        
        # Step 4: Feature extraction
        print("\nStep 4: Feature Extraction")
        features = extract_features(normalized_face)
        print(f"Extracted feature vector length: {len(features)}")
        
        if len(features) == 0:
            print("Feature extraction failed. Cannot proceed with recognition test.")
            return
        
        # Check feature vector for issues
        if np.isnan(features).any():
            print("Warning: Features contain NaN values")
        if np.isinf(features).any():
            print("Warning: Features contain Inf values")
        
        # Step 5: Recognition
        print("\nStep 5: Recognition")
        # Test with different thresholds
        for threshold in [0.1, 0.3, 0.5, 0.7]:
            name, confidence = recognize_face(normalized_face, confidence_threshold=threshold)
            print(f"Threshold {threshold}: Recognized as {name} with confidence {confidence:.4f}")
        
    except Exception as e:
        print(f"Error testing recognition pipeline: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("Face Recognition Model Verification and Diagnostics")
    print("==================================================")
    
    verify_model()
    test_recognition_pipeline()