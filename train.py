import os
import argparse
import numpy as np
import shutil
import random
import time
from utils import load_dataset, train_classifier

def create_subset(source_dir, target_dir, num_people, min_images):
    """Create a subset with specific number of people"""
    os.makedirs(target_dir, exist_ok=True)
    
    # Find all valid people (with minimum images)
    valid_people = []
    for person in os.listdir(source_dir):
        person_dir = os.path.join(source_dir, person)
        if os.path.isdir(person_dir):
            images = [f for f in os.listdir(person_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
            if len(images) >= min_images:
                valid_people.append(person)
    
    # Check if we have enough valid people
    if len(valid_people) == 0:
        print(f"No people with at least {min_images} images found in {source_dir}")
        return []
        
    # Print how many valid people we found
    print(f"Found {len(valid_people)} people with at least {min_images} images")
    
    # Take a random sample
    selected_people = random.sample(valid_people, min(num_people, len(valid_people)))
    
    # Copy the selected people's directories
    for person in selected_people:
        src_dir = os.path.join(source_dir, person)
        dst_dir = os.path.join(target_dir, person)
        if not os.path.exists(dst_dir):
            shutil.copytree(src_dir, dst_dir)
    
    print(f"Created subset with {len(selected_people)} people in {target_dir}")
    return selected_people

def train_on_dataset(data_dir, min_images=3, model_path='models/face_classifier.pkl', use_augmentation=True):
    """Train on the specified dataset"""
    try:
        # Record start time
        start_time = time.time()
        
        # Make sure the data directory exists
        if not os.path.exists(data_dir):
            print(f"Error: Dataset directory '{data_dir}' does not exist")
            return False
            
        # Check if the directory contains at least some subdirectories
        subdirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
        if len(subdirs) == 0:
            print(f"Error: No person directories found in '{data_dir}'")
            print("The dataset should be organized with one directory per person, containing their face images")
            return False
            
        print(f"Found {len(subdirs)} potential person directories in '{data_dir}'")
        
        # Check how many people have at least min_images
        valid_people_count = 0
        for person_dir in subdirs:
            full_path = os.path.join(data_dir, person_dir)
            images = [f for f in os.listdir(full_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if len(images) >= min_images:
                valid_people_count += 1
        
        print(f"Of these, {valid_people_count} people have at least {min_images} images")
        
        if valid_people_count == 0:
            print(f"Error: No people with at least {min_images} images found")
            return False
            
        # Load dataset and extract features
        print(f"Loading dataset with min_images_per_person={min_images}...")
        face_features, face_labels, person_names = load_dataset(
            data_dir, min_images_per_person=min_images, use_augmentation=use_augmentation)
        
        if len(face_features) == 0:
            print("Error: No valid face features extracted")
            return False
            
        # Train classifier
        model_data = train_classifier(face_features, face_labels, person_names, model_path)
        
        # Record end time
        end_time = time.time()
        training_time = end_time - start_time
        
        # Print summary
        print("\nTraining Summary:")
        print(f"Dataset: {data_dir}")
        print(f"People: {len(model_data['person_names'])} (out of {len(subdirs)} directories)")
        print(f"Face images: {len(face_labels)}")
        print(f"Training time: {training_time:.2f} seconds")
        print(f"Model accuracy: {model_data['metrics']['accuracy']:.4f}")
        print(f"Precision: {model_data['metrics']['precision']:.4f}")
        print(f"Recall: {model_data['metrics']['recall']:.4f}")
        print(f"F1-score: {model_data['metrics']['f1']:.4f}")
        
        return True
    except Exception as e:
        print(f"Error during training: {str(e)}")
        return False

def incremental_training(source_dir, min_images=3, start_people=3, increment=1, max_people=10, use_augmentation=True):
    """Train incrementally on subsets of increasing size"""
    # Create base directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('subsets', exist_ok=True)
    
    # Start with a small subset
    current_people = start_people
    success = False
    
    while current_people <= max_people:
        subset_dir = f"subsets/subset_{current_people}"
        model_path = f"models/classifier_{current_people}.pkl"
        
        print(f"\n=== Training on {current_people} people ===")
        
        # Create subset
        selected_people = create_subset(source_dir, subset_dir, current_people, min_images)
        
        # If no valid people were found, break
        if len(selected_people) == 0:
            print("No valid people with enough images found. Cannot train.")
            break
            
        # If fewer people were selected than requested, adjust current_people
        if len(selected_people) < current_people:
            print(f"Warning: Only {len(selected_people)} people with {min_images}+ images available")
            current_people = len(selected_people)
        
        # Train on subset
        success = train_on_dataset(subset_dir, min_images, model_path, use_augmentation)
        
        if success:
            # Copy to standard model location
            print(f"Copying model to standard location: models/face_classifier.pkl")
            shutil.copy2(model_path, "models/face_classifier.pkl")
            
            # Try a larger subset
            current_people += increment
        else:
            # If failed, try a smaller increment
            if increment > 1:
                increment = max(1, increment // 2)
                print(f"Reducing increment to {increment} people")
            else:
                # We've hit our limit
                print(f"Cannot train on more than {current_people - increment} people with current memory")
                break
    
    print("\n=== Incremental Training Complete ===")
    if success:
        print(f"Successfully trained on up to {current_people - increment} people")
        print(f"Final model: models/face_classifier.pkl")
    else:
        print("Training failed. Check error messages above.")

def main():
    """Main function for parsing arguments and training models"""
    parser = argparse.ArgumentParser(description='Train face recognition model')
    parser.add_argument('--data_dir', type=str, default='lfw',
                        help='Directory containing person subdirectories with face images')
    parser.add_argument('--mode', type=str, choices=['direct', 'incremental'], default='direct',
                        help='Training mode: direct or incremental')
    parser.add_argument('--min_images', type=int, default=3,
                        help='Minimum number of images required per person')
    parser.add_argument('--augmentation', action='store_true',
                        help='Apply data augmentation to increase training samples')
    
    # Incremental training parameters
    parser.add_argument('--start_people', type=int, default=3,
                        help='Number of people to start with for incremental training')
    parser.add_argument('--increment', type=int, default=1,
                        help='Number of people to add in each increment')
    parser.add_argument('--max_people', type=int, default=10,
                        help='Maximum number of people to train on')
    
    args = parser.parse_args()
    
    # Ensure models directory exists
    os.makedirs('models', exist_ok=True)
    
    print(f"Face Recognition Training - Mode: {args.mode}")
    print(f"Dataset: {args.data_dir}")
    print(f"Minimum images per person: {args.min_images}")
    print(f"Augmentation: {'Enabled' if args.augmentation else 'Disabled'}")
    
    if not os.path.exists(args.data_dir):
        print(f"Error: Dataset directory '{args.data_dir}' does not exist")
        print("You can download a dataset using download_models.py")
        return
    
    if args.mode == 'direct':
        print(f"\n=== Direct Training on {args.data_dir} ===")
        train_on_dataset(
            args.data_dir, 
            min_images=args.min_images, 
            use_augmentation=args.augmentation
        )
    else:
        print(f"\n=== Incremental Training ===")
        print(f"Starting with {args.start_people} people")
        print(f"Increment: {args.increment} people")
        print(f"Maximum: {args.max_people} people")
        
        incremental_training(
            args.data_dir,
            min_images=args.min_images,
            start_people=args.start_people,
            increment=args.increment,
            max_people=args.max_people,
            use_augmentation=args.augmentation
        )

if __name__ == "__main__":
    main()