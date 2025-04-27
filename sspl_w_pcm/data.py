import os
import numpy as np
import torch
import json
import cv2
from PIL import Image

def main():
    """Create mock data for testing the pipeline"""
    print("Creating mock data...")
    
    # Create directories
    os.makedirs("./metadata/train", exist_ok=True)
    os.makedirs("./metadata/val", exist_ok=True)
    os.makedirs("./metadata/train/frames", exist_ok=True)
    os.makedirs("./metadata/val/frames", exist_ok=True)
    os.makedirs("./metadata/train/audio_features", exist_ok=True)
    os.makedirs("./metadata/val/audio_features", exist_ok=True)
    
    # Create mock training data
    create_mock_data("train", 3) # Create 3 training samples
    
    # Create mock validation data
    create_mock_data("val", 2)    # Create 2 validation samples
    
    print("Mock data creation complete!")

def create_mock_data(split, num_samples):
    """Create mock data for a specific split"""
    # Create metadata file
    metadata = []
    
    for i in range(num_samples):
        # Create a unique video ID
        video_id = f"mock_{split}_{i:04d}"
        
        # Create a random frame (224x224x3)
        frame = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        frame_path = f"./metadata/{split}/frames/{video_id}.jpg"
        
        # Save as image
        img = Image.fromarray(frame)
        img.save(frame_path)
        
        # Create two different views of the same frame for contrastive learning
        frame_view1 = frame.copy()
        frame_view2 = frame.copy()
        
        # Add some noise to second view (different augmentation)
        frame_view2 = np.clip(frame_view2 + np.random.normal(0, 25, frame_view2.shape), 0, 255).astype(np.uint8)
        
        frame_view1_path = f"./metadata/{split}/frames/{video_id}_view1.jpg"
        frame_view2_path = f"./metadata/{split}/frames/{video_id}_view2.jpg"
        
        # Save both views
        img_view1 = Image.fromarray(frame_view1)
        img_view2 = Image.fromarray(frame_view2)
        img_view1.save(frame_view1_path)
        img_view2.save(frame_view2_path)
        
        # Create mock audio features (128-dimensional)
        # Ensure all audio features have the same dimensions
        audio_feature = np.random.randn(128).astype(np.float32)
        audio_path = f"./metadata/{split}/audio_features/{video_id}.npy"
        np.save(audio_path, audio_feature)
        
        # Add to metadata
        metadata.append({
            "video_id": video_id,
            "frame_path": frame_path,
            "frame_view1_path": frame_view1_path,
            "frame_view2_path": frame_view2_path,
            "audio_feature_path": audio_path
        })
    
    # Save metadata as JSON
    with open(f"./metadata/{split}/metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

if __name__ == "__main__":
    main()