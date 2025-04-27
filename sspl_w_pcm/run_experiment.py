import os
import subprocess
import sys

def main():
    """Run the experiment with 5 samples and 2 epochs"""
    print("Starting experiment with 5 samples and 2 epochs")
    
    # Get the current directory path to ensure we're using the correct path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_script = os.path.join(current_dir, "data.py")
    main_script = os.path.join(current_dir, "main_mock.py")
    
    # Step 1: Generate the mock data
    print("\n=== Generating mock data ===")
    print(f"Running data generation script at: {data_script}")
    
    # Create data.py if it doesn't exist
    if not os.path.exists(data_script):
        create_data_script(data_script)
    
    subprocess.run([sys.executable, data_script], check=True)
    
    # Step 2: Run the training with simplified parameters
    print("\n=== Starting training ===")
    subprocess.run([
        sys.executable, main_script,
        "--num_train", "5",
        "--num_epoch", "2",
        "--batch_size", "2",
        "--disp_iter", "1",
        "--eval_epoch", "1",
        "--arch_frame", "dummy",
        "--arch_sound", "dummy",
        "--arch_selfsuperlearn_head", "dummy"
    ], check=True)
    
    print("\n=== Experiment completed ===")

def create_data_script(path):
    """Create the data.py file with mock data generation code"""
    with open(path, 'w') as f:
        f.write('''
import os
import numpy as np
from PIL import Image
import json
import shutil

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
    create_mock_data("train", 5)
    
    # Create mock validation data
    create_mock_data("val", 2)
    
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
        
        # Create mock audio features (128-dimensional)
        audio_feature = np.random.randn(128).astype(np.float32)
        audio_path = f"./metadata/{split}/audio_features/{video_id}.npy"
        np.save(audio_path, audio_feature)
        
        # Add to metadata
        metadata.append({
            "video_id": video_id,
            "frame_path": frame_path,
            "audio_feature_path": audio_path
        })
    
    # Save metadata as JSON
    with open(f"./metadata/{split}/metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

if __name__ == "__main__":
    main()
''')

if __name__ == "__main__":
    main()