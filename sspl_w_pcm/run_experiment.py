import os
import subprocess
import sys
import datetime

def main():
    print("Starting experiment with 5 samples and 2 epochs")
    
    # Create timestamped output directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"./experiments/run_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output will be saved to: {output_dir}")
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_script = os.path.join(current_dir, "data.py")
    main_script = os.path.join(current_dir, "main_mock.py")
    
    # Create log file in output directory
    log_file = os.path.join(output_dir, "experiment_log.txt")
    with open(log_file, "w") as f:
        f.write(f"Experiment started at {timestamp}\n")
        f.write(f"Output directory: {output_dir}\n\n")
    
    print("\n=== Generating mock data ===")
    print(f"Running data generation script at: {data_script}")
    
    if not os.path.exists(data_script):
        create_data_script(data_script)
    
    # Run data generation and capture output
    result = subprocess.run(
        [sys.executable, data_script], 
        check=True,
        capture_output=True,
        text=True
    )
    
    # Save data generation output to log
    with open(log_file, "a") as f:
        f.write("=== Data Generation Output ===\n")
        f.write(result.stdout)
        f.write("\n\n")
    
    print("\n=== Starting training ===")
    # Run training with output directory specified
    result = subprocess.run([
        sys.executable, main_script,
        "--num_train", "5",
        "--num_epoch", "4",
        "--batch_size", "2",
        "--disp_iter", "1",
        "--eval_epoch", "4",
        "--arch_frame", "synth",
        "--arch_sound", "synth",
        "--arch_selfsuperlearn_head", "synth",
        "--ckpt", "./sspl_w_pcm_flickr10k",
    ], check=True, capture_output=True, text=True)
    
    # Save training output to log
    with open(log_file, "a") as f:
        f.write("=== Training Output ===\n")
        f.write(result.stdout)
        if result.stderr:
            f.write("\n=== Training Errors ===\n")
            f.write(result.stderr)
    
    print(f"\n=== Experiment completed and saved to {output_dir} ===")

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
    create_mock_data("train", 6)
    
    # Create mock validation data
    create_mock_data("val", 4)
    
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