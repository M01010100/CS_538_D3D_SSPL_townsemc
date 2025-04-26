import os
import numpy as np
import requests
from PIL import Image
import subprocess
from scipy.io import wavfile

def main():
    # Create necessary directories
    os.makedirs("metadata/SoundNet_Flickr/5k_labeled/Data/frames/", exist_ok=True)
    os.makedirs("metadata/SoundNet_Flickr/5k_labeled/Data/audio/", exist_ok=True)
    os.makedirs("metadata/SoundNet_Flickr/5k_labeled/Annotations/", exist_ok=True)
    
    # Download sample images
    for i in range(1, 6):
        img_path = f"metadata/SoundNet_Flickr/5k_labeled/Data/frames/sample_{i}.jpg"
        # Download random image if it doesn't exist
        if not os.path.exists(img_path):
            try:
                response = requests.get(f"https://picsum.photos/224/224", stream=True)
                if response.status_code == 200:
                    with open(img_path, 'wb') as f:
                        f.write(response.content)
                    print(f"Created image {img_path}")
                else:
                    # If download fails, create a random image
                    img = Image.fromarray(np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8))
                    img.save(img_path)
                    print(f"Created random image {img_path}")
            except:
                # Fallback to random image generation
                img = Image.fromarray(np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8))
                img.save(img_path)
                print(f"Created random image {img_path}")
    
    # Create sample audio files
    for i in range(1, 6):
        audio_path = f"metadata/SoundNet_Flickr/5k_labeled/Data/audio/sample_{i}.wav"
        if not os.path.exists(audio_path):
            # Generate white noise (3 seconds)
            sample_rate = 16000
            duration = 3
            noise = np.random.uniform(-1, 1, sample_rate * duration)
            wavfile.write(audio_path, sample_rate, noise.astype(np.float32))
            print(f"Created audio {audio_path}")
    
    # Create sample annotations (simple XML files)
    for i in range(1, 6):
        xml_path = f"metadata/SoundNet_Flickr/5k_labeled/Annotations/sample_{i}.xml"
        if not os.path.exists(xml_path):
            # Create a simple XML annotation
            with open(xml_path, 'w') as f:
                f.write(f'''<?xml version="1.0" encoding="utf-8"?>
<annotation>
    <filename>sample_{i}.jpg</filename>
    <size>
        <width>224</width>
        <height>224</height>
    </size>
    <object>
        <name>sound_source</name>
        <bndbox>
            <xmin>{50+i*10}</xmin>
            <ymin>{50+i*10}</ymin>
            <xmax>{150+i*10}</xmax>
            <ymax>{150+i*10}</ymax>
        </bndbox>
    </object>
</annotation>''')
            print(f"Created annotation {xml_path}")
    
    # Create metadata files
    train_meta_path = "metadata/SoundNet_Flickr/5k_labeled/train_data.csv"
    test_meta_path = "metadata/SoundNet_Flickr/5k_labeled/test_data.csv"
    
    with open(train_meta_path, 'w') as f:
        for i in range(1, 4):  # First 3 samples for training
            f.write(f"sample_{i},0,3\n")
    
    with open(test_meta_path, 'w') as f:
        for i in range(4, 6):  # Last 2 samples for testing
            f.write(f"sample_{i},0,3\n")
    
    print("Mock dataset created successfully!")

if __name__ == "__main__":
    main()