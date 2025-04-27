import os
import json
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset

class VideoDatasetMock(Dataset):
    def __init__(self, root, split='train', img_size=224, mode='ssl', data_len=None):
        """
        Args:
            root (string): Root directory of the dataset.
            split (string): 'train' or 'val'.
            img_size (int): Size to resize images to.
            mode (string): 'ssl' for self-supervised learning or 'sup' for supervised.
            data_len (int): Optional limit on number of data points to use.
        """
        self.root = root
        self.split = split
        self.img_size = img_size
        self.mode = mode
        
        # Load metadata
        metadata_path = os.path.join(root, split, 'metadata.json')
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        # Limit data length if specified
        if data_len is not None:
            self.metadata = self.metadata[:min(data_len, len(self.metadata))]
        
        # Setup transforms
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        print(f"Loaded {len(self.metadata)} samples for {split} split")
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        # Get metadata for this item
        item = self.metadata[idx]
        video_id = item['video_id']
        
        # Load frame
        frame = Image.open(item['frame_path']).convert('RGB')
        frame = self.transform(frame)
        
        # Load audio features and ensure shape is [1, 128]
        audio_feat = np.load(item['audio_feature_path'])
        
        # Check if audio feat is the correct size, pad if necessary
        if audio_feat.shape[0] != 128:
            # Pad or truncate to 128 dimensions
            if audio_feat.shape[0] < 128:
                # Pad with zeros
                padded = np.zeros(128, dtype=np.float32)
                padded[:audio_feat.shape[0]] = audio_feat
                audio_feat = padded
            else:
                # Truncate
                audio_feat = audio_feat[:128]
        
        audio_feat = torch.from_numpy(audio_feat).float().unsqueeze(0)  # [1, 128]
        
        # Create a sample with both the frame and audio features
        sample = {
            'frame': frame,
            'audio_feat': audio_feat,
            'data_id': video_id
        }
        
        return sample