import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset
import cv2

class VideoDatasetMock(Dataset):
    def __init__(self, root='./metadata', split='train', img_size=224, mode='ssl', data_len=None, dataset_type='mock'):
        self.root = root
        self.split = split
        self.img_size = img_size
        self.mode = mode
        self.dataset_type = dataset_type  # 'mock', 'flickr', or 'vggss'
        
        # Different paths based on dataset type
        if dataset_type == 'flickr':
            metadata_path = os.path.join(root, 'flickr_test249', 'metadata.json')
        elif dataset_type == 'vggss':
            metadata_path = os.path.join(root, 'vggss_test_4692', 'metadata.json')
        else:  # mock dataset
            metadata_path = os.path.join(root, split, 'metadata.json')
        
        # Check if metadata file exists, if not create a mock one
        if not os.path.exists(metadata_path):
            if dataset_type != 'mock':
                raise FileNotFoundError(f"Could not find metadata at {metadata_path}. Please make sure the dataset exists.")
            
            print(f"Creating mock metadata for {split}")
            os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
            mock_data = []
            num_samples = 30 if split == 'train' else 10
            for i in range(num_samples):
                mock_data.append({
                    'id': f'{split}_sample_{i:03d}',
                    'frame_path': f'frames/{split}_sample_{i:03d}.jpg',
                    'audio_path': f'audio/{split}_sample_{i:03d}.npy',
                    'width': 640,
                    'height': 480,
                    'bbox': [100, 100, 300, 300]  # [x1, y1, x2, y2]
                })
            with open(metadata_path, 'w') as f:
                json.dump(mock_data, f)
            
        # Load metadata
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
            
        # Limit dataset size if specified
        if data_len is not None and split == 'train':
            self.metadata = self.metadata[:data_len]
            
        print(f"Loaded {len(self.metadata)} samples from {dataset_type} dataset")
        
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        item = self.metadata[idx]
        video_id = item['id']
        
        if self.dataset_type == 'mock':
            # Create synthetic data
            img = np.random.rand(3, self.img_size, self.img_size).astype(np.float32)
            audio_feat = np.random.rand(128).astype(np.float32)
        else:
            # Load real data
            try:
                frame_path = os.path.join(self.root, self.dataset_type, item['frame_path'])
                if os.path.exists(frame_path):
                    # Load and preprocess image
                    img = cv2.imread(frame_path)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (self.img_size, self.img_size))
                    img = np.transpose(img, (2, 0, 1)) / 255.0  # Convert to CHW format and normalize
                    img = img.astype(np.float32)
                else:
                    print(f"Warning: Frame not found at {frame_path}, using synthetic data")
                    img = np.random.rand(3, self.img_size, self.img_size).astype(np.float32)
                
                audio_path = os.path.join(self.root, self.dataset_type, item['audio_path'])
                if os.path.exists(audio_path):
                    # Load audio features
                    audio_feat = np.load(audio_path).astype(np.float32)
                    # Ensure it's the right shape (128)
                    if audio_feat.shape[0] != 128:
                        audio_feat = np.random.rand(128).astype(np.float32)
                else:
                    print(f"Warning: Audio not found at {audio_path}, using synthetic data")
                    audio_feat = np.random.rand(128).astype(np.float32)
            except Exception as e:
                print(f"Error loading data for {video_id}: {e}")
                # Fallback to synthetic data
                img = np.random.rand(3, self.img_size, self.img_size).astype(np.float32)
                audio_feat = np.random.rand(128).astype(np.float32)
        
        # Create bounding box
        if 'bbox' in item:
            box = item['bbox']
            x1, y1, x2, y2 = box
            
            # Normalize box coordinates
            x1 = x1 / item['width'] * self.img_size
            y1 = y1 / item['height'] * self.img_size
            x2 = x2 / item['width'] * self.img_size
            y2 = y2 / item['height'] * self.img_size
        else:
            # Default bounding box if not provided
            x1, y1, x2, y2 = 0.25 * self.img_size, 0.25 * self.img_size, 0.75 * self.img_size, 0.75 * self.img_size
        
        # Prepare return dictionary
        ret_dict = {
            'data_id': video_id
        }
        
        if self.mode == 'ssl':
            # Create two views for SSL
            img_view1 = img.copy()
            img_view2 = img.copy() + 0.1 * np.random.randn(*img.shape).astype(np.float32)
            
            ret_dict['frame_view1'] = torch.from_numpy(img_view1)
            ret_dict['frame_view2'] = torch.from_numpy(img_view2)
            ret_dict['audio_feat'] = torch.from_numpy(audio_feat)
            
        else:  # supervised mode
            ret_dict['frame'] = torch.from_numpy(img)
            ret_dict['audio_feat'] = torch.from_numpy(audio_feat)
            ret_dict['bbox'] = torch.tensor([x1, y1, x2, y2])
            
        return ret_dict