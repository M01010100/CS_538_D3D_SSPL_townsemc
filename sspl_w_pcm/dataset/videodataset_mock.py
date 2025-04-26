import os
import csv
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
import xml.etree.ElementTree as ET

class VideoDatasetMock(data.Dataset):
    def __init__(self, root, split='train', img_size=224, audio_len=3, mode='ssl', 
                 data_len=None, test_mode=None, crop_func=None, args=None):
        self.root = root
        self.split = split
        self.img_size = img_size
        self.audio_len = audio_len
        self.mode = mode
        self.test_mode = test_mode
        self.crop_func = crop_func
        self.args = args
        
        # Define paths
        if split == 'train':
            meta_file = os.path.join(root, 'SoundNet_Flickr/5k_labeled/train_data.csv')
        else:
            meta_file = os.path.join(root, 'SoundNet_Flickr/5k_labeled/test_data.csv')
            
        self.frame_path = os.path.join(root, 'SoundNet_Flickr/5k_labeled/Data/frames/')
        self.audio_path = os.path.join(root, 'SoundNet_Flickr/5k_labeled/Data/audio/')
        self.anno_path = os.path.join(root, 'SoundNet_Flickr/5k_labeled/Annotations/')
        
        # Load metadata
        self.metadata = []
        with open(meta_file, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                video_id = row[0]
                self.metadata.append(video_id)
        
        # Limit data length if specified
        if data_len is not None:
            self.metadata = self.metadata[:data_len]
        
        # Define transforms
        if self.split == 'train':
            self.img_transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
        else:
            self.img_transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
    
    def _load_frame(self, video_id):
        try:
            frame_path = os.path.join(self.frame_path, f"{video_id}.jpg")
            frame = Image.open(frame_path).convert('RGB')
            return frame
        except Exception as e:
            print(f"Error loading frame {video_id}, using fallback: {e}")
            # Fallback to random image
            random_img = np.random.randint(0, 256, (self.img_size, self.img_size, 3), dtype=np.uint8)
            return Image.fromarray(random_img)
    
    def _load_audio_features(self, video_id):
        try:
            # For simplicity, just return random features
            # In a real implementation, you would load and process audio
            return torch.randn(128, 1)  # Match VGGish output dimensions
        except:
            return torch.randn(128, 1)
    
    def _load_bbox(self, video_id):
        try:
            anno_path = os.path.join(self.anno_path, f"{video_id}.xml")
            tree = ET.parse(anno_path)
            root = tree.getroot()
            
            # Parse object bounding box
            obj = root.find('object')
            bbox = obj.find('bndbox')
            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)
            
            # Create binary mask (1 inside bounding box, 0 outside)
            mask = np.zeros((self.img_size, self.img_size), dtype=np.float32)
            
            # Scale bounding box to match image size
            x_scale = self.img_size / 224
            y_scale = self.img_size / 224
            
            xmin = int(xmin * x_scale)
            ymin = int(ymin * y_scale)
            xmax = int(xmax * x_scale)
            ymax = int(ymax * y_scale)
            
            mask[ymin:ymax, xmin:xmax] = 1.0
            return mask
            
        except Exception as e:
            print(f"Error loading bbox for {video_id}, using fallback: {e}")
            # Fallback to center box
            mask = np.zeros((self.img_size, self.img_size), dtype=np.float32)
            center_size = self.img_size // 3
            start = (self.img_size - center_size) // 2
            end = start + center_size
            mask[start:end, start:end] = 1.0
            return mask
    
    def __getitem__(self, index):
        # Get video ID
        video_id = self.metadata[index]
        
        # Load frame
        frame = self._load_frame(video_id)
        
        # Apply transformations
        if self.mode == 'ssl':
            # For self-supervised learning, create two views
            frame_view1 = self.img_transform(frame)
            frame_view2 = self.img_transform(frame)  # Apply different augmentation
            
            # Load audio features
            audio_feat = self._load_audio_features(video_id)
            
            # Return batch data
            return {
                'frame_view1': frame_view1,
                'frame_view2': frame_view2,
                'audio_feat': audio_feat,
                'data_id': video_id
            }
        else:
            # For evaluation, include ground truth
            frame_tensor = self.img_transform(frame)
            
            # Load audio features
            audio_feat = self._load_audio_features(video_id)
            
            # Load ground truth mask
            gt_map = self._load_bbox(video_id)
            
            # Return batch data
            return {
                'frame': frame_tensor,
                'audio_feat': audio_feat,
                'gt_map': gt_map,
                'data_id': video_id
            }
    
    def __len__(self):
        return len(self.metadata)