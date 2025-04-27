"""
Construct network parts based on existing network classes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Define dummy networks for testing
class DummyFrameNet(nn.Module):
    def __init__(self, out_dim=512):
        super(DummyFrameNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        return x

class DummySoundNet(nn.Module):
    def __init__(self, out_dim=512):
        super(DummySoundNet, self).__init__()
        self.fc = nn.Linear(128, out_dim)
        # Add a projection layer for the case when input features are fewer than 128
        self.projection = nn.Linear(1, 128)
        
    def forward(self, x):
        # Debug print to verify shape
        print(f"DummySoundNet input shape: {x.shape}")
        
        # Handle the shape: [batch_size, 1, 128]
        if x.dim() == 3 and x.size(2) == 128:
            x = x.squeeze(1)  # Remove the second dimension if it's 1
        
        # Handle the shape: [batch_size, 128, 1]
        if x.dim() == 3 and x.size(2) == 1:
            x = x.squeeze(2)  # Remove the last dimension if it's 1
        
        # If we still have a 3D tensor, take the mean along the time dimension
        if x.dim() == 3:
            x = x.mean(dim=2)
        
        # Handle the case when feature dimension is 1 instead of 128
        if x.size(1) == 1:
            x = self.projection(x)
            print(f"Projected audio features to shape: {x.shape}")
        
        # Now x should be [batch_size, 128]
        return self.fc(x)

class DummySSLHead(nn.Module):
    def __init__(self, in_dim=512, out_dim=128):
        super(DummySSLHead, self).__init__()
        self.projection = nn.Sequential(
            nn.Linear(64, 512),
            nn.ReLU(),
            nn.Linear(512, out_dim)
        )
        
    def forward(self, x1, x2=None):
        # For spatial features, first perform global average pooling
        if x1.dim() > 2:
            b, c, h, w = x1.size()
            x1 = F.adaptive_avg_pool2d(x1, 1).view(b, -1)
        
        if x2 is not None and x2.dim() > 2:
            b, c, h, w = x2.size()
            x2 = F.adaptive_avg_pool2d(x2, 1).view(b, -1)
        
        # Simple implementation that just projects the input
        if x2 is not None:
            return self.projection(x1), self.projection(x2)
        return self.projection(x1)

class ModelBuilder():
    # Build frame model
    def build_frame(self, arch='resnet18', train_from_scratch=False, weights=''):
        # Check which architecture to use
        if arch == 'resnet18':
            # This would normally load ResNet-18, but we'll use dummy for now
            print("Using dummy frame network instead of resnet18")
            net = DummyFrameNet()
        elif arch == 'resnet50':
            # This would normally load ResNet-50, but we'll use dummy for now
            print("Using dummy frame network instead of resnet50")
            net = DummyFrameNet()
        elif arch == 'dummy':
            # Provide a dummy network for testing
            net = DummyFrameNet()
        else:
            print(f"Architecture {arch} not recognized, falling back to dummy")
            net = DummyFrameNet()
            
        if len(weights) > 0:
            print(f'Loading weights for frame model: {weights}')
            net.load_state_dict(torch.load(weights))
            
        return net

    # Build sound model
    def build_sound(self, arch='vggish', weights=''):
        if arch == 'vggish':
            # This would normally load VGGish, but we'll use dummy for now
            print("Using dummy sound network instead of vggish")
            net = DummySoundNet()
        elif arch == 'dummy':
            # Provide a dummy network for testing
            net = DummySoundNet()
        else:
            print(f"Architecture {arch} not recognized, falling back to dummy")
            net = DummySoundNet()
            
        if len(weights) > 0:
            print(f'Loading weights for sound model: {weights}')
            net.load_state_dict(torch.load(weights))
            
        return net
            
    # Build SSL head model
    def build_ssl_head(self, arch='simclr'):
        if arch == 'simclr':
            # This would normally load SimCLR head, but we'll use dummy for now
            print("Using dummy SSL head instead of simclr")
            net = DummySSLHead()
        elif arch == 'simsiam':
            # This would normally load SimSiam head, but we'll use dummy for now
            print("Using dummy SSL head instead of simsiam")
            net = DummySSLHead()
        elif arch == 'dummy':
            # Provide a dummy network for testing
            net = DummySSLHead()
        else:
            print(f"Architecture {arch} not recognized, falling back to dummy")
            net = DummySSLHead()
            
        return net
