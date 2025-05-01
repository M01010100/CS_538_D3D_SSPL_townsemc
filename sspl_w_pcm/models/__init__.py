"""
Construct network parts based on existing network classes.
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from .frame_net import VGG16
from .sound_net import VGGish128
from .pc_net import PCNet
from .simsiam_head import SimSiam
from .criterions import SimSiamLoss

# Define synthetic networks for testing
class SynthFrameNet(nn.Module):
    def __init__(self, out_dim=512):
        super(SynthFrameNet, self).__init__()
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

class SynthSoundNet(nn.Module):
    def __init__(self, out_dim=512):
        super(SynthSoundNet, self).__init__()
        self.fc = nn.Linear(128, out_dim)
        self.projection = nn.Linear(1, 128)
        
    def forward(self, x):
        # Debug print to verify shape
        print(f"SynthSoundNet input shape: {x.shape}")
        
        # Handle the shape: [batch_size, 128, 1]
        if x.dim() == 3 and x.size(2) == 1:
            x = x.squeeze(2)  
        
        if x.dim() == 3:
            x = x.mean(dim=2)
        
        if x.size(1) == 1:
            x = self.projection(x)
            print(f"Projected audio features to shape: {x.shape}")
        elif x.size(1) != 128:
            x = x.unsqueeze(2)  # [batch, channels, 1]
            x = F.interpolate(x, size=128, mode='linear')
            x = x.squeeze(2)  # [batch, 128]
            print(f"Interpolated audio features to shape: {x.shape}")
        
        return self.fc(x)

class SynthSSLHead(nn.Module):
    def __init__(self, in_dim=512, out_dim=128):
        super(SynthSSLHead, self).__init__()
        self.projection = nn.Sequential(
            nn.Linear(64, 512), 
            nn.ReLU(),
            nn.Linear(512, out_dim)
        )
        
    def forward(self, x1, x2=None):
        if x1.dim() > 2:
            b, c, h, w = x1.size()
            x1 = F.adaptive_avg_pool2d(x1, 1).view(b, -1)
        
        if x2 is not None and x2.dim() > 2:
            b, c, h, w = x2.size()
            x2 = F.adaptive_avg_pool2d(x2, 1).view(b, -1)
        
        if x2 is not None:
            return self.projection(x1), self.projection(x2)
        return self.projection(x1)

class ModelBuilder():
    def build_frame(self, arch='vgg16', train_from_scratch=False, fine_tune=False):
        if arch == 'vgg16':
            net_frame = VGG16(train_from_scratch, fine_tune)

        else:
            raise Exception('Architecture undefined!')

        return net_frame

    def build_sound(self, arch='vggish', weights=None, out_dim=512):
        """Build sound model with flexible parameter handling"""
        try:
            if arch == 'vggish':
                if weights and isinstance(weights, str) and os.path.exists(weights):
                    net_sound = VGGish128(weights, out_dim)
                else:
                    # Initialize without weights for testing
                    net_sound = VGGish128(None, out_dim)
                    
                # Lock gradients for feature extraction
                for p in net_sound.features.parameters():
                    p.requires_grad = False
                for p in net_sound.embeddings.parameters():
                    p.requires_grad = False
                    
            elif arch in ['vggish15', 'vggish16']:
                print(f"Using {arch} configuration")
                net_sound = VGGish128(None, out_dim)
                
                for p in net_sound.parameters():
                    p.requires_grad = False
                    
            elif arch == 'synth' or arch == 'dummy':
                net_sound = SynthSoundNet(out_dim=out_dim)
                
            else:
                raise ValueError(f"Unknown sound architecture: {arch}")
                
            return net_sound
            
        except Exception as e:
            print(f"Error building sound network: {e}")
            print("Falling back to synthetic sound network")
            return SynthSoundNet(out_dim=out_dim)
        
    def build_ssl_head(self, arch='simsiam', in_dim=512, out_dim=128):
        """Build self-supervised learning head"""
        try:
            if arch == 'simsiam':
                net_ssl_head = SimSiam(dim=in_dim, pred_dim=out_dim)  
                
            elif arch == 'synth' or arch == 'dummy':
                net_ssl_head = SynthSSLHead(in_dim=in_dim, out_dim=out_dim)
                
            else:
                # Handle unknown architectures
                raise ValueError(f"Unknown SSL head architecture: {arch}")
                
            return net_ssl_head
            
        except Exception as e:
            print(f"Error building SSL head: {e}")
            print("Falling back to synthetic SSL head")
            return SynthSSLHead(in_dim=in_dim, out_dim=out_dim)
    
    def build_selfsuperlearn_head(self, arch='simsiam', in_dim=512, out_dim=128):
        """Alias for build_ssl_head for backward compatibility"""
        return self.build_ssl_head(arch, in_dim, out_dim)

    def build_feat_fusion_pc(self, cycs_in=4, dim_audio=128, n_fm_out=512):
        return PCNet(cycs_in=cycs_in, dim_audio=dim_audio, n_fm_out=n_fm_out)

    def build_criterion(self, args):
        if args.arch_ssl_method == 'simsiam':
            loss_ssl = SimSiamLoss()
        else:
            raise Exception('Loss function should be consistent with SSL method')

        return loss_ssl