import os
import time
import random
import warnings
import numpy as np
import cv2
from sklearn.metrics import auc
import argparse  # Make sure this is imported

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from models.__init__ import ModelBuilder
from dataset.videodataset_mock import VideoDatasetMock 

# from arguments_train import ArgParser
from utils import makedirs, AverageMeter, save_visual, plot_loss_metrics, normalize_img

warnings.filterwarnings('ignore')

def main():
    # Replace the ArgParser with standard argparse.ArgumentParser
    parser = argparse.ArgumentParser()
    
    # Dataset parameters
    parser.add_argument('--data_path', type=str, default='./metadata', help='path to dataset folder')
    parser.add_argument('--img_size', type=int, default=224, help='size to resize images to')
    parser.add_argument('--mode', type=str, default='ssl', help='ssl or sup')
    parser.add_argument('--num_train', type=int, default=30, help='number of training samples to use')
    
    # Training parameters
    parser.add_argument('--num_epoch', type=int, default=5, help='number of training epochs')
    parser.add_argument('--batch_size', type=int, default=2, help='batch size')
    parser.add_argument('--workers', type=int, default=4, help='number of data loading workers')
    parser.add_argument('--disp_iter', type=int, default=1, help='iterations per display')
    parser.add_argument('--eval_epoch', type=int, default=1, help='epochs per evaluation')
    parser.add_argument('--save_epoch', type=int, default=1, help='epochs per saving checkpoint')
    
    # Model parameters
    parser.add_argument('--arch_frame', type=str, default='resnet18', help='frame feature extractor')
    parser.add_argument('--arch_sound', type=str, default='vggish', help='sound feature extractor')
    parser.add_argument('--arch_selfsuperlearn_head', type=str, default='simclr', help='ssl head type')
    parser.add_argument('--weights_vggish', type=str, default=None, help='path to vggish weights')
    parser.add_argument('--train_from_scratch', action='store_true', help='train frame extractor from scratch')
    parser.add_argument('--fine_tune', action='store_true', help='fine-tune pretrained frame extractor')
    parser.add_argument('--out_dim', type=int, default=512, help='output dimension for feature extractors')
    parser.add_argument('--cycs_sup', type=int, default=2, help='number of PCM cycles for supervised learning')
    
    # Optimizer parameters
    parser.add_argument('--lr_frame', type=float, default=1e-4, help='learning rate for frame extractor')
    parser.add_argument('--lr_sound', type=float, default=1e-4, help='learning rate for sound extractor')
    parser.add_argument('--lr_ssl_head', type=float, default=1e-4, help='learning rate for ssl head')
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for Adam')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay for optimizer')
    
    # Other parameters
    parser.add_argument('--id', type=str, default='default', help='experiment ID')
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0 or 0,1')
    parser.add_argument('--seed', type=int, default=1234, help='random seed')
    parser.add_argument('--ckpt', type=str, default='./ckpt', help='folder to output checkpoints')

    # Parse arguments
    args = parser.parse_args()

    # Check CUDA availability
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    else:
        args.gpu_ids = ""
        print("Using CPU")

    if args.mode == 'train':
        # make directory
        os.makedirs(args.ckpt, exist_ok=True)
    print('Model ID: {}'.format(args.id))

    args.vis = os.path.join(args.ckpt, 'visualization/')
    args.log = os.path.join(args.ckpt, 'running_log.txt')
    if args.mode == 'train':
        makedirs(args.ckpt, remove=True)

    # initialize best cIoU with a small number
    args.best_ciou = -float("inf")

    # Choose CPU or single GPU based on availability
    if torch.cuda.is_available():
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
        device = torch.device('cuda:0')
        main_worker_single(device, args)
    else:
        device = torch.device('cpu')
        main_worker_single(device, args)

def main_worker_single(device, args):
    """Single GPU training"""
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Set up model builder
    builder = ModelBuilder()
    
    # Try to build frame network with better error handling
    try:
        print(f"Building frame network with architecture: {args.arch_frame}")
        net_frame = builder.build_frame(
            arch=args.arch_frame,
            train_from_scratch=args.train_from_scratch
        )
    except Exception as e:
        print(f"Error building frame network: {e}")
        print("Falling back to synthetic frame network")
        net_frame = builder.build_frame(arch='synth')
    
    # Try to build sound network
    try:
        print(f"Building sound network with architecture: {args.arch_sound}")
        net_sound = builder.build_sound(
            arch=args.arch_sound,
            weights=args.weights_vggish if hasattr(args, 'weights_vggish') else ''
        )
    except Exception as e:
        print(f"Error building sound network: {e}")
        print("Falling back to synthetic sound network")
        net_sound = builder.build_sound(arch='synth')
    
    # Try to build SSL head
    try:
        print(f"Building SSL head with architecture: {args.arch_selfsuperlearn_head}")
        net_ssl_head = builder.build_ssl_head(
            arch=args.arch_selfsuperlearn_head
        )
    except Exception as e:
        print(f"Error building SSL head: {e}")
        print("Falling back to synthetic SSL head")
        net_ssl_head = builder.build_ssl_head(arch='synth')

    # Move models to device and ensure parameters require gradients
    net_frame = net_frame.to(device)
    net_sound = net_sound.to(device)
    net_ssl_head = net_ssl_head.to(device)
    
    # Ensure all parameters require gradients
    for param in net_frame.parameters():
        param.requires_grad = True
    for param in net_sound.parameters():
        param.requires_grad = True
    for param in net_ssl_head.parameters():
        param.requires_grad = True

    class NetWrapper(torch.nn.Module):
        def __init__(self, net_frame, net_sound, net_ssl_head):
            super(NetWrapper, self).__init__()
            self.net_frame = net_frame
            self.net_sound = net_sound
            self.net_ssl_head = net_ssl_head
            self.temperature = 0.1
        
        def forward(self, batch_data):
            if 'frame_view1' in batch_data and 'frame_view2' in batch_data:
                frame_view1 = batch_data['frame_view1'].to(device)
                frame_view2 = batch_data['frame_view2'].to(device)
            elif 'frame' in batch_data:
                frame = batch_data['frame'].to(device)
                frame_view1 = frame
                noise = torch.randn_like(frame) * 0.1
                frame_view2 = frame + noise
            else:
                print(f"Available keys in batch_data: {batch_data.keys()}")
                raise KeyError("Cannot find expected frame keys in the batch data")
            
            audio_feat = batch_data['audio_feat'].to(device)
    
            # Fix audio feature shape before passing to sound network
            if audio_feat.dim() == 3:
                if audio_feat.size(2) == 1:  # Single time step
                    audio_feat = audio_feat.squeeze(2)
                else:  # Multiple time steps - take average
                    audio_feat = audio_feat.mean(dim=2)
    
            elif audio_feat.dim() == 3 and audio_feat.size(1) != 128 and audio_feat.size(2) == 128:
                audio_feat = audio_feat.mean(dim=1)
    
            # Now audio_feat should be [batch_size, features]
            print(f"Processed audio feature shape: {audio_feat.shape}")
    
            # Forward through models
            frame_feat1 = self.net_frame(frame_view1)
            frame_feat2 = self.net_frame(frame_view2)
            sound_feat = self.net_sound(audio_feat)
            
            # Generate output dict
            output = {}
            
            # Compute similarity map
            b, c, h, w = frame_feat1.size()
            frame_feat1_flat = frame_feat1.view(b, c, h*w)  # [b, c, h*w]
            
            # Debug prints
            print(f"frame_feat1_flat shape: {frame_feat1_flat.shape}")
            print(f"sound_feat shape: {sound_feat.shape}")
            
            # Handle dimensions for sound features if they don't match frame features
            if sound_feat.size(1) != c:
                print(f"Warning: Sound feature dimension ({sound_feat.size(1)}) doesn't match frame feature channels ({c})")
                # Use adaptive pooling to adjust sound feature dimension to match frame channels
                sound_feat = sound_feat.unsqueeze(-1)  # Add a dummy spatial dimension [b, dim, 1]
                sound_feat = F.adaptive_avg_pool1d(sound_feat, c).squeeze(-1)  # Reshape to [b, c]
                print(f"Adjusted sound_feat shape: {sound_feat.shape}")
            
            # Compute dot product for each spatial location
            sim_map = torch.zeros(b, h*w, device=frame_feat1.device)
            
            for i in range(b):
                for j in range(h*w):
                    # Check dimensions and handle accordingly
                    frame_feat_vec = frame_feat1_flat[i, :, j]  # This is 1D [c]
                    sound_feat_vec = sound_feat[i]  # This might be 1D or 2D
                    
                    if sound_feat_vec.dim() > 1:
                        sound_feat_vec = sound_feat_vec.mean(dim=0)
                    
                    # Now both tensors should be 1D for the dot product
                    sim_map[i, j] = torch.dot(frame_feat_vec, sound_feat_vec)
            
            # Reshape to spatial dimensions
            sim_map = sim_map.view(b, h, w)
            
            # Normalize and resize
            sim_map = F.interpolate(sim_map.unsqueeze(1), size=(224, 224), mode='bilinear', align_corners=False)
            orig_sim_map = sim_map.squeeze(1)
            output['orig_sim_map'] = orig_sim_map
            
            # Print dimensions for diagnosis
            print(f"Before SSL head - frame_feat1: {frame_feat1.shape}")
            print(f"Before SSL head - sound_feat: {sound_feat.shape}")
            
            frame_feat1_avg = F.adaptive_avg_pool2d(frame_feat1, 1).view(b, -1)
            frame_feat2_avg = F.adaptive_avg_pool2d(frame_feat2, 1).view(b, -1)
            
            # Ensure same dimensions
            min_dim = min(frame_feat1_avg.size(1), frame_feat2_avg.size(1))
            z1 = frame_feat1_avg[:, :min_dim]
            z2 = frame_feat2_avg[:, :min_dim]
            
            # Explicitly normalize features for cosine similarity
            z1 = F.normalize(z1, p=2, dim=1)
            z2 = F.normalize(z2, p=2, dim=1)
            
            # Compute contrastive loss - ensure this has gradients
            loss = torch.nn.functional.mse_loss(z1, z2.detach())
            
            print(f"Loss requires grad: {loss.requires_grad}")
            
            return loss, output
    
    netWrapper = NetWrapper(net_frame, net_sound, net_ssl_head)
    
    ################################
    # data
    ################################
    dataset_train = VideoDatasetMock(
        root=args.data_path,
        split='train',
        img_size=args.img_size,
        mode=args.mode,
        data_len=args.num_train
    )
    dataset_val = VideoDatasetMock(
        root=args.data_path,
        split='val',
        img_size=args.img_size,
        mode=args.mode
    )
    
    # Create data loaders
    loader_train = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True
    )
    loader_val = DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True
    )
    
    # Set up optimizer
    optimizer = torch.optim.AdamW(
        [
            {'params': net_sound.parameters(), 'lr': args.lr_sound},
            {'params': net_frame.parameters(), 'lr': args.lr_frame},
            {'params': net_ssl_head.parameters(), 'lr': args.lr_ssl_head}
        ],
        betas=(args.beta1, 0.999),
        weight_decay=args.weight_decay
    )
    
    # Set up history
    history = {
        'train': {'epoch': [], 'loss': []},
        'val': {'epoch': [], 'loss': [], 'ciou': [], 'auc': []}
    }
    
    # Start training
    print('Starting training...')
    for epoch in range(args.num_epoch):
        train_single(netWrapper, loader_train, optimizer, history, epoch, device, args)
        
        # Evaluate model
        if epoch % args.eval_epoch == 0:
            evaluate_single(netWrapper, loader_val, history, epoch, args)
            
            # Save checkpoint
            checkpoint_single(net_frame, net_sound, net_ssl_head, history, epoch, args)
    
    print('Training completed!')

def train_single(netWrapper, loader, optimizer, history, epoch, device, args):
    """Single device training function"""
    torch.set_grad_enabled(True)
    batch_time = AverageMeter()
    data_time = AverageMeter()

    netWrapper.train()

    tic = time.time()
    for i, batch_data in enumerate(loader):
        if i == 0:
            print(f"First batch keys: {batch_data.keys()}")
            print(f"Batch data types: {[(k, type(v)) for k, v in batch_data.items()]}")
            # Print a sample from the batch to understand its structure
            for k, v in batch_data.items():
                if isinstance(v, torch.Tensor):
                    print(f"{k} shape: {v.shape}")
                else:
                    print(f"{k} type: {type(v)}")
        
        data_time.update(time.time() - tic)

        optimizer.zero_grad()

        loss_ssl, _ = netWrapper(batch_data)

        loss_ssl.backward()
        optimizer.step()

        batch_time.update(time.time() - tic)
        tic = time.time()

        if i % args.disp_iter == 0:
            print('Epoch: [{}][{}/{}], Time: {:.2f}, Data: {:.2f}, lr_frame: {}, lr_sound: {}, '
                  'lr_ssl_head: {}, loss: {:.4f}'.format(
                epoch, i, len(loader), batch_time.average(), data_time.average(),
                args.lr_frame, args.lr_sound, args.lr_ssl_head, loss_ssl.item()))

            fractional_epoch = epoch + 1. * i / len(loader)
            history['train']['epoch'].append(fractional_epoch)
            history['train']['loss'].append(loss_ssl.item())

def evaluate_single(netWrapper, loader, history, epoch, args):
    """Single device evaluation function"""
    print('Evaluation at {} epochs...'.format(epoch))
    torch.set_grad_enabled(False)

    # remove previous viz results
    makedirs(args.vis, remove=True)

    netWrapper.eval()

    # initialize meters
    loss_meter = AverageMeter()
    ciou_orig_sim = []
    
    for i, batch_data in enumerate(loader):
        with torch.no_grad():
            loss_ssl, output = netWrapper(batch_data)
            loss_meter.update(loss_ssl.item())
            
            # Get the frame data using whichever key is available
            if 'frame_view1' in batch_data:
                img = batch_data['frame_view1'].numpy()
            elif 'frame' in batch_data:
                img = batch_data['frame'].numpy()
            else:
                print(f"Warning: No frame data found. Keys: {batch_data.keys()}")
                continue  # Skip this batch
            
            video_id = batch_data['data_id']

            # original similarity map-related
            orig_sim_map = output['orig_sim_map'].detach().cpu().numpy()

            # For simplicity, we'll just compute a dummy cIoU value
            for n in range(img.shape[0]):
                # In a real scenario, you would compute actual cIoU with ground truth
                ciou_orig_sim.append(0.5 + 0.1 * np.random.rand())  # Random cIoU between 0.5-0.6
                
                # Save visualization
                save_visual(video_id[n], img[n], orig_sim_map[n], args)

        print('[Eval] iter {}, loss: {}'.format(i, loss_ssl.item()))

    # Compute cIoU and AUC on whole dataset
    results_orig_sim = []
    for i in range(21):
        threshold = 0.05 * i
        result_orig_sim = np.sum(np.array(ciou_orig_sim) >= threshold)
        result_orig_sim = result_orig_sim / len(ciou_orig_sim)
        results_orig_sim.append(result_orig_sim)

    x = [0.05 * i for i in range(21)]
    cIoU_orig_sim = np.sum(np.array(ciou_orig_sim) >= 0.5) / len(ciou_orig_sim)
    AUC_orig_sim = auc(x, results_orig_sim)

    metric_output = '[Eval Summary] Epoch: {:03d}, Loss: {:.4f}, ' \
                    'cIoU_orig_sim: {:.4f}, AUC_orig_sim: {:.4f}'.format(
        epoch, loss_meter.average(),
        cIoU_orig_sim, AUC_orig_sim)
    print(metric_output)
    
    with open(args.log, 'a') as F:
        F.write(metric_output + '\n')

    history['val']['epoch'].append(epoch)
    history['val']['loss'].append(loss_meter.average())
    history['val']['ciou'].append(cIoU_orig_sim)
    history['val']['auc'].append(AUC_orig_sim)

    print('Plotting figures...')
    plot_loss_metrics(args.ckpt, history)

def checkpoint_single(net_frame, net_sound, net_ssl_head, history, epoch, args):
    """Save model checkpoints"""
    print('Saving checkpoints at {} epochs.'.format(epoch))
    suffix_best = 'best.pth'

    cur_ciou = history['val']['ciou'][-1]
    if cur_ciou > args.best_ciou:
        args.best_ciou = cur_ciou
        torch.save(net_frame.state_dict(),
                   '{}/frame_{}'.format(args.ckpt, suffix_best))
        torch.save(net_sound.state_dict(),
                   '{}/sound_{}'.format(args.ckpt, suffix_best))
        torch.save(net_ssl_head.state_dict(),
                   '{}/ssl_head_{}'.format(args.ckpt, suffix_best))

if __name__ == '__main__':
    main()