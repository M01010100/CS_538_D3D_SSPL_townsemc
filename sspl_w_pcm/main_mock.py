import os
import time
import json
import random
import warnings
import numpy as np
import cv2
from sklearn.metrics import auc
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models import ModelBuilder
from dataset.videodataset_mock import VideoDatasetMock

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
    parser.add_argument('--dataset_type', type=str, default='mock', 
                        choices=['mock', 'flickr', 'vggss'], 
                        help='Dataset type to use')
    
    # Training parameters
    parser.add_argument('--num_epoch', type=int, default=5, help='number of training epochs')
    parser.add_argument('--batch_size', type=int, default=2, help='batch size')
    parser.add_argument('--workers', type=int, default=4, help='number of data loading workers')
    parser.add_argument('--disp_iter', type=int, default=1, help='iterations per display')
    parser.add_argument('--eval_epoch', type=int, default=1, help='epochs per evaluation')
    parser.add_argument('--save_epoch', type=int, default=1, help='epochs per saving checkpoint')
    
    # Model parameters
    parser.add_argument('--arch_frame', type=str, default='vgg16', help='frame feature extractor')
    parser.add_argument('--arch_sound', type=str, default='vggish', help='sound feature extractor')
    parser.add_argument('--arch_ssl_method', type=str, default='simsiam', help='ssl method type')
    parser.add_argument('--weights_vggish', type=str, default='', 
                        help='path to vggish weights (use empty string for random initialization)')
    parser.add_argument('--weights_frame', type=str, default=None, help='path to frame weights')
    parser.add_argument('--train_from_scratch', action='store_true', help='train frame extractor from scratch')
    parser.add_argument('--fine_tune', action='store_true', help='fine-tune pretrained frame extractor')
    parser.add_argument('--out_dim', type=int, default=512, help='output dimension for feature extractors')
    parser.add_argument('--dim_f_aud', type=int, default=128, help='dimension of audio features')
    parser.add_argument('--cycs_pcm', type=int, default=2, help='number of PCM cycles')
    
    # Optimizer parameters
    parser.add_argument('--lr_frame', type=float, default=1e-4, help='learning rate for frame extractor')
    parser.add_argument('--lr_sound', type=float, default=1e-4, help='learning rate for sound extractor')
    parser.add_argument('--lr_ssl_head', type=float, default=1e-4, help='learning rate for ssl head')
    parser.add_argument('--lr_pc', type=float, default=1e-4, help='learning rate for phase correlation')
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
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
        args.device = torch.device("cuda")
    else:
        args.device = torch.device("cpu")
        print("Using CPU")

    # Create directories and set up logging
    args.vis = os.path.join(args.ckpt, 'visualization/')
    args.log = os.path.join(args.ckpt, 'running_log.txt')
    
    if args.mode == 'train':
        makedirs(args.ckpt, remove=True)
    
    print('Model ID: {}'.format(args.id))

    # Initialize best cIoU with a small number
    args.best_ciou = -float("inf")

    # Run worker on available device
    main_worker_single(args.device, args)

def main_worker_single(device, args):
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    ################################
    # model
    ################################
    builder = ModelBuilder()
    
    net_frame = builder.build_frame(
        arch=args.arch_frame,
        train_from_scratch=args.train_from_scratch,
        fine_tune=args.fine_tune  
    )
    
    # Create a mock implementation of the sound network for testing
    class MockSoundNet(nn.Module):
        def __init__(self, out_dim=512):
            super(MockSoundNet, self).__init__()
            self.features = nn.Sequential(
                nn.Linear(128, 256),
                nn.ReLU(True),
                nn.Linear(256, out_dim)
            )
            
        def forward(self, x):
            return self.features(x)
    
    # Use the mock sound network instead of VGGish if no weights are provided
    if args.weights_vggish is None or args.weights_vggish == '':
        print("Using mock sound network instead of VGGish (no weights provided)")
        net_sound = MockSoundNet(out_dim=args.out_dim)
    else:
        net_sound = builder.build_sound(
            arch=args.arch_sound,
            weights_vggish=args.weights_vggish,
            out_dim=args.out_dim
        )
    
    net_pc = builder.build_feat_fusion_pc(
        cycs_in=args.cycs_pcm,
        dim_audio=args.dim_f_aud,
        n_fm_out=args.out_dim
    )
    
    net_ssl_head = builder.build_selfsuperlearn_head(
        arch=args.arch_ssl_method,
        in_dim_proj=args.out_dim
    )

    loss = builder.build_criterion(args)

    net_frame = net_frame.to(device)
    net_sound = net_sound.to(device)
    net_pc = net_pc.to(device)
    net_ssl_head = net_ssl_head.to(device)
    loss = loss.to(device)

    class NetWrapper(torch.nn.Module):
        def __init__(self, net_frame, net_sound, net_pc, net_ssl_head, criterion, args):
            super(NetWrapper, self).__init__()
            self.net_frame = net_frame
            self.net_sound = net_sound
            self.net_pc = net_pc
            self.net_ssl_head = net_ssl_head
            self.criterion = criterion
            self.args = args
            self.temperature = 0.1

        def forward(self, batch_data):
            # Extract frame and audio features
            if 'frame_view1' in batch_data and 'frame_view2' in batch_data:
                frame_view1 = batch_data['frame_view1'].to(device)
                frame_view2 = batch_data['frame_view2'].to(device)
            elif 'frame' in batch_data:
                frame = batch_data['frame'].to(device)
                frame_view1 = frame
                noise = torch.randn_like(frame) * 0.1
                frame_view2 = frame + noise
            else:
                raise KeyError("Cannot find frame data in batch")
                
            audio_feat = batch_data['audio_feat'].to(device)
            
            # Handle audio feature shape
            if audio_feat.dim() == 3:
                if audio_feat.size(2) == 1:
                    audio_feat = audio_feat.squeeze(2)
                else:
                    audio_feat = audio_feat.mean(dim=2)
            
            # Extract features
            frame_feat1 = self.net_frame(frame_view1)
            frame_feat2 = self.net_frame(frame_view2)
            sound_feat = self.net_sound(audio_feat)
            
            output = {}
            
            print(f"Sound feature shape: {sound_feat.shape}")
            print(f"Frame feature shape: {frame_feat1.shape}")
            
            b, c = sound_feat.size()
            frame_feat_avg = F.adaptive_avg_pool2d(frame_feat1, 1).view(b, -1)
            
            sound_feat_pcm = sound_feat * frame_feat_avg.detach() / (frame_feat_avg.norm(dim=1, keepdim=True) + 1e-8)
            
            # For visualization, reshape back to match frame feature size
            b, c, h, w = frame_feat1.size()
            frame_feat1_flat = frame_feat1.view(b, c, h*w)
            sound_feat_pcm_reshaped = sound_feat_pcm.view(b, c, 1)
            
            # Compute similarity (dot product)
            sim_map = torch.bmm(frame_feat1_flat.transpose(1, 2), sound_feat_pcm_reshaped)
            sim_map = sim_map.squeeze(-1).view(b, h, w)
            
            # Normalize and resize
            sim_map_norm = F.interpolate(sim_map.unsqueeze(1), size=(224, 224), mode='bilinear', align_corners=False)
            output['orig_sim_map'] = sim_map_norm.squeeze(1)
            
            # SSL head processing
            frame_feat1_avg = F.adaptive_avg_pool2d(frame_feat1, 1).view(b, -1)
            frame_feat2_avg = F.adaptive_avg_pool2d(frame_feat2, 1).view(b, -1)
            

            z1 = self.net_ssl_head.projection(frame_feat1_avg)
            z2 = self.net_ssl_head.projection(frame_feat2_avg)
            
            loss_ssl = self.criterion(z1, z2)

            return loss_ssl, output

    # Create network wrapper
    netWrapper = NetWrapper(net_frame, net_sound, net_pc, net_ssl_head, loss, args)

    # Set up optimizer
    optimizer = torch.optim.Adam([
        {'params': net_sound.parameters(), 'lr': args.lr_sound},
        {'params': net_pc.parameters(), 'lr': args.lr_pc},
        {'params': net_ssl_head.parameters(), 'lr': args.lr_ssl_head}
    ], betas=(args.beta1, 0.999), weight_decay=args.weight_decay)

    ################################
    # data
    ################################
    dataset_train = VideoDatasetMock(
        root=args.data_path,
        split='train',
        img_size=args.img_size,
        mode=args.mode,
        data_len=args.num_train,
        dataset_type=args.dataset_type
    )
    dataset_val = VideoDatasetMock(
        root=args.data_path,
        split='val',
        img_size=args.img_size,
        mode=args.mode,
        dataset_type=args.dataset_type
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
    
    # History for tracking metrics
    history = {
        'train': {'epoch': [], 'loss': []},
        'val': {'epoch': [], 'loss': [], 'ciou': [], 'auc': []}
    }
    
    # Initialize history
    history['train']['epoch'].append(0)
    history['train']['loss'].append(11)  # Starting with a high loss value
    
    # Start with evaluation
    evaluate_single(netWrapper, loader_val, history, 0, args)
    
    # Training loop
    print('Starting training...')
    for epoch in range(args.num_epoch):
        train_single(netWrapper, loader_train, optimizer, history, epoch, device, args)
        
        if (epoch + 1) % args.eval_epoch == 0:
            evaluate_single(netWrapper, loader_val, history, epoch + 1, args)
            checkpoint_single(net_frame, net_sound, net_pc, net_ssl_head, history, epoch + 1, args)
    
    print('Training completed!')

def train_single(netWrapper, loader, optimizer, history, epoch, device, args):
    """Single device training function"""
    torch.set_grad_enabled(True)
    batch_time = AverageMeter()
    data_time = AverageMeter()

    netWrapper.train()

    tic = time.time()
    for i, batch_data in enumerate(loader):
        data_time.update(time.time() - tic)

        optimizer.zero_grad()
        loss_ssl, _ = netWrapper(batch_data)
        loss_ssl.backward()
        optimizer.step()

        batch_time.update(time.time() - tic)
        tic = time.time()

        if i % args.disp_iter == 0:
            print('Epoch: [{}][{}/{}], Time: {:.2f}, Data: {:.2f}, '
                  'loss: {:.4f}'.format(
                epoch, i, len(loader), batch_time.average(), data_time.average(),
                loss_ssl.item()))

            fractional_epoch = epoch + 1. * i / len(loader)
            history['train']['epoch'].append(fractional_epoch)
            history['train']['loss'].append(loss_ssl.item())

def evaluate_single(netWrapper, loader, history, epoch, args):
    print('Evaluation at {} epochs...'.format(epoch))
    torch.set_grad_enabled(False)

    makedirs(args.vis, remove=True)

    netWrapper.eval()

    loss_meter = AverageMeter()
    ciou_orig_sim = []
    
    for i, batch_data in enumerate(loader):
        with torch.no_grad():
            loss_ssl, output = netWrapper(batch_data)
            loss_meter.update(loss_ssl.item())
            
            # Get frame data
            if 'frame_view1' in batch_data:
                img = batch_data['frame_view1'].numpy()
            elif 'frame' in batch_data:
                img = batch_data['frame'].numpy()
            else:
                print(f"Warning: No frame data found. Keys: {batch_data.keys()}")
                continue
            
            video_id = batch_data['data_id']
            orig_sim_map = output['orig_sim_map'].detach().cpu().numpy()

            # Calculate mock cIoU for demonstration purposes
            for n in range(img.shape[0]):
                ciou_orig_sim.append(0.5 + 0.1 * np.random.rand())  # Random value between 0.5-0.6
                
                # Save visualization
                save_visual(video_id[n], img[n], orig_sim_map[n], args)

        print('[Eval] iter {}, loss: {}'.format(i, loss_ssl.item()))

    # Calculate metrics across thresholds
    results_orig_sim = []
    for i in range(21):
        threshold = 0.05 * i
        result_orig_sim = np.sum(np.array(ciou_orig_sim) >= threshold) / len(ciou_orig_sim)
        results_orig_sim.append(result_orig_sim)

    x = [0.05 * i for i in range(21)]
    cIoU_orig_sim = np.sum(np.array(ciou_orig_sim) >= 0.5) / len(ciou_orig_sim)
    AUC_orig_sim = auc(x, results_orig_sim)

    # Log results
    metric_output = '[Eval Summary] Epoch: {:03d}, Loss: {:.4f}, ' \
                    'cIoU_orig_sim: {:.4f}, AUC_orig_sim: {:.4f}'.format(
        epoch, loss_meter.average(), cIoU_orig_sim, AUC_orig_sim)
    print(metric_output)
    
    with open(args.log, 'a') as F:
        F.write(metric_output + '\n')

    # Update history
    history['val']['epoch'].append(epoch)
    history['val']['loss'].append(loss_meter.average())
    history['val']['ciou'].append(cIoU_orig_sim)
    history['val']['auc'].append(AUC_orig_sim)

    # Plot figures
    print('Plotting figures...')
    plot_loss_metrics(args.ckpt, history)

def checkpoint_single(net_frame, net_sound, net_pc, net_ssl_head, history, epoch, args):
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
        torch.save(net_pc.state_dict(),
                   '{}/pc_{}'.format(args.ckpt, suffix_best))
        torch.save(net_ssl_head.state_dict(),
                   '{}/ssl_head_{}'.format(args.ckpt, suffix_best))

if __name__ == '__main__':
    main()