import os
import time
import random
import warnings
import numpy as np
import cv2
from sklearn.metrics import auc

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from models import ModelBuilder
from dataset.videodataset_mock import VideoDatasetMock  # Use our mock dataset

from arguments_train import ArgParser
from utils import makedirs, AverageMeter, save_visual, plot_loss_metrics, normalize_img

warnings.filterwarnings('ignore')

def main():
    # First create the mock dataset
    from data import main as create_mock_data
    create_mock_data()
    
    # arguments
    parser = ArgParser()
    args = parser.parse_train_arguments()
    
    # Override some arguments for testing
    args.num_gpus = 1
    args.gpu_ids = '0'
    args.batch_size_per_gpu = 2
    args.num_epoch = 2
    args.num_train = 3  # We created 3 training samples
    args.batch_size = args.num_gpus * args.batch_size_per_gpu
    args.weights_vggish = None  # Skip VGGish weights

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
    """Single GPU version without distributed training"""
    random.seed(args.seed)
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
    
    # Use dummy feature extractor if no weights
    if args.weights_vggish is None or not os.path.exists(args.weights_vggish):
        class DummySoundNet(nn.Module):
            def __init__(self, out_dim=512):
                super(DummySoundNet, self).__init__()
                self.fc = nn.Linear(128, out_dim)
            
            def forward(self, x):
                return self.fc(x)
        
        net_sound = DummySoundNet(out_dim=args.out_dim)
        print("Using dummy sound network")
    else:
        net_sound = builder.build_sound(
            arch=args.arch_sound,
            weights_vggish=args.weights_vggish,
            out_dim=args.out_dim
        )
        
    net_pc = builder.build_feat_fusion_pc(
        cycs_in=args.cycs_sup,
        dim_audio=args.out_dim
    )
    net_ssl_head = builder.build_selfsuperlearn_head(
        arch=args.arch_selfsuperlearn_head,
        in_dim_proj=args.out_dim,
    )

    # Move models to device
    net_frame = net_frame.to(device)
    net_sound = net_sound.to(device)
    net_pc = net_pc.to(device)
    net_ssl_head = net_ssl_head.to(device)
    
    # Create wrapper model for training
    class NetWrapper(torch.nn.Module):
        def __init__(self, net_frame, net_sound, net_pc, net_ssl_head):
            super(NetWrapper, self).__init__()
            self.net_frame = net_frame
            self.net_sound = net_sound
            self.net_pc = net_pc
            self.net_ssl_head = net_ssl_head
            self.temperature = 0.1
        
        def forward(self, batch_data):
            # Extract features
            frame_view1 = batch_data['frame_view1'].to(device)
            frame_view2 = batch_data['frame_view2'].to(device)
            audio_feat = batch_data['audio_feat'].to(device)
            
            # Forward through models
            frame_feat1 = self.net_frame(frame_view1)
            frame_feat2 = self.net_frame(frame_view2)
            sound_feat = self.net_sound(audio_feat)
            
            # Generate output dict
            output = {}
            
            # Compute similarity map
            b, c, h, w = frame_feat1.size()
            frame_feat1_flat = frame_feat1.view(b, c, -1)
            sound_feat_exp = sound_feat.unsqueeze(-1)
            
            # Simple similarity calculation
            sim_map = torch.bmm(frame_feat1_flat.transpose(1, 2), sound_feat_exp)
            sim_map = sim_map.view(b, h, w)
            
            # Normalize similarity map
            sim_map = F.interpolate(sim_map.unsqueeze(1), size=(224, 224), mode='bilinear', align_corners=False)
            orig_sim_map = sim_map.squeeze(1)
            output['orig_sim_map'] = orig_sim_map
            
            # Simplified SSL loss (using cosine similarity)
            z1 = self.net_ssl_head(frame_feat1, sound_feat)
            z2 = self.net_ssl_head(frame_feat2, sound_feat)
            
            loss = -torch.mean(F.cosine_similarity(z1, z2.detach(), dim=1))
            
            return loss, output
    
    netWrapper = NetWrapper(net_frame, net_sound, net_pc, net_ssl_head)
    
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
            {'params': net_pc.parameters(), 'lr': args.lr_frame},
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
            checkpoint_single(net_frame, net_sound, net_pc, net_ssl_head, history, epoch, args)
    
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
            
            img = batch_data['frame_view1'].numpy()
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
                   '{}/pcm_{}'.format(args.ckpt, suffix_best))
        torch.save(net_ssl_head.state_dict(),
                   '{}/ssl_head_{}'.format(args.ckpt, suffix_best))

if __name__ == '__main__':
    main()