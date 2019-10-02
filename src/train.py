"""
Author: Davy Neven
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import os
import shutil
import time
import cv2
import numpy as np

import torchvision
from matplotlib import pyplot as plt
from tqdm import tqdm

import torch
import train_config
from criterions.my_loss import SpatialEmbLoss
from datasets import get_dataset
from models import get_model
from utils.utils import AverageMeter, Cluster, Logger, Visualizer, draw_flow

import torchvision.utils as vutils

from tensorboardX import SummaryWriter
import datetime

torch.backends.cudnn.benchmark = True

args = train_config.get_args()

if args['save']:
    if not os.path.exists(args['save_dir']):
        os.makedirs(args['save_dir'])

if args['display']:
    plt.ion()
else:
    plt.ioff()
    plt.switch_backend("agg")

# set device
device = torch.device("cuda:0" if args['cuda'] else "cpu")

# train dataloader
train_dataset = get_dataset(
    args['train_dataset']['name'], args['train_dataset']['kwargs'])
train_dataset_it = torch.utils.data.DataLoader(
    train_dataset, batch_size=args['train_dataset']['batch_size'], shuffle=True, drop_last=True, num_workers=args['train_dataset']['workers'], pin_memory=True if args['cuda'] else False)


# val dataloader
val_dataset = get_dataset(
    args['val_dataset']['name'], args['val_dataset']['kwargs'])
val_dataset_it = torch.utils.data.DataLoader(
    val_dataset, batch_size=args['val_dataset']['batch_size'], shuffle=True, drop_last=True, num_workers=args['train_dataset']['workers'], pin_memory=True if args['cuda'] else False)


# set model
model = get_model(args['model']['name'], args['model']['kwargs'])
model.init_output(args['loss_opts']['n_sigma'])
model = torch.nn.DataParallel(model).to(device)

# set criterion
criterion = SpatialEmbLoss(**args['loss_opts'])
criterion = torch.nn.DataParallel(criterion).to(device)

# set optimizer
optimizer = torch.optim.Adam(
    model.parameters(), lr=args['lr'], weight_decay=1e-4)

def lambda_(epoch):
    return pow((1-((epoch)/args['n_epochs'])), 0.9)

scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer, lr_lambda=lambda_,)

# clustering
cluster = Cluster()

# Visualizer
visualizer = Visualizer(('image', 'pred', 'sigma', 'seed'))

# Logger
logger = Logger(('train', 'val', 'iou'), 'loss')

# resume
start_epoch = 0
best_iou = 0
if args['resume_path'] is not None and os.path.exists(args['resume_path']):
    print('Resuming model from {}'.format(args['resume_path']))
    state = torch.load(args['resume_path'])
    start_epoch = state['epoch'] + 1
    best_iou = state['best_iou']
    model.load_state_dict(state['model_state_dict'], strict=True)
    optimizer.load_state_dict(state['optim_state_dict'])
    logger.data = state['logger_data']

def train(epoch,writer):

    # define meters
    loss_meter = AverageMeter()

    # put model into training mode
    model.train()

    # set this only when it is finetuning
    # for module in model.modules():
    #     if isinstance(module, torch.nn.modules.BatchNorm1d):
    #         module.eval()
    #     if isinstance(module, torch.nn.modules.BatchNorm2d):
    #         module.eval()
    #     if isinstance(module, torch.nn.modules.BatchNorm3d):
    #         module.eval()

    for param_group in optimizer.param_groups:
        print('learning rate: {}'.format(param_group['lr']))

    for i, sample in enumerate(tqdm(train_dataset_it)):

        im = sample['image']
        instances = sample['instance'].squeeze()
        class_labels = sample['label'].squeeze()

        output = model(im)
        loss = criterion(output, instances, class_labels, **args['loss_w'])
        loss = loss.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #output.detach().cpu()
        #torch.cuda.empty_cache()


        if args['display'] and i % args['display_it'] == 0:
            with torch.no_grad():
                visualizer.display(im[0], 'image')
                
                predictions = cluster.cluster_with_gt(output[0], instances[0], n_sigma=args['loss_opts']['n_sigma'])
                visualizer.display([predictions.cpu(), instances[0].cpu()], 'pred')

                sigma = output[0][2].cpu()
                sigma = (sigma - sigma.min())/(sigma.max() - sigma.min())
                sigma[instances[0] == 0] = 0
                visualizer.display(sigma, 'sigma')

                seed = torch.sigmoid(output[0][3]).cpu()
                visualizer.display(seed, 'seed')

        loss_meter.update(loss.item())

    if args['tensorboard']:
        with torch.no_grad():
            color_map = draw_flow(torch.tanh(output[0][0:2]))
            seed = torch.sigmoid(output[0][3:11]).cpu()
            sigma = output[0][2].cpu()
            sigma = (sigma - sigma.min()) / (sigma.max() - sigma.min())
            sigma[instances[0] == 0] = 0

            #predictions = cluster.cluster_with_gt(output[0], instances[0], n_sigma=args['loss_opts']['n_sigma'])


            color_map = color_map.transpose(2, 0, 1)

            seed_visual = seed.unsqueeze(1)

            seed_show = vutils.make_grid(seed_visual, nrow=8, normalize=True, scale_each=True)

            writer.add_image('Input', im[0], epoch)
            writer.add_image('InstanceGT', instances[0].unsqueeze(0).cpu().numpy(), epoch)
            writer.add_image('ColorMap', color_map, epoch)
            writer.add_image('SeedMap', seed_show, epoch)
            writer.add_image('SigmaMap', sigma.unsqueeze(0).cpu().numpy(), epoch)
            #writer.add_image('Prediction', predictions.unsqueeze(0).cpu().numpy(), epoch)

    return loss_meter.avg

def val(epoch,writer_val):

    # define meters
    loss_meter, iou_meter = AverageMeter(), AverageMeter()

    # put model into eval mode
    model.eval()

    with torch.no_grad():

        for i, sample in enumerate(tqdm(val_dataset_it)):

            im = sample['image']
            instances = sample['instance'].squeeze()
            class_labels = sample['label'].squeeze()

            output = model(im)
            loss = criterion(output, instances, class_labels, **args['loss_w'], iou=True, iou_meter=iou_meter)
            loss = loss.mean()


            if args['display'] and i % args['display_it'] == 0:
                with torch.no_grad():
                    visualizer.display(im[0], 'image')
                
                    predictions = cluster.cluster_with_gt(output[0], instances[0], n_sigma=args['loss_opts']['n_sigma'])
                    visualizer.display([predictions.cpu(), instances[0].cpu()], 'pred')
    
                    sigma = output[0][2].cpu()
                    sigma = (sigma - sigma.min())/(sigma.max() - sigma.min())
                    sigma[instances[0] == 0] = 0
                    visualizer.display(sigma, 'sigma')
    
                    seed = torch.sigmoid(output[0][3]).cpu()
                    visualizer.display(seed, 'seed')

            loss_meter.update(loss.item())

        if args['tensorboard']:
            with torch.no_grad():
                color_map = draw_flow(torch.tanh(output[0][0:2]))
                seed = torch.sigmoid(output[0][3:11]).cpu()
                sigma = output[0][2].cpu()
                sigma = (sigma - sigma.min()) / (sigma.max() - sigma.min())
                sigma[instances[0] == 0] = 0

                #predictions = cluster.cluster_with_gt(output[0], instances[0], n_sigma=args['loss_opts']['n_sigma'])

                color_map = color_map.transpose(2, 0, 1)

                seed_visual = seed.unsqueeze(1)

                seed_show = vutils.make_grid(seed_visual, nrow=8, normalize=True, scale_each=True)

                writer_val.add_image('Input', im[0], epoch)
                writer_val.add_image('InstanceGT', instances[0].unsqueeze(0).cpu().numpy(), epoch)
                writer_val.add_image('ColorMap', color_map, epoch)
                writer_val.add_image('SeedMap', seed_show, epoch)
                writer_val.add_image('SigmaMap', sigma.unsqueeze(0).cpu().numpy(), epoch)
                #writer_val.add_image('Prediction', predictions.unsqueeze(0).cpu().numpy(), epoch)

    return loss_meter.avg, iou_meter.avg

def save_checkpoint(state, is_best, name='checkpoint.pth'):
    print('=> saving checkpoint')
    file_name = os.path.join(args['save_dir'], name)
    torch.save(state, file_name)
    if is_best:
        shutil.copyfile(file_name, os.path.join(
            args['save_dir'], 'best_iou_model.pth'))

def get_writer(base):

    date = str(datetime.datetime.now()).split(' ')[0]
    time = str(datetime.datetime.now()).split(' ')[1].split('.')[0]
    log_name = 'W'+base+date+'_'+time

    # tensorboardX
    writer = SummaryWriter('/n/pfister_lab2/Lab/xingyu/InstanceSeg/Outputs/runs/'+log_name)
    return writer

writer = get_writer('train')
writer_val = get_writer('val')
for epoch in range(start_epoch, args['n_epochs']):

    print('Starting epoch {}'.format(epoch))
    scheduler.step(epoch)

    train_loss = train(epoch,writer)
    val_loss, val_iou = val(epoch,writer_val)

    print('===> train loss: {:.2f}'.format(train_loss))
    print('===> val loss: {:.2f}, val iou: {:.2f}'.format(val_loss, val_iou))

    logger.add('train', train_loss)
    logger.add('val', val_loss)
    logger.add('iou', val_iou)
    logger.plot(save=args['save'], save_dir=args['save_dir'])

    writer.add_scalar('train', train_loss, epoch)
    writer.add_scalar('val', val_loss, epoch)
    writer.add_scalar('iou', val_iou, epoch)
    
    is_best = val_iou > best_iou
    best_iou = max(val_iou, best_iou)
        
    if args['save']:
        state = {
            'epoch': epoch,
            'best_iou': best_iou, 
            'model_state_dict': model.state_dict(), 
            'optim_state_dict': optimizer.state_dict(),
            'logger_data': logger.data
        }
        save_checkpoint(state, is_best)
