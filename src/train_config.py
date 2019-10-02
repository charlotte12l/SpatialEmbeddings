"""
Author: Davy Neven
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import copy
import os

from PIL import Image

import torch
from utils import transforms as my_transforms

CITYSCAPES_DIR=os.environ.get('CITYSCAPES_DIR')

args = dict(

    cuda=True,
    display=False,
    display_it=10,
    tensorboard=True,

    save=True,
    save_dir='/n/pfister_lab2/Lab/xingyu/InstanceSeg/Outputs/checkpoints/Wmulti',
    #resume_path='/n/pfister_lab2/Lab/xingyu/InstanceSeg/Outputs/checkpoints/Wmulti/best_iou_model.pth',
    resume_path = None,
#####please remember the fix layers
    train_dataset = {
        'name': 'cityscapes',
        'kwargs': {
            'root_dir': CITYSCAPES_DIR,
            'type': 'crops',
            'size': 3000,
            'transform': my_transforms.get_transform([
                {
                    'name': 'RandomCrop',
                    'opts': {
                        'keys': ('image', 'instance','label'),
                        'size': (512, 512),
                    }
                },
                {
                    'name': 'ToTensor',
                    'opts': {
                        'keys': ('image', 'instance', 'label'),
                        'type': (torch.FloatTensor, torch.ByteTensor, torch.ByteTensor),
                    }
                },
            ]),
        },
        'batch_size': 12,
        'workers': 8
    }, 

    val_dataset = {
        'name': 'cityscapes',
        'kwargs': {
            'root_dir': CITYSCAPES_DIR,
            'type': 'val',
            'transform': my_transforms.get_transform([
                {
                    'name': 'ToTensor',
                    'opts': {
                        'keys': ('image', 'instance', 'label'),
                        'type': (torch.FloatTensor, torch.ByteTensor, torch.ByteTensor),
                    }
                },
            ]),
        },
        'batch_size': 16,
        'workers': 8
    }, 

    model = {
        'name': 'branched_erfnet', 
        'kwargs': {
            'num_classes': [3,8]
        }
    }, 

    lr=5e-4,
    n_epochs=200,

    # loss options
    loss_opts={
        'to_center': True,
        'n_sigma': 1,
        'foreground_weight': 10,
    },
    loss_w={
        'w_inst': 1,
        'w_var': 10,
        'w_seed': 1,
    },
)


def get_args():
    return copy.deepcopy(args)
