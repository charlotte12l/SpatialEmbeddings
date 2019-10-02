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
    compare = False,

    save=True,
    save_dir='/n/pfister_lab2/Lab/xingyu/InstanceSeg/Outputs/SpatialEbd/multiclass0_rs_1',
    #checkpoint_path='/n/pfister_lab2/Lab/xingyu/InstanceSeg/SpatialEmbeddings/src/pretrained_models/cars_pretrained_model.pth',
    checkpoint_path='/n/pfister_lab2/Lab/xingyu/InstanceSeg/Outputs/checkpoints/multiclass0_rs/best_iou_model.pth',
    dataset= {
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
        }
    },
        
    model = {
        'name': 'branched_erfnet',
        'kwargs': {
            'num_classes': [3, 8],
        }
    }
)


def get_args():
    return copy.deepcopy(args)
