"""
Author: Davy Neven
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import os
import time

import torchvision
from matplotlib import pyplot as plt
from tqdm import tqdm

import test_config
import torch
from datasets import get_dataset
from models import get_model
from utils.utils import Cluster, Visualizer

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image

class_ids = (24, 25, 26, 27, 28, 31, 32, 33)

def draw_flow(x):
    m = np.shape(x)[1]
    n = np.shape(x)[2]
    hsv = np.zeros((m, n, 3),dtype=np.uint8)

    hsv[:,:, 1] = 255

    x = x.cpu().numpy()
    #x = cv2.UMat(x.cpu().numpy())
    mag, ang = cv2.cartToPolar(x[0], x[1])
    hsv[:,:, 0] = ang * 180 / np.pi / 2
    hsv[:,:, 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    bgr = 255-bgr
    #cv2.imshow("colored flow", bgr)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    return bgr

def prepare_img(image):
    if isinstance(image, Image.Image):
        return image

    if isinstance(image, torch.Tensor):
        image.squeeze_()
        image = image.numpy()

    if isinstance(image, np.ndarray):
        if image.ndim == 3 and image.shape[0] in {1, 3}:
            image = image.transpose(1, 2, 0)
        return image

torch.backends.cudnn.benchmark = True

args = test_config.get_args()

if args['display']:
    plt.ion()
else:
    plt.ioff()
    plt.switch_backend("agg")

if args['save']:
    if not os.path.exists(args['save_dir']):
        os.makedirs(args['save_dir'])

# set device
device = torch.device("cuda:0" if args['cuda'] else "cpu")

# dataloader
dataset = get_dataset(
    args['dataset']['name'], args['dataset']['kwargs'])
dataset_it = torch.utils.data.DataLoader(
    dataset, batch_size=1, shuffle=False, drop_last=False, num_workers=4, pin_memory=True if args['cuda'] else False)

# load model
model = get_model(args['model']['name'], args['model']['kwargs'])
model = torch.nn.DataParallel(model).to(device)

# load snapshot
if os.path.exists(args['checkpoint_path']):
    state = torch.load(args['checkpoint_path'])
    model.load_state_dict(state['model_state_dict'], strict=True)
else:
    assert(os.path.exists(args['checkpoint_path']), 'checkpoint_path {} does not exist!'.format(args['checkpoint_path']))

model.eval()

# cluster module
cluster = Cluster()

# Visualizer
visualizer = Visualizer(('image', 'pred', 'sigma', 'seed'))

id_n = 0

with torch.no_grad():

    for sample in tqdm(dataset_it):

        id_n = id_n +1
        im = sample['image']
        instances = sample['instance'].squeeze()
        semantic = sample['label'].squeeze()

        output = model(im)

        
        instance_map, predictions = cluster.cluster(output[0], threshold=0.9)
        #print(torch.max(instance_map))


        if args['compare']:
            color_map = draw_flow(torch.tanh(output[0][0:2]))
            seed = torch.sigmoid(output[0][3]).cpu()
            sigma = output[0][2].cpu()
            sigma = (sigma - sigma.min())/(sigma.max() - sigma.min())
            print(sigma.shape)
            sigma[instances == 0] = 0
            print(instances.shape)
            # loop over instances
            pred_mask = torch.zeros([1, 1024, 2048]).byte()
            for id, pred in enumerate(predictions):
                pred_mask += pred['mask'].unsqueeze(0)

            pred_mask = torchvision.transforms.ToPILImage()(pred_mask)

            # [Image][Semantic (gt)][Instance (gt)][Vector map (pred)][Seed map (pred)][Margin map (pred)] x 10
            images_list = [im,instances,semantic,color_map,seed,sigma,pred_mask]
            fig, ax = plt.subplots(ncols=len(images_list),figsize=(40,5),dpi=600)
            for i in range(len(images_list)):
                ax[i].cla()
                ax[i].set_axis_off()
                ax[i].imshow(prepare_img(images_list[i]))
            save_dir = '/n/pfister_lab2/Lab/xingyu/InstanceSeg/Outputs/SpatialEbd/cmp/' + str(id_n) + '.png'
            plt.savefig(save_dir)

        if args['display']:

            visualizer.display(im, 'image', id_n)
                
            visualizer.display([instance_map.cpu(), instances.cpu()], 'pred', id_n)

            sigma = output[0][2].cpu()
            sigma = (sigma - sigma.min())/(sigma.max() - sigma.min())
            sigma[instances == 0] = 0
            visualizer.display(sigma, 'sigma', id_n)

            seed = torch.sigmoid(output[0][3]).cpu()
            visualizer.display(seed, 'seed', id_n)

        if args['save']:

            base, _ = os.path.splitext(os.path.basename(sample['im_name'][0]))

            txt_file = os.path.join(args['save_dir'], base + '.txt')
            with open(txt_file, 'w') as f:

                # loop over instances
                for id, pred in enumerate(predictions):
                    im_name = base + '_{:02d}.png'.format(id)
                    im = torchvision.transforms.ToPILImage()(
                        (pred['mask']).unsqueeze(0))

                    # write image
                    im.save(os.path.join(args['save_dir'], im_name))
                    
                    # write to file
                    #cl = np.unique(instance_map[pred['mask']!=0].cpu().numpy())
                    #assert ((len(cl)==1),print(cl))
                    #cl = int(cl)
                    cl = pred['label']
                    score = pred['score']
                    f.writelines("{} {} {:.02f}\n".format(im_name, cl, score))
                
                #im = torchvision.transforms.ToPILImage()(pred_mask)
                #im.save(os.path.join(args['save_dir'], im_name))
