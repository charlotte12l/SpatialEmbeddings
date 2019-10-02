"""
Author: Davy Neven
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import collections
import os
import threading

import cv2

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

import torch

class_ids = (24, 25, 26, 27, 28, 31, 32, 33)

class AverageMeter(object):

    def __init__(self, num_classes=1):
        self.num_classes = num_classes
        self.reset()
        self.lock = threading.Lock()

    def reset(self):
        self.sum = [0] * self.num_classes
        self.count = [0] * self.num_classes
        self.avg_per_class = [0] * self.num_classes
        self.avg = 0

    def update(self, val, cl=0):
        with self.lock:
            self.sum[cl] += val
            self.count[cl] += 1
            self.avg_per_class = [
                x/y if x > 0 else 0 for x, y in zip(self.sum, self.count)]
            self.avg = sum(self.avg_per_class)/len(self.avg_per_class)

class Visualizer:

    def __init__(self, keys):
        self.wins = {k:None for k in keys}
        
    def display(self, image, key, id_n):

        n_images = len(image) if isinstance(image, (list, tuple)) else 1
    
        if self.wins[key] is None:
            self.wins[key] = plt.subplots(ncols=n_images)
    
        fig, ax = self.wins[key]
        n_axes = len(ax) if isinstance(ax, collections.Iterable) else 1
    
        assert n_images == n_axes
    
        if n_images == 1:
            ax.cla()
            ax.set_axis_off()
            ax.imshow(self.prepare_img(image))
        else:
            for i in range(n_images):
                ax[i].cla()
                ax[i].set_axis_off()
                ax[i].imshow(self.prepare_img(image[i]))

        save_dir = '/n/pfister_lab2/Lab/xingyu/InstanceSeg/Outputs/SpatialEbd/masks/' + str(key) + '_' + str(id_n)+'.png'
        plt.savefig(save_dir)
        #plt.draw()
        
        self.mypause(0.01)

    @staticmethod
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

    @staticmethod
    def mypause(interval):
        backend = plt.rcParams['backend']
        if backend in matplotlib.rcsetup.interactive_bk:
            figManager = matplotlib._pylab_helpers.Gcf.get_active()
            if figManager is not None:
                canvas = figManager.canvas
                if canvas.figure.stale:
                    canvas.draw()
                canvas.start_event_loop(interval)
                return

class Cluster:

    def __init__(self, ):

        xm = torch.linspace(0, 2, 2048).view(1, 1, -1).expand(1, 1024, 2048)
        ym = torch.linspace(0, 1, 1024).view(1, -1, 1).expand(1, 1024, 2048)
        xym = torch.cat((xm, ym), 0)

        self.xym = xym.cuda()

    def cluster_with_gt(self, prediction, instance, n_sigma=1,):

        height, width = prediction.size(1), prediction.size(2)
    
        xym_s = self.xym[:, 0:height, 0:width]  # 2 x h x w
    
        spatial_emb = torch.tanh(prediction[0:2]) + xym_s  # 2 x h x w
        sigma = prediction[2:2+n_sigma]  # n_sigma x h x w
    
        instance_map = torch.zeros(height, width).byte().cuda()
    
        unique_instances = instance.unique()
        unique_instances = unique_instances[unique_instances != 0]
    
        for id in unique_instances:
    
            mask = instance.eq(id).view(1, height, width)
    
            center = spatial_emb[mask.expand_as(spatial_emb)].view(
                2, -1).mean(1).view(2, 1, 1)  # 2 x 1 x 1
    
            s = sigma[mask.expand_as(sigma)].view(n_sigma, -1).mean(1).view(n_sigma, 1, 1)
            s = torch.exp(s*10)  # n_sigma x 1 x 1
    
            dist = torch.exp(-1 * torch.sum(torch.pow(spatial_emb - center, 2)*s, 0))
    
            proposal = (dist > 0.5)
            instance_map[proposal] = id
    
        return instance_map

    def cluster(self, prediction, n_sigma=1, threshold=0.5):

        height, width = prediction.size(1), prediction.size(2)
        xym_s = self.xym[:, 0:height, 0:width]
        
        spatial_emb = torch.tanh(prediction[0:2]) + xym_s  # 2 x h x w
        sigma = prediction[2:2+n_sigma]  # n_sigma x h x w
        seed_map = torch.sigmoid(prediction[2+n_sigma:2+n_sigma + 8])  # 1 x h x w
       
        instance_map = torch.zeros(height, width).byte()
        instances = []

        count = 1
        #instance_mask_all = torch.zeros(height, width).byte()
        for i in range(8):
            label = class_ids[i]
                #print(seed_map.shape) # 8 1024 2048
            one_seed_map = seed_map[i]
            mask = one_seed_map > 0.5
            #print((one_seed_map>0.5).sum())# 0
            #print((instance_map==0).sum())# 2097152
            #print(mask.shape)# 1024 2048
            #print(mask.sum())# 0
            #print(i)
            #print(torch.max(one_seed_map).item())

            if mask.sum() > 128:
                #print('mask>128')
                spatial_emb_masked = spatial_emb[mask.expand_as(spatial_emb)].view(2, -1)
                sigma_masked = sigma[mask.expand_as(sigma)].view(n_sigma, -1)
                seed_map_masked = one_seed_map[mask].view(1, -1)
                #print(torch.max(seed_map_masked).item())

                unclustered = torch.ones(mask.sum()).byte().cuda()
                instance_map_masked = torch.zeros(mask.sum()).byte().cuda()

                while(unclustered.sum() > 128):

                    seed = (seed_map_masked * unclustered.float()).argmax().item()
                    seed_score = (seed_map_masked * unclustered.float()).max().item()
                    #print(seed_score)
                    if seed_score < threshold:
                        break
                    center = spatial_emb_masked[:, seed:seed+1]
                    unclustered[seed] = 0
                    s = torch.exp(sigma_masked[:, seed:seed+1]*10)
                    dist = torch.exp(-1*torch.sum(torch.pow(spatial_emb_masked -
                                                            center, 2)*s, 0, keepdim=True))

                    proposal = (dist > 0.5).squeeze()

                    if proposal.sum() > 128:
                        if unclustered[proposal].sum().float()/proposal.sum().float() > 0.5:
                            instance_map_masked[proposal.squeeze()] = count # can return a instance map and class map
                            instance_mask = torch.zeros(height, width).byte()
                            instance_mask[mask.squeeze().cpu()] = proposal.cpu().byte()
                            #instance_mask_all += instance_mask
                            #print(np.unique(instance_mask.cpu().numpy()))
                            instances.append(
                                {'mask': instance_mask.squeeze()*255, 'score': seed_score, 'label': label})
                            count += 1
                            #print(count)

                    unclustered[proposal] = 0

                instance_map[mask.squeeze().cpu()] = instance_map_masked.cpu()

        return instance_map, instances

class Logger:

    def __init__(self, keys, title=""):

        self.data = {k: [] for k in keys}
        self.title = title
        self.win = None

        print('created logger with keys:  {}'.format(keys))

    def plot(self, save=False, save_dir=""):

        if self.win is None:
            self.win = plt.subplots()
        fig, ax = self.win
        ax.cla()

        keys = []
        for key in self.data:
            keys.append(key)
            data = self.data[key]
            ax.plot(range(len(data)), data, marker='.')

        ax.legend(keys, loc='upper right')
        ax.set_title(self.title)

        plt.draw()
        Visualizer.mypause(0.001)

        if save:
            # save figure
            fig.savefig(os.path.join(save_dir, self.title + '.png'))

            # save data as csv
            df = pd.DataFrame.from_dict(self.data)
            df.to_csv(os.path.join(save_dir, self.title + '.csv'))

    def add(self, key, value):
        assert key in self.data, "Key not in data"
        self.data[key].append(value)

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