# Data loading based on https://github.com/NVIDIA/flownet2-pytorch

import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F

import os
import math
import random
import cv2
from glob import glob

from utils import frame_utils
from utils.augmentor import FlowAugmentor, SparseFlowAugmentor


class Event_FlowDataset(data.Dataset):
    def __init__(self, aug_params=None, sparse=False):
        self.augmentor = None
        self.sparse = sparse
        if aug_params is not None:
            if sparse:
                self.augmentor = SparseFlowAugmentor(**aug_params)
            else:
                self.augmentor = FlowAugmentor(**aug_params)

        self.is_test = False
        self.init_seed = False
        self.flow_list = []
        self.image_list = []
        self.ev_voxel_list = []
        self.extra_info = []

    def __getitem__(self, index):

        if self.is_test:
            img1 = frame_utils.read_gen(self.image_list[index][0])
            img2 = frame_utils.read_gen(self.image_list[index][1])
            img1 = np.array(img1).astype(np.uint8)[..., :3]
            img2 = np.array(img2).astype(np.uint8)[..., :3]
            img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
            img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
            return img1, img2, self.extra_info[index]

        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True

        index = index % len(self.image_list)
        valid = None
        if self.sparse:
            flow, valid = frame_utils.readFlowKITTI(self.flow_list[index])
        else:
            flow = frame_utils.read_gen(self.flow_list[index])

        img1 = frame_utils.read_gen(self.image_list[index][0])
        img2 = frame_utils.read_gen(self.image_list[index][1])

        flow = np.array(flow).astype(np.float32)
        img1 = np.array(img1).astype(np.uint8)
        img2 = np.array(img2).astype(np.uint8)

        event1_voxel_00 = cv2.imread(self.ev_voxel_list[index][0][0], cv2.IMREAD_GRAYSCALE)
        event1_voxel_01 = cv2.imread(self.ev_voxel_list[index][0][1], cv2.IMREAD_GRAYSCALE)
        event1_voxel_02 = cv2.imread(self.ev_voxel_list[index][0][2], cv2.IMREAD_GRAYSCALE)
        event_voxel1 = np.array([event1_voxel_00, event1_voxel_01, event1_voxel_02])
        event_voxel1 = np.swapaxes(event_voxel1, 2, 0)
        event_voxel1 = np.swapaxes(event_voxel1, 0, 1)

        event2_voxel_00 = cv2.imread(self.ev_voxel_list[index][1][0], cv2.IMREAD_GRAYSCALE)
        event2_voxel_01 = cv2.imread(self.ev_voxel_list[index][1][1], cv2.IMREAD_GRAYSCALE)
        event2_voxel_02 = cv2.imread(self.ev_voxel_list[index][1][2], cv2.IMREAD_GRAYSCALE)
        event_voxel2 = np.array([event2_voxel_00, event2_voxel_01, event2_voxel_02])
        event_voxel2 = np.swapaxes(event_voxel2, 2, 0)
        event_voxel2 =np.swapaxes(event_voxel2, 0, 1)

        # grayscale images
        if len(img1.shape) == 2:
            img1 = np.tile(img1[...,None], (1, 1, 3))
            img2 = np.tile(img2[...,None], (1, 1, 3))
        else:
            img1 = img1[..., :3]
            img2 = img2[..., :3]

        if self.augmentor is not None:
            if self.sparse:
                img1, img2, flow, valid = self.augmentor(img1, img2, flow, valid)
                event_voxel1, event_voxel2, _, _ = self.augmentor(event_voxel1, event_voxel2, flow, valid)
            else:
                img1, img2, flow = self.augmentor(img1, img2, flow)
                event_voxel1, event_voxel2, _ = self.augmentor(event_voxel1, event_voxel2, flow)

        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        flow = torch.from_numpy(flow).permute(2, 0, 1).float()
        event_voxel1 = torch.from_numpy(event_voxel1).permute(2, 0, 1).float()
        event_voxel2 = torch.from_numpy(event_voxel2).permute(2, 0, 1).float()

        if valid is not None:
            valid = torch.from_numpy(valid)
        else:
            valid = (flow[0].abs() < 1000) & (flow[1].abs() < 1000)


        return img1, img2, flow, valid.float(), event_voxel1, event_voxel2


    def __rmul__(self, v):
        self.flow_list = v * self.flow_list
        self.image_list = v * self.image_list
        return self
        
    def __len__(self):
        return len(self.image_list)
        

class Event_KITTI(Event_FlowDataset):
    def __init__(self, aug_params=None, split='training', root='/Datasets/Event-KITTI/HS-KITTI'):
        super(Event_KITTI, self).__init__(aug_params, sparse=True)
        if split == 'testing':
            self.is_test = True

        root = os.path.join(root, split)
        images1 = sorted(glob(os.path.join(root, 'image_2/*_10_blured.png')))
        images2 = sorted(glob(os.path.join(root, 'image_2/*_11_blured.png')))

        for img1, img2 in zip(images1, images2):
            frame_id = img1.split('/')[-1]
            self.extra_info += [ [frame_id] ]
            self.image_list += [ [img1, img2] ]

        ev_voxel1_0s = sorted(glob(os.path.join(root, 'image_2_events/event_voxel/*_10_00.png')))
        ev_voxel1_1s = sorted(glob(os.path.join(root, 'image_2_events/event_voxel/*_10_01.png')))
        ev_voxel1_2s = sorted(glob(os.path.join(root, 'image_2_events/event_voxel/*_10_02.png')))
        ev_voxel2_0s = sorted(glob(os.path.join(root, 'image_2_events/event_voxel/*_11_00.png')))
        ev_voxel2_1s = sorted(glob(os.path.join(root, 'image_2_events/event_voxel/*_11_01.png')))
        ev_voxel2_2s = sorted(glob(os.path.join(root, 'image_2_events/event_voxel/*_11_02.png')))

        for ev_voxel1_0, ev_voxel1_1, ev_voxel1_2, ev_voxel2_0, ev_voxel2_1, ev_voxel2_2 in zip(ev_voxel1_0s, ev_voxel1_1s, ev_voxel1_2s, ev_voxel2_0s, ev_voxel2_1s, ev_voxel2_2s):
            self.ev_voxel_list += [[[ev_voxel1_0, ev_voxel1_1, ev_voxel1_2],[ev_voxel2_0, ev_voxel2_1, ev_voxel2_2]]]

        if split == 'training':
            self.flow_list = sorted(glob(os.path.join(root, 'flow_noc/*_10.png')))
            
class Event_DSEC(Event_FlowDataset):
    def __init__(self, aug_params=None, split='training', root='/Datasets/DSEC/HS-DSEC'):
        super(Event_DSEC, self).__init__(aug_params, sparse=True)
        if split == 'training':
            traintxt = "training.txt"
        elif split == 'testing':
            traintxt = "testing.txt"
        trantxtpath = os.path.join(root, traintxt)
        trainfile = np.genfromtxt(trantxtpath, delimiter=' ', dtype=str)
        img1_path_list = trainfile[:, 0]
        img2_path_list = trainfile[:, 1]

        ev1_voxel_path_list = trainfile[:, 2]
        ev2_voxel_path_list = trainfile[:, 3]
        
        flow_path_list = trainfile[:, 4]

        img1_search_list = []
        img2_search_list = []

        ev1_voxel_search_list = []
        ev2_voxel_search_list = []

        flow_search_list = []

        for i in range(len(img1_path_list)):
            img1_search_path = "*"+img1_path_list[i]+"*"
            img1s = sorted(glob(os.path.join(root ,img1_search_path)))
            img1s = img1s[0]
            img1_search_list.append(img1s)

        for i in range(len(img2_path_list)):
            img2_search_path = "*"+img2_path_list[i]+"*"
            img2s = sorted(glob(os.path.join(root ,img2_search_path)))
            img2s = img2s[0]
            img2_search_list.append(img2s)

        for img1 ,img2 in zip(img1_search_list ,img2_search_list):
            self.image_list += [[img1 ,img2]]

        for i in range(len(ev1_voxel_path_list)):
            ev1_voxel_search_path = "*"+ev1_voxel_path_list[i]+"*"
            ev1_voxels = sorted(glob(os.path.join(root ,ev1_voxel_search_path)))
            ev1_voxel_search_list.append(ev1_voxels)

        for i in range(len(ev2_voxel_path_list)):
            ev2_voxel_search_path = "*"+ev2_voxel_path_list[i]+"*"
            ev2_voxels = sorted(glob(os.path.join(root ,ev2_voxel_search_path)))
            ev2_voxel_search_list.append(ev2_voxels)

        for ev1_voxel ,ev2_voxel in zip(ev1_voxel_search_list ,ev2_voxel_search_list):
            self.ev_voxel_list += [[ev1_voxel ,ev2_voxel]]
        
        for i in range(len(flow_path_list)):
            flow_search_path = "*" + flow_path_list[i] + "*"
            flows = sorted(glob(os.path.join(root, flow_search_path)))
            flow_search_list.append(flows[0])
            
        # print(self.image_list)
        self.flow_list = flow_search_list

def fetch_dataloader(args, TRAIN_DS='C+T+K+S+H'):
    """ Create the data loader for the corresponding trainign set """

    if args.stage == 'kitti':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.2, 'max_scale': 0.4, 'do_flip': False}
        train_dataset = Event_KITTI(aug_params, split='training')
        
    elif args.stage == 'dsec':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.2, 'max_scale': 0.4, 'do_flip': False}
        train_dataset = Event_DSEC(aug_params, split='training')

    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, 
        pin_memory=False, shuffle=True, num_workers=4, drop_last=True)

    print('Training with %d image pairs' % len(train_dataset))
    return train_loader

