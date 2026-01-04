import numpy as np
from core.utils import flow_viz
import matplotlib.pyplot as plt
import imageio
import os
from PIL import Image

if __name__ == '__main__':
    flo_dir = 'inference_results/Night_kitti/'
    vis_dir =  'vis_results/Night_kitti/'
    
    flow_dir = flo_dir + 'infer/'
    gt_dir =  flo_dir + 'GT/'
    
    flow_vis_dir = vis_dir + 'infer/'
    gt_vis_dir =  vis_dir + 'GT/'
    os.makedirs(flow_vis_dir, exist_ok=True)
    os.makedirs(gt_vis_dir, exist_ok=True)
    
    flow_list = sorted(os.listdir(flow_dir))
    gt_list = sorted(os.listdir(gt_dir))
    
    for i in range(len(gt_list)):
        filename = f"{i:06d}.png"
        print('------------------------------------------------------------')
        print(filename)
        
        flow = flow_viz.read_flo(flow_dir+flow_list[i])
        gt = flow_viz.read_flo(gt_dir+gt_list[i])
        
        flow_img = flow_viz.flow_to_image(flow)
        gt_img = flow_viz.flow_to_image(gt)
        
        flow_img = Image.fromarray(flow_img, 'RGB')
        gt_img = Image.fromarray(gt_img, 'RGB')
        
        flow_img.save(flow_vis_dir+filename)
        gt_img.save(gt_vis_dir+filename)
