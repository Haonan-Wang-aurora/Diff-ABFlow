# Flow visualization code used from https://github.com/tomrunia/OpticalFlow_Visualization


# MIT License
#
# Copyright (c) 2018 Tom Runia
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Tom Runia
# Date Created: 2018-08-03

import numpy as np

def make_colorwheel():
    """
    Generates a color wheel for optical flow visualization as presented in:
        Baker et al. "A Database and Evaluation Methodology for Optical Flow" (ICCV, 2007)
        URL: http://vision.middlebury.edu/flow/flowEval-iccv07.pdf

    Code follows the original C++ source code of Daniel Scharstein.
    Code follows the the Matlab source code of Deqing Sun.

    Returns:
        np.ndarray: Color wheel
    """

    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((ncols, 3))
    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255*np.arange(0,RY)/RY)
    col = col+RY
    # YG
    colorwheel[col:col+YG, 0] = 255 - np.floor(255*np.arange(0,YG)/YG)
    colorwheel[col:col+YG, 1] = 255
    col = col+YG
    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.floor(255*np.arange(0,GC)/GC)
    col = col+GC
    # CB
    colorwheel[col:col+CB, 1] = 255 - np.floor(255*np.arange(CB)/CB)
    colorwheel[col:col+CB, 2] = 255
    col = col+CB
    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.floor(255*np.arange(0,BM)/BM)
    col = col+BM
    # MR
    colorwheel[col:col+MR, 2] = 255 - np.floor(255*np.arange(MR)/MR)
    colorwheel[col:col+MR, 0] = 255
    return colorwheel


def flow_uv_to_colors(u, v, convert_to_bgr=False):
    """
    Applies the flow color wheel to (possibly clipped) flow components u and v.

    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun

    Args:
        u (np.ndarray): Input horizontal flow of shape [H,W]
        v (np.ndarray): Input vertical flow of shape [H,W]
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.

    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    flow_image = np.zeros((u.shape[0], u.shape[1], 3), np.uint8)
    colorwheel = make_colorwheel()  # shape [55x3]
    ncols = colorwheel.shape[0]
    rad = np.sqrt(np.square(u) + np.square(v))
    a = np.arctan2(-v, -u)/np.pi
    fk = (a+1) / 2*(ncols-1)
    k0 = np.floor(fk).astype(np.int32)
    k1 = k0 + 1
    k1[k1 == ncols] = 0
    f = fk - k0
    for i in range(colorwheel.shape[1]):
        tmp = colorwheel[:,i]
        col0 = tmp[k0] / 255.0
        col1 = tmp[k1] / 255.0
        col = (1-f)*col0 + f*col1
        idx = (rad <= 1)
        col[idx]  = 1 - rad[idx] * (1-col[idx])
        col[~idx] = col[~idx] * 0.75   # out of range
        # Note the 2-i => BGR instead of RGB
        ch_idx = 2-i if convert_to_bgr else i
        flow_image[:,:,ch_idx] = np.floor(255 * col)
    return flow_image


def flow_to_image(flow_uv, clip_flow=None, convert_to_bgr=False):
    """
    Expects a two dimensional flow image of shape.

    Args:
        flow_uv (np.ndarray): Flow UV image of shape [H,W,2]
        clip_flow (float, optional): Clip maximum of flow values. Defaults to None.
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.

    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    assert flow_uv.ndim == 3, 'input flow must have three dimensions'
    assert flow_uv.shape[2] == 2, 'input flow must have shape [H,W,2]'
    if clip_flow is not None:
        flow_uv = np.clip(flow_uv, 0, clip_flow)
    u = flow_uv[:,:,0]
    v = flow_uv[:,:,1]
    rad = np.sqrt(np.square(u) + np.square(v))
    rad_max = np.max(rad)
    epsilon = 1e-5
    u = u / (rad_max + epsilon)
    v = v / (rad_max + epsilon)
    return flow_uv_to_colors(u, v, convert_to_bgr)

def read_flo(filename):
    with open(filename, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        if 202021.25 != magic:
            print('Magic number incorrect. Invalid .flo file')
        else:
            w = np.fromfile(f, np.int32, count=1)
            h = np.fromfile(f, np.int32, count=1)
            data = np.fromfile(f, np.float32, count=int(2*w*h))
            # Reshape data into 3D array (columns, rows, bands)
            data2D = np.resize(data, (h[0], w[0],2))
            return data2D

def write_flo(filename, flow):
    """
    write optical flow in Middlebury .flo format
    :param flow: optical flow map
    :param filename: optical flow file path to be saved
    :return: None
    """
    f = open(filename, 'wb')
    magic = np.array([202021.25], dtype=np.float32)
    (height, width) = flow.shape[0:2]
    w = np.array([width], dtype=np.int32)
    print(w)
    h = np.array([height], dtype=np.int32)
    print(h)
    magic.tofile(f)
    w.tofile(f)
    h.tofile(f)
    flow.tofile(f)
    f.close()

def visualize_optical_flow(flow, savepath=None, return_image=False, text=None, scaling=None):
    # flow -> numpy array 2 x height x width
    # 2,h,w -> h,w,2
    # flow = flow.transpose(1,2,0)
    flow[np.isinf(flow)]=0
    # Use Hue, Saturation, Value colour model
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=float)

    # The additional **0.5 is a scaling factor
    mag = np.sqrt(flow[...,0]**2+flow[...,1]**2)**0.5

    ang = np.arctan2(flow[...,1], flow[...,0])
    ang[ang<0]+=np.pi*2
    hsv[..., 0] = ang/np.pi/2.0 # Scale from 0..1
    hsv[..., 1] = 1
    if scaling is None:
        hsv[..., 2] = (mag-mag.min())/(mag-mag.min()).max() # Scale from 0..1
    else:
        mag[mag>scaling]=scaling
        hsv[...,2] = mag/scaling
    rgb = colors.hsv_to_rgb(hsv)
    # This all seems like an overkill, but it's just to exactly match the cv2 implementation
    bgr = np.stack([rgb[...,2],rgb[...,1],rgb[...,0]], axis=2)
    plot_with_pyplot = False
    if plot_with_pyplot:
        fig = plt.figure(frameon=False)
        plot = plt.imshow(bgr)
        plot.axes.get_xaxis().set_visible(False)
        plot.axes.get_yaxis().set_visible(False)
    if text is not None:
        plt.text(0, -5, text)

    if savepath is not None:
        if plot_with_pyplot:
            fig.savefig(savepath, bbox_inches='tight', dpi=200)
            plt.close()
        else: #Plot with skimage
            out = bgr*255
            io.imsave(savepath, out.astype('uint8'))
    return bgr, (mag.min(), mag.max())


from skimage.color import hsv2rgb
def flow2rgb(flow_map, max_value):
    # 2 x H x w
    h, w, _ = flow_map.shape
    flow_u, flow_v = flow_map[:, :, 0], flow_map[:, :, 1]
    n = 8
    mag = np.sqrt(np.sum(np.square(flow_map), axis=2))
    angle = np.arctan2(flow_v, flow_u)

    im_h = np.mod(angle / (2 * np.pi) + 1.0, 1.0)
    im_s = np.clip(mag * n / max_value, 0, 1)
    im_v = np.clip(n - im_s, 0, 1)
    im_hsv = np.stack((im_h, im_s, im_v), axis=2)

    im = hsv2rgb(im_hsv)
    return im

