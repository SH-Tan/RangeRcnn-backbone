"""
to generate bird eye view, we have to filter point cloud
first. which means we have to limit coordinates


"""
import numpy as np
from functools import partial
import os
import torch
import sys

if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
  sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2

# POINT_CLOUD_RANGE: [0, -40, -3, 70.4, 40, 1] res: 0.1  800*704  stride: 4 200*176

#[0,69.12] [-39.68,39.68]  bev : 496*432 y*x

res = 0.1
# image size would be 800*704


def load_pc(f):
    b = np.fromfile(f, dtype=np.float32)
    return b.reshape((-1, 4))[:, :3]

a = '/home/tan/Desktop/rangeview/data/kitti/bin/002394.bin'
points = load_pc(a)


def gen_bev_map(pc, y_range=[-40, 40], x_range=[0, 70.4], res=0.1):
    # bc = pc.shape[0]

    b,c,n = pc.shape

    w = int((y_range[1] - y_range[0])/res)
    h = int((x_range[1] - x_range[0])/res)

    bev = []

    for i in range(b):
        point = pc[i].permute(1,0).contiguous()

        point = point.numpy()

        x = point[:,0]
        y = point[:,1]
        z = point[:,2]

        im = np.zeros((h, w, c), dtype=np.float32)

        # filter point cloud
        f_filt = np.logical_and((x>x_range[0]), (x<x_range[1]))
        s_filt = np.logical_and((y>-y_range[1]), (y<-y_range[0]))
        filt = np.logical_and(f_filt, s_filt)
        indices = np.argwhere(filt).flatten()
        x = x[indices]
        y = y[indices]
        z = z[indices]
        point = point[indices]

        
        # print(indices.shape)

        # convert coordinates to 
        x_img = (-y/res).astype(np.int32)
        y_img = (-x/res).astype(np.int32)
        # shifting image, make min pixel is 0,0
        x_img -= int(np.floor(y_range[0]/res))
        y_img += int(np.ceil(x_range[1]/res))

        x_max = int((y_range[1]-y_range[0])/res-1)
        y_max = int((x_range[1]-x_range[0])/res-1)

        x_img = np.clip(a=x_img, a_min=0, a_max=x_max)
        y_img = np.clip(a=y_img, a_min=0, a_max=y_max)

        # crop z to make it not bigger than 255
        height_range = (-3, 1)
        z_c = np.clip(a=z, a_min=height_range[0], a_max=height_range[1])

        '''
        def scale_to_255(a, min, max, dtype=np.uint8):

            return (((a - min) / float(max - min)) * 255).astype(dtype)

        pixel_values = scale_to_255(pixel_values, min=height_range[0], max=height_range[1])
        '''

        # according to width and height generate image
        z_c = z_c.reshape(-1,1)

        im[x_img, y_img] = z_c*point
        
        # im = torch.from_numpy(im).permute(2,1,0).contiguous()

        bev.append(im)


        # cropped_cloud = np.vstack([x, y, z]).transpose() # 转置

    
    bev = np.array(bev, dtype=np.float32)
    # print(bev)
    bev = torch.from_numpy(bev).permute(0,3,2,1).contiguous()
    # print(bev)
    # print(type(bev))

    return im

# im, cropped_cloud = gen_bev_map(points)
# cv2.imshow('rr', im)
# cv2.waitKey(0)

'''
x = cropped_cloud[:,0]
y = cropped_cloud[:,1]
z = cropped_cloud[:,2]
mlab.points3d(x, y, z, z,  mode="point", colormap='spectral')
mlab.show()
'''