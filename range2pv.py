from laserscan import  LaserScan, SemLaserScan
import numpy as np
import os
from rangenet import RangeNet
import torch
import torch.nn as nn
from collections import OrderedDict
import yaml
from easydict import EasyDict
from pathlib import Path


def merge_new_config(config, new_config):
    if '_BASE_CONFIG_' in new_config:
        with open(new_config['_BASE_CONFIG_'], 'r') as f:
            try:
                yaml_config = yaml.load(f, Loader=yaml.FullLoader)
            except:
                yaml_config = yaml.load(f)
        config.update(EasyDict(yaml_config))

    for key, val in new_config.items():
        if not isinstance(val, dict):
            config[key] = val
            continue
        if key not in config:
            config[key] = EasyDict()
        merge_new_config(config[key], val)

    return config

def cfg_from_yaml_file(cfg_file, config):
    with open(cfg_file, 'r') as f:
        try:
            new_config = yaml.load(f, Loader=yaml.FullLoader)
        except:
            new_config = yaml.load(f)

        merge_new_config(config=config, new_config=new_config)

    return config


cfg = EasyDict()
cfg.ROOT_DIR = (Path(__file__).resolve().parent / '../').resolve()
cfg.LOCAL_RANK = 0

cfg = cfg_from_yaml_file('modelcfg.yaml', cfg)



def load_data_to_gpu(batch_dict):
    for key, val in batch_dict.items():
        if not isinstance(val, np.ndarray):
            continue
        elif key in ['frame_id', 'metadata', 'calib']:
            continue
        elif key in ['range_in', 'x', 'y']:
            continue
        elif key in ['image_shape']:
            batch_dict[key] = torch.from_numpy(val).int().cuda()
        else:
            batch_dict[key] = torch.from_numpy(val).float().cuda()






rangeset = LaserScan(project=True)

# for f in os.listdir('./data/kitti/bin/'):
    # print(f)

for i in range (1):

    f = '002394.bin'

    out_dict = rangeset.open_scan('./data/kitti/bin/' + f, f)

    # points N * 3
    # scan N * 4
    # re N * 1

    # load_data_to_gpu(out_dict)

    x = out_dict['x']
    y = out_dict['y']

    data = out_dict['range_in'].permute(2,0,1).contiguous()

    poi_cor = out_dict['ori_xyz']
    ord_p = out_dict['points']
    p_cuda = torch.from_numpy(ord_p).float()
    p_cuda = p_cuda.unsqueeze(0).permute(0,2,1)
    p_cuda = p_cuda.contiguous()
    print(poi_cor[y,x,:].shape)

    # poi_cor = out_dict['ori_xyz'].permute(2,0,1).contiguous()

    data = data.unsqueeze(0) # 1*5*64*1024 N*C*H*W
    # poi_cor = poi_cor.unsqueeze(0).cpu().numpy() # 1*3*64*1024


    # print(poi_cor[y,x].shape)

    net = RangeNet(data, cfg)

    down, up = net(data)  # out 1*64*64*1024 N*C*H*W

    output = up[len(up)-1]

    res = torch.cat((p_cuda,output[:,:,y,x]),1)

    print(res.shape)

    # print(output[:,:,y,x].shape)




