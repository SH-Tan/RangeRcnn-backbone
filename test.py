
import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict
from rangenet import RangeNet,DRB
import yaml
from easydict import EasyDict
from pathlib import Path
import random
from pv2bev import gen_bev_map

random.seed(10)

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

# print(cfg)


class downsample(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()

        self.c1 = nn.Conv2d(cin, cout, kernel_size=(1,1))
        self.drop = nn.Dropout2d(p=0.2)
        self.pooling = nn.MaxPool2d((2,2), stride=2)
        self.bn = nn.BatchNorm2d(cout, eps=1e-3, momentum=0.01)
        self.drb = DRB(cout, cout)
        self.relu = nn.ReLU()

    def forward(self, in_f):
        i_drb = self.c1(in_f)
        # i_drb_bn = self.bn(i_drb)
        o_drb = self.drb(i_drb)
        o_drop = self.drop(o_drb)
        out = self.pooling(o_drop)
        out = self.relu(out)

        return out




class upsample(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.drb = DRB(2*cin, cout)
        self.bn = nn.BatchNorm2d(cout, eps=1e-3, momentum=0.01)
        self.relu = nn.ReLU()

    def forward(self, in_f, down_con):
        up_in = nn.functional.interpolate(in_f, scale_factor=(2,2), mode='bilinear', align_corners=True)
        drb_in = torch.cat((up_in,down_con), 1)
        drb_out = self.drb(drb_in)

        return drb_out



dil_c1 = nn.Sequential(OrderedDict([
            ('c1', nn.Conv2d(5, 32, kernel_size=(3, 3), stride=1, bias=False, dilation=1, padding=1)),
            ('bn1', nn.BatchNorm2d(32, eps=1e-3, momentum=0.01)),
            ('relu1', nn.ReLU())
        ]))

dil_c2 = nn.Sequential(OrderedDict([
            ('c2', nn.Conv2d(32, 32, kernel_size=(3, 3), stride=1, bias=False, dilation=3, padding=3)),
            ('bn2', nn.BatchNorm2d(32, eps=1e-3, momentum=0.01)),
            ('relu2', nn.ReLU())
        ]))

dil_c3 = nn.Sequential(OrderedDict([
            ('c3', nn.Conv2d(32, 32, kernel_size=(3, 3), stride=1, bias=False, dilation=5, padding=5)),
            ('bn3', nn.BatchNorm2d(32, eps=1e-3, momentum=0.01)),
            ('relu3', nn.ReLU())
        ]))

x = torch.randn(2,3,2)
z = torch.randn(2,5,6)

a = np.random.randn(5,)
b = np.random.randn(2,)
val = [a,b]
max_n = max([len(x) for x in val])
poi = []
for i, p in enumerate(val):
    print(p)
    pad_l = max_n - len(p)
    p_pad = np.pad(p, (0, pad_l), mode='constant', constant_values=0)
    print(p_pad)
    poi.append(p_pad)

c = np.stack(poi,axis = 0)

print(c.shape)


# i = gen_bev_map(x)


# y = x[:,0,:]

# filt = np.logical_and((y>0), (y<1))
# indices = np.argwhere(filt)
# fil = y[indices]

# fil = y[:,indices[:,1]]


'''
p_y = np.zeros([5],np.int)
p_x = np.zeros([5],np.int)

for i in range(5):
    p_y[i] = random.randint(0,2)
    p_x[i] = random.randint(0,2)

print(p_y)
print(p_x)

val = np.array([0,1,2,3,4])


proj_xyz = np.full((3, 3), -1, dtype=np.float32)

proj_xyz[p_y,p_x] = val

print(proj_xyz[p_y,p_x].shape)
print(proj_xyz)
'''
'''
net = RangeNet(x, cfg)

down, up = net(x)

up_in = nn.functional.interpolate(x, scale_factor=(2,2), mode='bilinear', align_corners=True)


out = up[len(up) - 1]

print(out.shape)
'''
# print(net)

'''
o = dil_c1(x)
o1 = dil_c2(o)
o2 = dil_c3(o1)
conta = torch.cat((o,o1,o2), 1)

print(o.shape)
print(o1.shape)
print(o2.shape)
print(conta.shape)
'''