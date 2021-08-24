from laserscan import  LaserScan, SemLaserScan
import numpy as np
import os





def transform_points_to_range():
    return 



rangeset = LaserScan(project=True)

for f in os.listdir('./data/kitti/bin/'):
    print(f)
    rangeset.open_scan('./data/kitti/bin/' + f, f)





