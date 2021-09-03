import numpy as np
import mayavi.mlab

#雷达点云可视化
def point_vis(file_str):
    # lidar_path换成自己的.bin文件路径
    pointcloud = np.fromfile(str(file_str), dtype=np.float32, count=-1).reshape([-1, 4])

    x = pointcloud[:, 0]  # x position of point
    y = pointcloud[:, 1]  # y position of point
    z = pointcloud[:, 2]  # z position of point

    r = pointcloud[:, 3]  # reflectance value of point
    d = np.sqrt(x ** 2 + y ** 2)  # Map Distance from sensor

    degr = np.degrees(np.arctan(z / d))

    vals = 'height'
    if vals == "height":
        col = z
    else: 
        col = d

    fig = mayavi.mlab.figure(bgcolor=(0, 0, 0), size=(640, 500))
    mayavi.mlab.points3d(x, y, z,
                         col,  # Values used for Color
                         mode="point",
                         colormap='spectral',  # 'bone', 'copper', 'gnuplot'
                         # color=(0, 1, 0),   # Used a fixed (r,g,b) instead
                         figure=fig,
                         )

    mayavi.mlab.show()

if __name__ == '__main__':
    point_vis('./data/kitti/bin/002394.bin')
