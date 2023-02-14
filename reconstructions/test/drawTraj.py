import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import open3d as o3d
from multiset import *

import sys
np.set_printoptions(threshold=sys.maxsize)


points = np.load('/home/uav/DROID-SLAM/reconstructions/outdoorGazebo/poses.npy')
points=np.hsplit(points,np.array([3,4]))[0]

lines=np.array([[x,x+1] for x in range(points.shape[0]-1)])


traj=o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines))


o3d.visualization.draw_geometries([traj])


#A = np.load('save.npy')
#B = np.load('remove.npy')
#aset = Multiset([tuple(x) for x in A])
#bset = Multiset([tuple(x) for x in B])
#
#f=aset-bset
#
#pcd = o3d.geometry.PointCloud()
#
#pcd.points = o3d.utility.Vector3dVector(f)
#o3d.visualization.draw_geometries([pcd])