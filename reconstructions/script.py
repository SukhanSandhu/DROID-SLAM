import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# import open3d as o3d

# import sys
# np.set_printoptions(threshold=sys.maxsize)



# A = np.load('benchpoints2.npy')
# B = np.load('benchremovepoints2.npy')
# aset = set([tuple(x) for x in A])
# bset = set([tuple(x) for x in B])
# f=aset-bset

# pcd = o3d.geometry.PointCloud()

# pcd.points = o3d.utility.Vector3dVector(f)
# o3d.visualization.draw_geometries([pcd])

A = np.load('bench/poses.npy')
print(A.shape)