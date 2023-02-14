import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import open3d as o3d





A = np.load('rotatedPoints.npy')



pcd = o3d.geometry.PointCloud()

pcd.points = o3d.utility.Vector3dVector(A)
o3d.visualization.draw_geometries([pcd])
