import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import open3d as o3d
pcd = o3d.io.read_point_cloud("points.ply")


#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#ax.scatter(v[:,0],v[:,1],v[:,2], s=1)
#plt.show()
#v=v.reshape(3,-1)
#plt.plot(v[0],v[1])
#plt.show()

o3d.visualization.draw_geometries([pcd])