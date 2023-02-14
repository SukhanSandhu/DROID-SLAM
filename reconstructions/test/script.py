import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import open3d as o3d
from multiset import *

import sys
np.set_printoptions(threshold=sys.maxsize)



A = np.load('../outdoorGazebo/save.npy')
B = np.load('../outdoorGazebo/remove.npy')

traj = np.load('../outdoorGazebo/trajectory.npy')
remTraj = np.load('../outdoorGazebo/rem_trajectory.npy')

#B = np.load('remove.npy')
aset = Multiset([tuple(x) for x in A])
bset = Multiset([tuple(x) for x in B])

trajset= Multiset([tuple(x) for x in traj])
remTrajset= Multiset([tuple(x) for x in remTraj])

f=aset-bset
fTraj=trajset-remTrajset

final=np.asarray(f)
finalTraj=np.asarray(fTraj)

print(final.shape)
print(finalTraj.shape)
#
np.save("../outdoorGazebo/finalPoints.npy",final)
np.save("../outdoorGazebo/finalTrajPoints.npy",finalTraj)


pcd = o3d.geometry.PointCloud()

pcd.points = o3d.utility.Vector3dVector(final)

lines=np.array([[x,x+1] for x in range(finalTraj.shape[0]-1)])


traj=o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(finalTraj),
        lines=o3d.utility.Vector2iVector(lines))



o3d.visualization.draw_geometries([pcd,traj])
