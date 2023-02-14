import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import open3d as o3d

f=open("outdoor_world.txt","r")

data=f.read().strip('\n').split("\n")
meataData=[int(x) for x in data[0].split(',')]

mapData=data[1:]

height=len(mapData)
print("height = ",height)

densityMap=[]
for x in mapData:
    
    data=[int(y) for y in x.split(",")]
    densityMap.append(data)

densityMap=np.array(densityMap)
densityMap=np.reshape(densityMap,(height,meataData[1],meataData[2]))
print(densityMap.shape)


scene = []
for x in range(meataData[1]):
    for y in range(meataData[2]):
        for z in range(height):
            if(densityMap[z][x][y]>1):
                mesh_box = o3d.geometry.TriangleMesh.create_box(width=1.0,height=1.0,depth=1.0)
                mesh_box.paint_uniform_color([0.0, z/height, 1-z/height])
                mesh_box.translate((y,x,z))
                scene.append(mesh_box)

o3d.visualization.draw_geometries(scene)