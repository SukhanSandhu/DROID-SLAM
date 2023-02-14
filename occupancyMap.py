import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import open3d as o3d

from multiset import *

class occupancyMap:
    def __init__(self,pointData,lod,thresh):
        self.origin=[0,0,0]
        self.lod=lod
        self.thresh=thresh
        self.points=pointData
        self.dim=(0,0,0)
        self.calcDim() #no of cubes along each axis
        print(self.dim)
        self.densityMap=np.full(self.dim,0,dtype=int)
        self.occupancy=np.full(self.dim,0,dtype=int)
        self.fillDensityMap()
        self.fillOccupancy()

        

    def calcDim(self):
        data=np.reshape(self.points,(3,-1))
        self.xmax=self.points[0][0]
        self.xmin=self.xmax
        self.ymax=self.points[0][1]
        self.ymin=self.ymax
        self.zmax=self.points[0][2]
        self.zmin=self.zmax

        for point in self.points:

            self.xmax=max(self.xmax,point[0])
            self.xmin=min(self.xmin,point[0])

            self.ymax=max(self.ymax,point[1])
            self.ymin=min(self.ymin,point[1])

            self.zmax=max(self.zmax,point[2])
            self.zmin=min(self.zmin,point[2])


        dimx=(self.xmax-self.xmin)*self.lod
        dimy=(self.ymax-self.ymin)*self.lod
        dimz=(self.zmax-self.zmin)*self.lod
        
        dim=max(dimx,dimy,dimz)

        self.dim=(int(dimx)+1,int(dimy)+1,int(dimz)+1)


    def fillDensityMap(self):
        self.map=np.ndarray(shape=self.dim)
        for point in self.points:
            cube=self.getCube(point)
            self.densityMap[cube[0]][cube[1]][cube[2]]+=1

    def getCube(self,coords):
        x=int((coords[0]-self.xmin)*self.lod)
        y=int((coords[1]-self.ymin)*self.lod)
        z=int((coords[2]-self.zmin)*self.lod)
        return [x,y,z]

    def fillOccupancy(self):
        for x in range(self.dim[0]):
            for y in range(self.dim[1]):
                for z in range(self.dim[2]):
                    if(self.densityMap[x][y][z]>=self.thresh):
                        self.occupancy[x][y][z]=1

    def render(self):

        scene=[]

        points = np.load('reconstructions/outdoorGazebo/finalTrajPoints.npy')

        lines=np.array([[x,x+1] for x in range(points.shape[0]-1)])

        traj=o3d.geometry.LineSet(
               points=o3d.utility.Vector3dVector(points),
               lines=o3d.utility.Vector2iVector(lines))

        scene.append(traj)

        for x in range(self.dim[0]):
            for y in range(self.dim[1]):
                for z in range(self.dim[2]):
                    if(self.occupancy[x][y][z]==1):
                        mesh_box = o3d.geometry.TriangleMesh.create_box(width=1.0/self.lod,height=1.0/self.lod,depth=1.0/self.lod)
                        mesh_box.paint_uniform_color([1-x/self.dim[0]+.4, (1-y/self.dim[1])/2+.4, z/self.dim[2]+.4])
                        mesh_box.translate((x/self.lod+self.xmin,y/self.lod+self.ymin,z/self.lod+self.zmin))
                        scene.append(mesh_box)

        o3d.visualization.draw_geometries(scene)

data=np.load("reconstructions/outdoorGazebo/finalPoints.npy")
map1=occupancyMap(data,10,1)
map1.render()