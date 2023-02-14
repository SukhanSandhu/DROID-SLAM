import numpy as np
import math
import pyrr
from pyrr import Quaternion, Matrix33, Matrix44, Vector3, Vector4


data=np.load("reconstructions/outdoorGazebo/finalPoints.npy")

mat=pyrr.matrix44.create_from_x_rotation(math.pi)

for index,vector in enumerate(data):
    vec= Vector4.from_vector3(Vector3(vector), w=1.0)
    vec=pyrr.matrix44.apply_to_vector(mat,vec)
    data[index]=vec[:3]



np.save("reconstructions/outdoorGazebo/finalPoints.npy",data)

