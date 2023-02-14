import numpy as np

data = np.load('disps.npy')
print("Disps", data.shape)

data = np.load('images.npy')
print("Images", data.shape)

data = np.load('poses.npy')
print("Poses", data.shape)