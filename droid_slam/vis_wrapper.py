import torch
import cv2
import lietorch
import droid_backends
import time
import argparse
import numpy as np
import open3d as o3d

from lietorch import SE3
import geom.projective_ops as pops

class vis_wrapper:
    def __init__(self,):
        self.vis=o3d.visualization.VisualizerWithKeyCallback()
        
        self.trajPoints=np.ndarray(shape=(1,3))
        self.savePoints=np.ndarray(shape=(1,3))
        self.remoPoints=np.ndarray(shape=(1,3))



