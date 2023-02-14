import sys
sys.path.append('droid_slam')

from tqdm import tqdm
import numpy as np
import torch
import lietorch
import cv2
import os
import glob 
import time
import argparse
import rospy
import sys
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import threading

from torch.multiprocessing import Process, Queue
from droid import Droid

import torch.nn.functional as F

img=None
noImage=True
global imageAssigned
global q
q=False

def process_image(msg):
    global img
    global imageAssigned
    bridge = CvBridge()   
    img = bridge.imgmsg_to_cv2(msg, "bgr8")
    imageAssigned=True
    
    
def get_image():
    print("in get image")
    rospy.init_node('image_sub')
    rospy.loginfo('image_sub node started')
    rospy.Subscriber("/front_cam/camera/image", Image, process_image)


def show_image(image):
    image = image.permute(1, 2, 0).cpu().numpy()
    cv2.imshow('image', image / 255.0)
    cv2.waitKey(3)

def map():
    for (t, image, intrinsics) in tqdm(image_stream()):
        if(rospy.get_param("quit")==1):
            return
            
        if t < args.t0:
            continue

        if not args.disable_vis:
            show_image(image[0])

        global droid
        if droid is None:
            args.image_size = [image.shape[2], image.shape[3]]
            droid = Droid(args)
        
        droid.track(t, image, intrinsics=intrinsics)

def image_stream():
    """ image generator """
    
    fx, fy, cx, cy = calib[:4]

    K = np.eye(3)
    K[0,0] = fx
    K[0,2] = cx
    K[1,1] = fy
    K[1,2] = cy

    global imageAssigned
    imageAssigned=False
    while(not imageAssigned):
        continue

    global q
    while(not q):
        image = img
        if len(calib) > 4:
            image = cv2.undistort(image, K, calib[4:])

        h0, w0, _ = image.shape
        h1 = int(h0 * np.sqrt((384 * 512) / (h0 * w0)))
        w1 = int(w0 * np.sqrt((384 * 512) / (h0 * w0)))

        image = cv2.resize(image, (w1, h1))
        image = image[:h1-h1%8, :w1-w1%8]
        image = torch.as_tensor(image).permute(2, 0, 1)

        intrinsics = torch.as_tensor([fx, fy, cx, cy])
        intrinsics[0::2] *= (w1 / w0)
        intrinsics[1::2] *= (h1 / h0)

        yield 10, image[None], intrinsics


def save_reconstruction(droid, reconstruction_path):

    from pathlib import Path
    import random
    import string
    
    t = droid.video.counter.value
    tstamps = droid.video.tstamp[:t].cpu().numpy()
    images = droid.video.images[:t].cpu().numpy()
    disps = droid.video.disps_up[:t].cpu().numpy()
    poses = droid.video.poses[:t].cpu().numpy()
    intrinsics = droid.video.intrinsics[:t].cpu().numpy()

    Path("reconstructions/{}".format(reconstruction_path)).mkdir(parents=True, exist_ok=True)
    np.save("reconstructions/{}/tstamps.npy".format(reconstruction_path), tstamps)
    np.save("reconstructions/{}/images.npy".format(reconstruction_path), images)
    np.save("reconstructions/{}/disps.npy".format(reconstruction_path), disps)
    np.save("reconstructions/{}/poses.npy".format(reconstruction_path), poses)
    np.save("reconstructions/{}/intrinsics.npy".format(reconstruction_path), intrinsics)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    savePts = np.ndarray(shape=(1,3))
    remPts=np.ndarray(shape=(1,3))
    trajPts=np.ndarray(shape=(1,3))
    remTrajPts=np.ndarray(shape=(1,3))
    parser.add_argument("--imagedir", type=str, help="path to image directory")
    parser.add_argument("--calib", type=str, help="path to calibration file")
    parser.add_argument("--t0", default=0, type=int, help="starting frame")
    parser.add_argument("--stride", default=3, type=int, help="frame stride")

    parser.add_argument("--weights", default="droid.pth")
    parser.add_argument("--buffer", type=int, default=512)
    parser.add_argument("--image_size", default=[240, 320])
    parser.add_argument("--disable_vis", action="store_true")

    parser.add_argument("--beta", type=float, default=0.3, help="weight for translation / rotation components of flow")
    parser.add_argument("--filter_thresh", type=float, default=2.4, help="how much motion before considering new keyframe")
    parser.add_argument("--warmup", type=int, default=8, help="number of warmup frames")
    parser.add_argument("--keyframe_thresh", type=float, default=4.0, help="threshold to create a new keyframe")
    parser.add_argument("--frontend_thresh", type=float, default=16.0, help="add edges between frames whithin this distance")
    parser.add_argument("--frontend_window", type=int, default=25, help="frontend optimization window")
    parser.add_argument("--frontend_radius", type=int, default=2, help="force edges between frames within radius")
    parser.add_argument("--frontend_nms", type=int, default=1, help="non-maximal supression of edges")

    parser.add_argument("--backend_thresh", type=float, default=22.0)
    parser.add_argument("--backend_radius", type=int, default=2)
    parser.add_argument("--backend_nms", type=int, default=3)
    parser.add_argument("--upsample", action="store_true")
    parser.add_argument("--reconstruction_path", help="path to saved reconstruction")
    args = parser.parse_args()

    args.stereo = False
    torch.multiprocessing.set_start_method('spawn')
    
    global droid
    droid=None
    global calib
    rospy.set_param("quit",0)
    calib = np.loadtxt(args.calib, delimiter=" ")
    t1 = threading.Thread(target=map)
    t1.start()
    get_image()
    t1.join()
    # need high resolution depths
    
    if args.reconstruction_path is not None:
        args.upsample = True
    
    tstamps = []
    

    if args.reconstruction_path is not None:
        save_reconstruction(droid, args.reconstruction_path)

    traj_est = droid.terminate(image_stream())

    print("saving")
    while(droid.saveQueue.qsize()>0):
        savePts=np.append(savePts,droid.saveQueue.get(),axis=0)
        
    print("saving save pts",savePts.shape)
    np.save("reconstructions/outdoorGazebo/save.npy", savePts)
    while(droid.remQueue.qsize()>0):
        remPts=np.append(remPts,droid.remQueue.get(),axis=0)
        
    print("saving rem pts",remPts.shape)
    np.save("reconstructions/outdoorGazebo/remove.npy", remPts)
    while(droid.trajQueue.qsize()>0):
        trajPts=np.append(trajPts,droid.trajQueue.get(),axis=0)
        
    print("saving traj pts",trajPts.shape)
    np.save("reconstructions/outdoorGazebo/trajectory.npy", trajPts)

    while(droid.remTrajQueue.qsize()>0):
        remTrajPts=np.append(remTrajPts,droid.remTrajQueue.get(),axis=0)
        
    print("saving rem traj pts",remTrajPts.shape)
    np.save("reconstructions/outdoorGazebo/rem_trajectory.npy", remTrajPts)

    print("saving poses")
    np.save("reconstructions/outdoorGazebo/poses.npy", droid.video.poses[:droid.video.counter.value].cpu().numpy())
    #rospy.on_shutdown(shutdownHook)