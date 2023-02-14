import rospy
import sys
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import cv2
import numpy as np
from nav_msgs.msg import Odometry


global groundTruthTraj
groundTruthTraj=np.ndarray(shape=(1,3))
def odometryCb(data):
    global groundTruthTraj
    groundTruthTraj=np.append(groundTruthTraj,[[data.pose.pose.position.x,data.pose.pose.position.y,data.pose.pose.position.z]],axis=0)

result = cv2.VideoWriter('reconstructions/test/testVideo.avi',cv2.VideoWriter_fourcc(*'MJPG'),10, (640,480))

def process_image(msg):
    bridge = CvBridge()  
    img = bridge.imgmsg_to_cv2(msg, "bgr8")
    result.write(img)
    

def myhook():
    result.release()
    print("saving",groundTruthTraj.shape)
    np.save("reconstructions/test/groundTruthTraj.npy",groundTruthTraj)


rospy.init_node('record')
rospy.loginfo('image_sub node started')
rospy.Subscriber('undgro_truth/state',Odometry,odometryCb)
rospy.Subscriber("/front_cam/camera/image", Image, process_image)
rospy.on_shutdown(myhook)
rospy.spin()