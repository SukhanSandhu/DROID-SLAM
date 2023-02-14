import rospy
import numpy as np

from nav_msgs.msg import Odometry

global groundTruthTraj
groundTruthTraj=np.ndarray(shape=(1,3))
def odometryCb(data):
    global groundTruthTraj
    print(data.pose.pose.position.x,data.pose.pose.position.y,data.pose.pose.position.z)
    groundTruthTraj=np.append(groundTruthTraj,[[data.pose.pose.position.x,data.pose.pose.position.y,data.pose.pose.position.z]],axis=0)

def myhook():
    print("saving",groundTruthTraj.shape)
    np.save("reconstructions/test/groundTruthTraj.npy",groundTruthTraj)

if __name__ == "__main__":
    rospy.init_node('oodometry', anonymous=True) #make node 
    rospy.Subscriber('ground_truth/state',Odometry,odometryCb)
    rospy.on_shutdown(myhook)
    rospy.spin()