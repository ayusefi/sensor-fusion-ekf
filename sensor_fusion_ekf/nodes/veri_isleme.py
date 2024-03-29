#!/usr/bin/env python

import rospy
import sys
import numpy
from nav_msgs.msg import Odometry
from sensor_fusion_ekf.msg import * 
from tf.transformations import euler_from_quaternion


# /odom is the topic that the turtlebot publishes from nodelet manager:  /mobile_base_nodelet_manager
# /odom is of message type nav_msgs/Odometery
def get_robot_prediction():

	#make node
	rospy.init_node('tahmin', anonymous=True)
	rospy.Subscriber("odom", Odometry, get_state_belief)

	# spin() simply keeps python from exiting until this node is stopped
	rospy.spin()


def get_state_belief(bel_state):
	#get the yaw from the quaternion
	(roll,pitch,yaw) = euler_from_quaternion([bel_state.pose.pose.orientation.x, bel_state.pose.pose.orientation.y, bel_state.pose.pose.orientation.z, bel_state.pose.pose.orientation.w])
	x = bel_state.pose.pose.position.x
	y = bel_state.pose.pose.position.y

	covariance = bel_state.pose.covariance
	#rospy.loginfo(rospy.get_caller_id() + "I think I am at coordinates: x=%s, y=%s,th=%s",x,y,yaw)
	print ""
	print "I think I am at coordinates x=%s, y=%s, th=%s"%(bel_state.pose.pose.position.x, bel_state.pose.pose.position.y, yaw)
	print ""
	print "My covariaonce is: ", numpy.array(covariance[0]), numpy.array(covariance[7]),numpy.array(covariance[35]), numpy.array(covariance[14]), numpy.array(covariance[21]), numpy.array(covariance[28])
	pub = rospy.Publisher('state_estimate',Config, queue_size = 10)

	estimated_config = Config(x,y,yaw)
	rospy.loginfo(estimated_config)
	pub.publish(estimated_config)
	
#Main section of code that will continuously run unless rospy receives interuption (ie CTRL+C)
if __name__ == '__main__':
 
	try: get_robot_prediction()
	except rospy.ROSInterruptException: pass



