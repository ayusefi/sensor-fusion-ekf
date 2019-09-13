#!/usr/bin/env python

# Bu dugumun abone olan ve yayinlayan topicler.
# Abone oluyor: /odom
#		/scan
# Yayinliyor:	/durum_tahmin


# Lazim olan kutuphaneleri import ediyoruz.
import rospy
import sys
import numpy
import std_msgs.msg
import time
import matplotlib.pyplot as plt
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from sensor_fusion_ekf.msg import *
from tf.transformations import euler_from_quaternion
from math import sqrt
from matplotlib.patches import Ellipse



# FILTRE BASLAMASI
# DURUMUN BASLANGIC BILGISI
measurement = 0
avg_meas_dist= 0
predicted_state_est = numpy.matrix(numpy.zeros((3,1)))
PEst = numpy.eye(3)

# Estimation parameter of EKF
Q = numpy.diag([0.0005, 0.0005, 0.0005])**2	# covariance matrix of process noise	
R = numpy.diag([0.005, 0.005, 0.005])**2	# covariance matrix of observation noise


DT = 0.1  # time tick [s]
SIM_TIME = 50.0  # simulation time [s]

show_animation = True


# Yayinlayici topic'i olustur.
pub = rospy.Publisher('durum_tahmin', Config, queue_size=10)
time.sleep(5)
global pub


# Ilk olarak sensorlerden gelen butun verileri al.
#def veri_al():
	
	# Dugumu baslat.
#	rospy.init_node('tahmin_guncel', anonymous=True)

	# /scan topic'e abone ol.
#	rospy.Subscriber('scan', LaserScan, kinect_scan_tahmin)

	# /odom topic'e abone ol.
#	rospy.Subscriber('odom', Odometry, odom_durum_tahmin)

	# ekf islemi her 1/1127 saniyede guncelleniyor.
#	rospy.Timer(rospy.Duration(0.1127), olc_guncel, oneshot=False)

	# Python'un islmini dugum bitene kadar devam ettirir.
#	rospy.spin()


# Kincet'ten gerekli verileri al.
def kinect_scan_tahmin(scan_veri):
	
	global measurement
	global avg_meas_dist

	measurement = scan_veri.ranges
	
	topl_mesaf = 0
	uzunluk = 0
	
	for i in range (580,600):
		if str(measurement[i]) != 'nan' :
			topl_mesaf += measurement[i]
			uzunluk +=1
	
	if uzunluk != 0:
		avg_meas_dist = topl_mesaf/uzunluk


# Odometry'den gerekli verileri al.
def odom_durum_tahmin(odom_veri):
	global predicted_state_est
	
	predicted_state_est = motion_model(odom_veri)

#	rospy.logdebug(predicted_state_est)
#	print(predicted_state_est)

	# The Odom varians is a 6 x 6 matrix. We need 3 x 3 matrix (x, y, teta) and ignore the z, roll and pitch.
	Q = numpy.array([[odom_veri.pose.covariance[0],odom_veri.pose.covariance[1],odom_veri.pose.covariance[5]],[odom_veri.pose.covariance[6],odom_veri.pose.covariance[7],odom_veri.pose.covariance[11]],[odom_veri.pose.covariance[30],odom_veri.pose.covariance[31],odom_veri.pose.covariance[35]]])
	
#	print(Q)
	

def observation(xTrueConfi):
	global predicted_state_est
	xTrueConfi = predicted_state_est
	z = avg_meas_dist
	
	return xTrueConfi, z


def motion_model(odomet_veri):
	(roll, pitch, yaw) = euler_from_quaternion([odomet_veri.pose.pose.orientation.x, odomet_veri.pose.pose.orientation.y, odomet_veri.pose.pose.orientation.z, odomet_veri.pose.pose.orientation.w])
	x = odomet_veri.pose.pose.position.x
	y = odomet_veri.pose.pose.position.y
	
	state_est = Config(x,y,yaw)
	
	return state_est


def observation_model(xPrede):
	
	
	# Observation Model
	# Odometry gore beklenen measurementu hesaplamak.
	zz = numpy.array([avg_meas_dist,avg_meas_dist,avg_meas_dist])

	return zz


def jacobF():
	# Jacobian of Motion Model
	jF = numpy.array([[1,0,0],[0,1,0],[0,0,1]])
	
	return jF


def jacobH():
	# Jacobian of Observation Model
	jH = numpy.array([[9999, 0, 0],[0,1,0],[0,0,9999]])
	
	return jH


def olc_guncel(event):
	

	global PEst

	# Predict
	z = numpy.array([avg_meas_dist,avg_meas_dist])
	xTrueConfig = predicted_state_est
	xPred = predicted_state_est
	jF = jacobF()
	PPred = jF * PEst * numpy.transpose(jF) + Q
	
	# Update
	jH = jacobH()
	zPred = observation_model(xPred)
	y = numpy.array([avg_meas_dist,avg_meas_dist, avg_meas_dist]) - zPred	# The difference between the actual and expected measurement.
	S = jH * PPred * numpy.transpose(jH) + R	# Residual Covarians
	K = PPred * numpy.transpose(jH) * numpy.linalg.inv(S)	# Kalman gain
	guncel_durum_tahmin = numpy.array([xPred.x, xPred.y, xPred.th]) + numpy.dot(K,y)	# Guncellenmis durum tahmini
	
	# Storing predicted covariance for plot
	pre_cov_store = PPred
	
	# Guncellenmis covarians tahmini
	PEst= (numpy.eye(3) - numpy.cross(K,jH)) * PPred

	#storing updated covariance estimate for plotting
	up_cov_store = PEst

	durum_tahmin = Config(guncel_durum_tahmin[0],guncel_durum_tahmin[1],guncel_durum_tahmin[2])
	

#	rospy.logdebug(durum_tahmin)

#	rospy.loginfo(durum_tahmin)


	pub.publish(durum_tahmin)
#	return guncel_durum_tahmin, PEst


#def plot_covariance_ellipse(guncel_durum_tahmin):
#	Pxy = PEst[0:
	# Plotting
	
#	fig = plt.figure(1)
#	ax = fig.gca()
#	ax1 = plt.gca()
	
#	plt.cla()

	# Guncellenmis durum tahmini:
#	x_updated = []
#	y_updated = []
	
#	plt.ion() 

	
#	x_updated.append(durum_tahmin.x)
#	y_updated.append(durum_tahmin.y)

	# Update is plotted as blue points. 
#	plt.plot(x_updated,y_updated,'b*')
#	plt.ylabel("y")
#	plt.xlabel("x")

	

	# Tahmin edilen durum: 
#	x_predict = []
#	y_predict = []

#	x_predict.append(predicted_state_est.x)
#	y_predict.append(predicted_state_est.y)

	# Prediction is plotted as red points. 
#	plt.plot(x_predict, y_predict, 'ro')
#	plt.ylabel("y")
#	plt.xlabel("x")	

	# Plot the covariance
	# I expect the updated covariance to decrease in the direction of measurement and increase in the 
	# direction that I am not taking any measurements.  

#	lambda_pre,v=numpy.linalg.eig(pre_cov_store)
#	lambda_pre = numpy.sqrt(lambda_pre)

#	ax = plt.subplot(111, aspect = 'equal')

#	for j in xrange(1,4):
#		ell = Ellipse(xy=(numpy.mean(x_predict),numpy.mean(y_predict)), width=lambda_pre[0]/(j*19), height=lambda_pre[1]/(j*10),angle=numpy.rad2deg(numpy.arccos(v[0,0])))

#	ell.set_facecolor('none')
#	ax.add_artist(ell)

#	lambda_up,v=numpy.linalg.eig(up_cov_store)
#	lambda_up= numpy.sqrt(lambda_up)

#	ax = plt.subplot(111, aspect = 'equal')

#	for j in xrange(1,4):
#		ell = Ellipse(xy=(numpy.mean(x_updated),numpy.mean(y_updated)), width=lambda_up[0]/j*10, height=lambda_up[1]/j*10,angle=numpy.rad2deg(numpy.arccos(v[0,0])))
#	ell.set_facecolor('none')
#	ax.add_artist(ell)

#	plt.ioff()
#	plt.show()
#	plt.draw()
#	plt.grid


def main():
	print(__file__ + " start!!")

	

	# State Vector [x y yaw v]'
	xEst = numpy.array([0, 0, 0])
	xTrue = numpy.array([0, 0, 0])
	PEst = numpy.eye(3)
	z = numpy.zeros((1, 2))

	xDR = numpy.array([0, 0, 0])	# Odometry
	
	
	# Dugumu baslat.
	rospy.init_node('olcum_guncel2', anonymous=True)

	# /scan topic'e abone ol.
	rospy.Subscriber('scan', LaserScan, kinect_scan_tahmin)

	# /odom topic'e abone ol.
	rospy.Subscriber('odom', Odometry, odom_durum_tahmin)

	# ekf islemi her 1/1127 saniyede guncelleniyor.
#	rospy.Timer(rospy.Duration(0.1127), olc_guncel, oneshot=False)

	xTrueConfig = Config(xTrue[0], xTrue[1], xTrue[2])

	xDRConfig = Config(xDR[0], xDR[1], xDR[2])
	xEstConfig = Config(xEst[0], xEst[1], xEst[2])

	global xEstConfig
	global z
	

	# Python'un islmini dugum bitene kadar devam ettirir.

	
	
	# history
	hxEst = xEst
	hxTrue = xTrue
	hxDR = xTrue
	hz = numpy.zeros((1, 2))

#	while SIM_TIME >= time:
#		time += DT
		
#		xTrueConfig, z = observation(xTrueConfig)

	rospy.Timer(rospy.Duration(1), olc_guncel, oneshot=False)

		# store data history
       	hxEst = numpy.hstack((hxEst, xEst))
       	hxDR = numpy.hstack((hxDR, xDR))
	xTruestack1 = numpy.array([xTrueConfig.x, xTrueConfig.y, xTrueConfig.th])
       	hxTrue = numpy.hstack((hxTrue, xTruestack1))
       	hz = numpy.vstack((hz, z))
	
#		if show_animation:
#			plt.cla()
#			plt.ion()
#			plt.plot(hz[:, 0], hz[:, 1], ".g")
#			plt.plot(numpy.array(hxTrue[0, :]).flatten, numpy.array(hxTrue[1, :]).flatten(), "-b")
#			plt.plot(numpy.array(hxDR[0, :]).flatten(),
#                     		 numpy.array(hxDR[1, :]).flatten(), "-k")
#            		plt.plot(numpy.array(hxEst[0, :]).flatten(),
#                     		 numpy.array(hxEst[1, :]).flatten(), "-r")	

#			plt.axis("equal")
#       	   	plt.grid(True)
#        	    	plt.pause(0.001)
#			plt.ioff()
#			plt.show()
	rospy.spin()


if __name__ == '__main__':
	#program calistiginda, once measurementleri al
	try: main()
	except rospy.ROSInterruptException: pass
