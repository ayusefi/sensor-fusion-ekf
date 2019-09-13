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
#import time
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
predicted_state_est = 0
predicted_covarians_est = 0
Q = 0

# Yayinlayici topic'i olustur.
pub = rospy.Publisher('durum_tahmin', Config, queue_size=10)


# Ilk olarak sensorlerden gelen butun verileri al.
def veri_al():
	
	# Dugumu baslat.
	rospy.init_node('tahmin_guncel', anonymous=True)

	
	# /scan topic'e abone ol.
	rospy.Subscriber('scan', LaserScan, kinect_scan_tahmin)

	# /odom topic'e abone ol.
	rospy.Subscriber('odom', Odometry, odom_durum_tahmin)

	# ekf islemi her 1/1127 saniyede guncelleniyor.
	rospy.Timer(rospy.Duration(0.1127), olc_guncel, oneshot=False)



	# Python'un islmini dugum bitene kadar devam ettirir.
	rospy.spin()


# Kincet'ten gerekli verileri al.
def kinect_scan_tahmin(scan_veri):
	
	global measurement
	global avg_meas_dist

	measurement = scan_veri.ranges
	
	topl_mesaf = 0
	uzunluk = 0
	
	for i in range (290,310):
		if str(measurement[i]) != 'nan' :
			topl_mesaf += measurement[i]
			uzunluk +=1
	
	if uzunluk != 0:
		avg_meas_dist = topl_mesaf/uzunluk


# Odometry'den gerekli verileri al.
def odom_durum_tahmin(odom_veri):
	global predicted_state_est
	global predicted_covarians_est
	global Q
	
	predicted_state_est = motion_model(odom_veri)

#	rospy.logdebug(predicted_state_est)
#	print(predicted_state_est)

	# Odom varians, 6 x 6 bir matristir. Bize 3 x 3 matris(x,y,teta) lazim. z, roll ve pitch almiyoruz.
	Q = numpy.array([[odom_veri.pose.covariance[0],odom_veri.pose.covariance[1],odom_veri.pose.covariance[5]],[odom_veri.pose.covariance[6],odom_veri.pose.covariance[7],odom_veri.pose.covariance[11]],[odom_veri.pose.covariance[30],odom_veri.pose.covariance[31],odom_veri.pose.covariance[35]]])
	
#	print(Q)

	# state transition jacobian beliriyoruz.
	state_transition_jacobian = numpy.array([[1,0,0],[0,1,0],[0,0,1]])

#	print(state_transition_jacobian)
	
	predicted_covarians_est = state_transition_jacobian*predicted_covarians_est*numpy.transpose(state_transition_jacobian)+Q
	
#	print(predicted_covarians_est)

def motion_model(odomet_veri):
	(roll, pitch, yaw) = euler_from_quaternion([odomet_veri.pose.pose.orientation.x, odomet_veri.pose.pose.orientation.y, odomet_veri.pose.pose.orientation.z, odomet_veri.pose.pose.orientation.w])
	x = odomet_veri.pose.pose.position.x
	y = odomet_veri.pose.pose.position.y
	
	state_est = Config(x,y,yaw)
	
	return state_est

#	rospy.logdebug(predicted_state_est)
#	print(predicted_state_est)


def olc_guncel(event):
	global pub
	global predicted_covarians_est

	
	# Odometry gore beklenen measurementu hesaplamak.
	beklenen_measurement = numpy.cross(numpy.array([0, 1, 0]), numpy.array([predicted_state_est.x, predicted_state_est.y, predicted_state_est.th]))

	# Gercek sandigimiz ve beklenen measurementun arasindaki fark.
	olc_kal = avg_meas_dist - beklenen_measurement

	# measurement gurultusu
	olc_gurultu_cavarians = 0.005
	
	# measurement jacboian
	H = numpy.array([[9999, 0, 0],[0,1,0],[0,0,9999]])

	# Residual Covarians
	kal_covarians = H*predicted_covarians_est*numpy.transpose(H)+olc_gurultu_cavarians

	# Kalman gain
	kalman_gain = predicted_covarians_est*numpy.transpose(H)*numpy.linalg.inv(kal_covarians)

	# Guncellenmis durum tahmini
	guncel_durum_tahmin = numpy.array([predicted_state_est.x,predicted_state_est.y,predicted_state_est.th]) + numpy.dot(kalman_gain,olc_kal)
	
	# Storing predicted covariance for plot
	pre_cov_store = predicted_covarians_est
	
	# Guncellenmis covarians tahmini
	predicted_covarians_est= (numpy.identity(3) - numpy.cross(kalman_gain,H))*predicted_covarians_est

	#storing updated covariance estimate for plotting
	up_cov_store = predicted_covarians_est

	durum_tahmin = Config(guncel_durum_tahmin[0],guncel_durum_tahmin[1],guncel_durum_tahmin[2])

#	print(predicted_state_est.x)
#	durum_tahmin = Config(predicted_state_est.x, predicted_state_est.y, predicted_state_est.th)

	rospy.logdebug(durum_tahmin)

	rospy.loginfo(durum_tahmin)
	

	pub.publish(durum_tahmin)


	# Plotting
	
#	fig = plt.figure(1)
#	ax = fig.gca()
#	ax1 = plt.gca()
	
#	plt.cla()

	# Guncellenmis durum tahmini:
	x_updated = []
	y_updated = []
	
	plt.ion() 

	
	x_updated.append(durum_tahmin.x)
	y_updated.append(durum_tahmin.y)

	# Update is plotted as blue points. 
	plt.plot(x_updated,y_updated,'b*')
	plt.ylabel("y")
	plt.xlabel("x")

	

	# Tahmin edilen durum: 
	x_predict = []
	y_predict = []

	x_predict.append(predicted_state_est.x)
	y_predict.append(predicted_state_est.y)

	# Prediction is plotted as red points. 
	plt.plot(x_predict, y_predict, 'ro')
	plt.ylabel("y")
	plt.xlabel("x")	

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

	plt.ioff()
	plt.show()
#	plt.draw()
#	plt.grid

if __name__ == '__main__':
	#program calistiginda, once measurementleri al
	try: veri_al()
	except rospy.ROSInterruptException: pass
