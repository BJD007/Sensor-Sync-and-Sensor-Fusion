#!/usr/bin/env python3

import rospy
import message_filters
from sensor_msgs.msg import Range
from nav_msgs.msg import Odometry
import numpy as np
from geometry_msgs.msg import PoseWithCovariance, TwistWithCovariance
from filterpy.kalman import ExtendedKalmanFilter as EKF

class SonarFusion:
    def __init__(self):
        rospy.init_node('sonar_fusion_node', anonymous=True)

        # Subscribers for multiple sonars
        self.sonar_front = message_filters.Subscriber('/sonar/front', Range)
        self.sonar_back = message_filters.Subscriber('/sonar/back', Range)
        self.sonar_left = message_filters.Subscriber('/sonar/left', Range)
        self.sonar_right = message_filters.Subscriber('/sonar/right', Range)
        self.sonar_up = message_filters.Subscriber('/sonar/up', Range)
        self.sonar_down = message_filters.Subscriber('/sonar/down', Range)

        # Time synchronizer
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.sonar_front, self.sonar_back, self.sonar_left, 
             self.sonar_right, self.sonar_up, self.sonar_down],
            queue_size=10, slop=0.1)
        self.ts.registerCallback(self.sonar_callback)

        # Publisher for fused sonar data
        self.fused_pub = rospy.Publisher('/fused_sonar', Odometry, queue_size=10)

        # Initialize EKF for sensor fusion
        self.ekf = EKF(dim_x=6, dim_z=6)  # State: [x, y, z, vx, vy, vz]
        self.ekf.x = np.zeros(6)  # Initial state
        self.ekf.F = np.array([[1, 0, 0, 1, 0, 0],
                               [0, 1, 0, 0, 1, 0],
                               [0, 0, 1, 0, 0, 1],
                               [0, 0, 0, 1, 0, 0],
                               [0, 0, 0, 0, 1, 0],
                               [0, 0, 0, 0, 0, 1]])  # State transition matrix
        self.ekf.H = np.eye(6)  # Measurement function
        self.ekf.R = np.eye(6) * 0.1  # Measurement noise
        self.ekf.Q = np.eye(6) * 0.01  # Process noise

        self.last_time = rospy.Time.now()

    def sonar_callback(self, front, back, left, right, up, down):
        current_time = rospy.Time.now()
        dt = (current_time - self.last_time).to_sec()
        self.last_time = current_time

        # Update state transition matrix with dt
        self.ekf.F[:3, 3:] = np.eye(3) * dt

        # Predict
        self.ekf.predict()

        # Prepare measurement vector
        z = np.array([front.range, back.range, left.range, right.range, up.range, down.range])

        # Update
        self.ekf.update(z)

        # Publish fused data
        self.publish_fused_odom()

    def publish_fused_odom(self):
        odom = Odometry()
        odom.header.stamp = rospy.Time.now()
        odom.header.frame_id = "odom"
        odom.child_frame_id = "base_link"

        # Set the position
        odom.pose.pose.position.x = self.ekf.x[0]
        odom.pose.pose.position.y = self.ekf.x[1]
        odom.pose.pose.position.z = self.ekf.x[2]

        # Set the orientation (assuming no rotation for simplicity)
        odom.pose.pose.orientation.w = 1.0

        # Set the position covariance
        odom.pose.covariance = [0.0] * 36
        odom.pose.covariance[0] = self.ekf.P[0, 0]
        odom.pose.covariance[7] = self.ekf.P[1, 1]
        odom.pose.covariance[14] = self.ekf.P[2, 2]

        # Set the velocity
        odom.twist.twist.linear.x = self.ekf.x[3]
        odom.twist.twist.linear.y = self.ekf.x[4]
        odom.twist.twist.linear.z = self.ekf.x[5]

        # Set the velocity covariance
        odom.twist.covariance = [0.0] * 36
        odom.twist.covariance[0] = self.ekf.P[3, 3]
        odom.twist.covariance[7] = self.ekf.P[4, 4]
        odom.twist.covariance[14] = self.ekf.P[5, 5]

        self.fused_pub.publish(odom)

if __name__ == '__main__':
    try:
        fusion = SonarFusion()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
