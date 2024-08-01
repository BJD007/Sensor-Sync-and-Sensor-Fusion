#!/usr/bin/env python3

import rospy
import message_filters
from sensor_msgs.msg import Range, Imu
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovariance, TwistWithCovariance
import numpy as np
from filterpy.kalman import ExtendedKalmanFilter as EKF
from sklearn.cluster import DBSCAN

class AdvancedSonarFusion:
    def __init__(self):
        rospy.init_node('advanced_sonar_fusion_node', anonymous=True)

        # Subscribers for multiple sonars and IMU
        self.sonar_front = message_filters.Subscriber('/sonar/front', Range)
        self.sonar_back = message_filters.Subscriber('/sonar/back', Range)
        self.sonar_left = message_filters.Subscriber('/sonar/left', Range)
        self.sonar_right = message_filters.Subscriber('/sonar/right', Range)
        self.sonar_up = message_filters.Subscriber('/sonar/up', Range)
        self.sonar_down = message_filters.Subscriber('/sonar/down', Range)
        self.imu_sub = message_filters.Subscriber('/imu/data', Imu)

        # Time synchronizer
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.sonar_front, self.sonar_back, self.sonar_left, 
             self.sonar_right, self.sonar_up, self.sonar_down, self.imu_sub],
            queue_size=10, slop=0.1)
        self.ts.registerCallback(self.sonar_callback)

        # Publisher for fused sonar data
        self.fused_pub = rospy.Publisher('/fused_sonar', Odometry, queue_size=10)

        # Initialize EKF for sensor fusion
        self.ekf = EKF(dim_x=9, dim_z=6)  # State: [x, y, z, vx, vy, vz, ax, ay, az]
        self.ekf.x = np.zeros(9)  # Initial state
        self.ekf.F = np.eye(9)  # State transition matrix
        self.ekf.H = np.zeros((6, 9))  # Measurement function
        self.ekf.H[:, :3] = np.eye(6)  # Position measurements
        self.ekf.R = np.eye(6) * 0.1  # Measurement noise
        self.ekf.Q = np.eye(9) * 0.01  # Process noise

        self.last_time = rospy.Time.now()

        # Initialize DBSCAN for outlier rejection
        self.dbscan = DBSCAN(eps=0.5, min_samples=2)

        # Initialize map (for simplicity, a list of points)
        self.map = []

    def sonar_callback(self, front, back, left, right, up, down, imu):
        current_time = rospy.Time.now()
        dt = (current_time - self.last_time).to_sec()
        self.last_time = current_time

        # Update state transition matrix with dt
        self.ekf.F[:3, 3:6] = np.eye(3) * dt
        self.ekf.F[3:6, 6:] = np.eye(3) * dt

        # Predict step
        self.ekf.predict()

        # Prepare measurement vector
        z = np.array([front.range, back.range, left.range, right.range, up.range, down.range])

        # Outlier rejection using DBSCAN
        valid_measurements = self.reject_outliers(z)

        # Update step with valid measurements
        if valid_measurements.size > 0:
            self.ekf.update(valid_measurements)

        # Update state with IMU data
        self.update_state_with_imu(imu)

        # Update map with current position
        self.update_map()

        # Publish fused data
        self.publish_fused_odom()

        # Implement obstacle avoidance
        self.avoid_obstacles()

    def reject_outliers(self, measurements):
        # Use DBSCAN to reject outliers
        measurements = measurements.reshape(-1, 1)
        labels = self.dbscan.fit_predict(measurements)
        valid_measurements = measurements[labels != -1].flatten()
        return valid_measurements

    def update_state_with_imu(self, imu):
        # Update EKF state with IMU data
        self.ekf.x[6] = imu.linear_acceleration.x
        self.ekf.x[7] = imu.linear_acceleration.y
        self.ekf.x[8] = imu.linear_acceleration.z

    def update_map(self):
        # Add current position to map
        position = self.ekf.x[:3]
        self.map.append(position)

    def avoid_obstacles(self):
        # Implement a simple obstacle avoidance algorithm
        position = self.ekf.x[:3]
        for obstacle in self.map:
            distance = np.linalg.norm(position - obstacle)
            if distance < 1.0:  # If an obstacle is within 1 meter
                # Move away from the obstacle
                avoidance_vector = position - obstacle
                avoidance_vector /= np.linalg.norm(avoidance_vector)
                self.ekf.x[:3] += avoidance_vector * 0.1  # Move 0.1 meters away

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
        fusion = AdvancedSonarFusion()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
