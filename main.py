#!/usr/bin/env python3

import rospy
import message_filters
from sensor_msgs.msg import LaserScan, NavSatFix, Imu, Range, Image
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
from filterpy.kalman import ExtendedKalmanFilter as EKF

class SensorFusion:
    def __init__(self):
        rospy.init_node('sensor_fusion_node', anonymous=True)

        # Subscribers
        self.lidar_sub = message_filters.Subscriber('/scan', LaserScan)
        self.gps_sub = message_filters.Subscriber('/gps/fix', NavSatFix)
        self.imu_sub = message_filters.Subscriber('/imu/data', Imu)
        self.sonar_sub = message_filters.Subscriber('/sonar', Range)
        self.camera_sub = message_filters.Subscriber('/camera/image_raw', Image)

        # Time synchronizer
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.lidar_sub, self.gps_sub, self.imu_sub, self.sonar_sub, self.camera_sub],
            queue_size=10, slop=0.1)
        self.ts.registerCallback(self.sensor_callback)

        # Publishers
        self.fused_odom_pub = rospy.Publisher('/fused_odometry', Odometry, queue_size=10)

        # Other initializations
        self.bridge = CvBridge()
        self.last_position = np.zeros(3)
        self.last_orientation = np.array([0, 0, 0, 1])  # Quaternion [x, y, z, w]
        self.last_image = None
        self.orb = cv2.ORB_create()
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # Initialize EKF
        self.ekf = EKF(dim_x=9, dim_z=6)  # State: [x, y, z, vx, vy, vz, roll, pitch, yaw]
        self.ekf.x = np.zeros(9)  # Initial state
        self.ekf.F = np.eye(9)  # State transition matrix
        self.ekf.H = np.zeros((6, 9))  # Measurement function
        self.ekf.H[:3, :3] = np.eye(3)  # Position measurements
        self.ekf.H[3:, 6:] = np.eye(3)  # Orientation measurements
        self.ekf.R = np.eye(6) * 0.1  # Measurement noise
        self.ekf.Q = np.eye(9) * 0.01  # Process noise

    def sensor_callback(self, lidar_msg, gps_msg, imu_msg, sonar_msg, camera_msg):
        # Process LiDAR data
        ranges = np.array(lidar_msg.ranges)
        angles = np.linspace(lidar_msg.angle_min, lidar_msg.angle_max, len(ranges))
        
        # Process GPS data
        lat, lon, alt = gps_msg.latitude, gps_msg.longitude, gps_msg.altitude

        # Process IMU data
        orientation = np.array([
            imu_msg.orientation.x,
            imu_msg.orientation.y,
            imu_msg.orientation.z,
            imu_msg.orientation.w
        ])
        angular_velocity = np.array([
            imu_msg.angular_velocity.x,
            imu_msg.angular_velocity.y,
            imu_msg.angular_velocity.z
        ])
        linear_acceleration = np.array([
            imu_msg.linear_acceleration.x,
            imu_msg.linear_acceleration.y,
            imu_msg.linear_acceleration.z
        ])

        # Process Sonar data
        sonar_range = sonar_msg.range

        # Process Camera data
        try:
            cv_image = self.bridge.imgmsg_to_cv2(camera_msg, "bgr8")
        except Exception as e:
            rospy.logerr(f"Error converting image: {e}")
            return

        # Sensor fusion
        fused_position, fused_orientation = self.fuse_sensors(
            ranges, angles, lat, lon, alt, orientation, angular_velocity,
            linear_acceleration, sonar_range, cv_image)

        # Publish fused odometry
        self.publish_fused_odometry(fused_position, fused_orientation)

    def fuse_sensors(self, ranges, angles, lat, lon, alt, orientation,
                     angular_velocity, linear_acceleration, sonar_range, cv_image):
        # Predict step
        dt = 0.1  # Assume 10Hz update rate
        self.ekf.F[:3, 3:6] = np.eye(3) * dt  # Update position based on velocity
        self.ekf.predict()

        # Update step
        z = np.zeros(6)
        z[:3] = np.array([lat, lon, alt])  # GPS measurement
        z[3:] = R.from_quat(orientation).as_euler('xyz')  # IMU orientation measurement
        self.ekf.update(z)

        # Get fused position and orientation
        fused_position = self.ekf.x[:3]
        fused_orientation = R.from_euler('xyz', self.ekf.x[6:]).as_quat()

        # Use LiDAR for obstacle detection
        min_distance = np.min(ranges)
        if min_distance < 0.5:  # If obstacle is closer than 0.5m
            self.ekf.x[:3] = self.last_position  # Don't move

        # Use Sonar for height above ground
        self.ekf.x[2] = sonar_range

        # Visual odometry
        visual_displacement = self.visual_odometry(cv_image)
        self.ekf.x[:2] += visual_displacement

        # Update last known position and orientation
        self.last_position = fused_position
        self.last_orientation = fused_orientation

        return fused_position, fused_orientation

    def visual_odometry(self, current_image):
        if self.last_image is None:
            self.last_image = current_image
            return np.zeros(2)

        # Detect ORB features
        kp1, des1 = self.orb.detectAndCompute(self.last_image, None)
        kp2, des2 = self.orb.detectAndCompute(current_image, None)

        # Match features
        matches = self.bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)

        # Estimate motion
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        if M is not None:
            # Extract translation
            dx = M[0, 2]
            dy = M[1, 2]
        else:
            dx, dy = 0, 0

        self.last_image = current_image
        return np.array([dx, dy])

    def publish_fused_odometry(self, position, orientation):
        odom_msg = Odometry()
        odom_msg.header.stamp = rospy.Time.now()
        odom_msg.header.frame_id = "odom"
        odom_msg.child_frame_id = "base_link"

        odom_msg.pose.pose.position.x = position[0]
        odom_msg.pose.pose.position.y = position[1]
        odom_msg.pose.pose.position.z = position[2]

        odom_msg.pose.pose.orientation.x = orientation[0]
        odom_msg.pose.pose.orientation.y = orientation[1]
        odom_msg.pose.pose.orientation.z = orientation[2]
        odom_msg.pose.pose.orientation.w = orientation[3]

        self.fused_odom_pub.publish(odom_msg)

if __name__ == '__main__':
    try:
        fusion = SensorFusion()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
% Main function
% Main function
% Main function
% Main function
% Main function
% Main function
