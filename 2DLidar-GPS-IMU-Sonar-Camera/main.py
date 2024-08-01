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
from sklearn.cluster import DBSCAN
from geometry_msgs.msg import PoseWithCovarianceStamped

class SensorFusion:
    def __init__(self):
        # Initialize ROS node
        rospy.init_node('sensor_fusion_node', anonymous=True)

        # Set up subscribers for various sensors
        self.lidar_sub = message_filters.Subscriber('/scan', LaserScan)
        self.gps_sub = message_filters.Subscriber('/gps/fix', NavSatFix)
        self.imu_sub = message_filters.Subscriber('/imu/data', Imu)
        self.sonar_sub = message_filters.Subscriber('/sonar', Range)
        self.camera_sub = message_filters.Subscriber('/camera/image_raw', Image)

        # Time synchronizer to align sensor data
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.lidar_sub, self.gps_sub, self.imu_sub, self.sonar_sub, self.camera_sub],
            queue_size=10, slop=0.1)
        self.ts.registerCallback(self.sensor_callback)

        # Publishers for fused data
        self.fused_odom_pub = rospy.Publisher('/fused_odometry', Odometry, queue_size=10)
        self.pose_with_cov_pub = rospy.Publisher('/pose_with_covariance', PoseWithCovarianceStamped, queue_size=10)

        # Initialize OpenCV bridge
        self.bridge = CvBridge()

        # Initialize last known position and orientation
        self.last_position = np.zeros(3)
        self.last_orientation = np.array([0, 0, 0, 1])  # Quaternion [x, y, z, w]
        self.last_image = None

        # Initialize ORB feature detector and matcher
        self.orb = cv2.ORB_create(nfeatures=1000)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # Initialize Extended Kalman Filter
        self.init_ekf()

        # Initialize DBSCAN for obstacle clustering
        self.dbscan = DBSCAN(eps=0.5, min_samples=5)

    def init_ekf(self):
        # Initialize Extended Kalman Filter
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

        # Perform sensor fusion
        fused_position, fused_orientation, covariance = self.fuse_sensors(
            ranges, angles, lat, lon, alt, orientation, angular_velocity,
            linear_acceleration, sonar_range, cv_image)

        # Publish fused odometry and pose with covariance
        self.publish_fused_odometry(fused_position, fused_orientation)
        self.publish_pose_with_covariance(fused_position, fused_orientation, covariance)

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

        # Obstacle detection and avoidance using LiDAR
        self.obstacle_detection_and_avoidance(ranges, angles)

        # Use Sonar for height above ground
        self.ekf.x[2] = sonar_range

        # Visual odometry
        visual_displacement = self.visual_odometry(cv_image)
        self.ekf.x[:2] += visual_displacement

        # Update last known position and orientation
        self.last_position = fused_position
        self.last_orientation = fused_orientation

        return fused_position, fused_orientation, self.ekf.P

    def obstacle_detection_and_avoidance(self, ranges, angles):
        # Convert polar coordinates to Cartesian
        x = ranges * np.cos(angles)
        y = ranges * np.sin(angles)
        points = np.column_stack((x, y))

        # Cluster obstacles using DBSCAN
        clusters = self.dbscan.fit_predict(points)

        # Find the nearest cluster
        cluster_distances = []
        for cluster in set(clusters):
            if cluster != -1:  # -1 represents noise in DBSCAN
                cluster_points = points[clusters == cluster]
                distance = np.min(np.linalg.norm(cluster_points, axis=1))
                cluster_distances.append((cluster, distance))

        if cluster_distances:
            nearest_cluster, min_distance = min(cluster_distances, key=lambda x: x[1])
            if min_distance < 0.5:  # If obstacle is closer than 0.5m
                # Simple avoidance: move in the opposite direction
                avoidance_vector = -np.mean(points[clusters == nearest_cluster], axis=0)
                avoidance_vector /= np.linalg.norm(avoidance_vector)
                self.ekf.x[:2] += avoidance_vector * 0.1  # Move 0.1m in avoidance direction

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

        # Use only the top 10% of matches
        num_good_matches = int(len(matches) * 0.1)
        matches = matches[:num_good_matches]

        # Estimate motion
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        if M is not None:
            # Extract translation
            dx = M[0, 2]
            dy = M[1, 2]
            # Extract rotation
            rotation = np.arctan2(M[1, 0], M[0, 0])
            # Update orientation in EKF
            self.ekf.x[8] += rotation
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

    def publish_pose_with_covariance(self, position, orientation, covariance):
        pose_msg = PoseWithCovarianceStamped()
        pose_msg.header.stamp = rospy.Time.now()
        pose_msg.header.frame_id = "odom"

        pose_msg.pose.pose.position.x = position[0]
        pose_msg.pose.pose.position.y = position[1]
        pose_msg.pose.pose.position.z = position[2]

        pose_msg.pose.pose.orientation.x = orientation[0]
        pose_msg.pose.pose.orientation.y = orientation[1]
        pose_msg.pose.pose.orientation.z = orientation[2]
        pose_msg.pose.pose.orientation.w = orientation[3]

        pose_msg.pose.covariance = covariance.flatten().tolist()

        self.pose_with_cov_pub.publish(pose_msg)

if __name__ == '__main__':
    try:
        fusion = SensorFusion()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
