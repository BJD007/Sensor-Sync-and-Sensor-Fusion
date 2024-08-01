#!/usr/bin/env python3

import rospy
import message_filters
from sensor_msgs.msg import LaserScan, NavSatFix, Imu, Range, Image
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped, TwistWithCovarianceStamped
from cv_bridge import CvBridge
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import MerweScaledSigmaPoints
from sklearn.cluster import DBSCAN
import tf2_ros
import tf2_geometry_msgs
from std_msgs.msg import Float32MultiArray

class AdvancedSensorFusion:
    def __init__(self):
        # Initialize ROS node
        rospy.init_node('advanced_sensor_fusion_node', anonymous=True)

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
        self.twist_with_cov_pub = rospy.Publisher('/twist_with_covariance', TwistWithCovarianceStamped, queue_size=10)
        self.obstacle_pub = rospy.Publisher('/detected_obstacles', Float32MultiArray, queue_size=10)

        # Initialize OpenCV bridge
        self.bridge = CvBridge()

        # Initialize last known position, orientation, and image
        self.last_position = np.zeros(3)
        self.last_orientation = np.array([0, 0, 0, 1])  # Quaternion [x, y, z, w]
        self.last_image = None

        # Initialize ORB feature detector and matcher
        self.orb = cv2.ORB_create(nfeatures=2000)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # Initialize Unscented Kalman Filter
        self.init_ukf()

        # Initialize DBSCAN for obstacle clustering
        self.dbscan = DBSCAN(eps=0.5, min_samples=5)

        # Initialize TF2 buffer and listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # Parameters for adaptive noise estimation
        self.adaptive_noise_window = 10
        self.measurement_history = []

    def init_ukf(self):
        # Initialize Unscented Kalman Filter
        # State: [x, y, z, vx, vy, vz, qw, qx, qy, qz, wx, wy, wz]
        # where q is quaternion and w is angular velocity
        def fx(x, dt):
            # State transition function
            new_x = x.copy()
            new_x[:3] += x[3:6] * dt
            q = R.from_quat(x[6:10]).as_matrix()
            w = x[10:]
            dq = R.from_rotvec((w * dt)).as_matrix()
            new_q = R.from_matrix(q @ dq).as_quat()
            new_x[6:10] = new_q
            return new_x

        def hx(x):
            # Measurement function
            return np.concatenate([x[:3], R.from_quat(x[6:10]).as_euler('xyz')])

        # Initialize sigma points
        sigma_points = MerweScaledSigmaPoints(n=13, alpha=0.1, beta=2., kappa=-1)

        # Initialize UKF
        self.ukf = UKF(dim_x=13, dim_z=6, fx=fx, hx=hx, dt=0.1, points=sigma_points)
        self.ukf.x = np.zeros(13)
        self.ukf.x[6] = 1  # Initial quaternion w component
        self.ukf.P *= 0.1
        self.ukf.R = np.diag([0.1, 0.1, 0.1, 0.01, 0.01, 0.01])  # Measurement noise
        self.ukf.Q = np.eye(13) * 0.01  # Process noise

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
        fused_state, covariance = self.fuse_sensors(
            ranges, angles, lat, lon, alt, orientation, angular_velocity,
            linear_acceleration, sonar_range, cv_image)

        # Extract fused position and orientation
        fused_position = fused_state[:3]
        fused_orientation = fused_state[6:10]
        fused_velocity = fused_state[3:6]
        fused_angular_velocity = fused_state[10:]

        # Publish fused odometry, pose with covariance, and twist with covariance
        self.publish_fused_odometry(fused_position, fused_orientation, fused_velocity, fused_angular_velocity)
        self.publish_pose_with_covariance(fused_position, fused_orientation, covariance[:6,:6])
        self.publish_twist_with_covariance(fused_velocity, fused_angular_velocity, covariance[3:,3:])

    def fuse_sensors(self, ranges, angles, lat, lon, alt, orientation,
                     angular_velocity, linear_acceleration, sonar_range, cv_image):
        # Predict step
        self.ukf.predict()

        # Update step
        z = np.zeros(6)
        z[:3] = np.array([lat, lon, alt])  # GPS measurement
        z[3:] = R.from_quat(orientation).as_euler('xyz')  # IMU orientation measurement
        self.ukf.update(z)

        # Obstacle detection and avoidance using LiDAR
        obstacles = self.obstacle_detection_and_avoidance(ranges, angles)

        # Publish detected obstacles
        self.publish_obstacles(obstacles)

        # Use Sonar for height above ground
        self.ukf.x[2] = sonar_range

        # Visual odometry
        visual_displacement, visual_rotation = self.visual_odometry(cv_image)
        self.ukf.x[:2] += visual_displacement
        self.ukf.x[6:10] = R.from_matrix(
            R.from_quat(self.ukf.x[6:10]).as_matrix() @ R.from_euler('z', visual_rotation).as_matrix()
        ).as_quat()

        # Adaptive noise estimation
        self.update_noise_parameters(z)

        return self.ukf.x, self.ukf.P

    def obstacle_detection_and_avoidance(self, ranges, angles):
        # Convert polar coordinates to Cartesian
        x = ranges * np.cos(angles)
        y = ranges * np.sin(angles)
        points = np.column_stack((x, y))

        # Cluster obstacles using DBSCAN
        clusters = self.dbscan.fit_predict(points)

        obstacles = []
        for cluster in set(clusters):
            if cluster != -1:  # -1 represents noise in DBSCAN
                cluster_points = points[clusters == cluster]
                center = np.mean(cluster_points, axis=0)
                radius = np.max(np.linalg.norm(cluster_points - center, axis=1))
                obstacles.append((center, radius))

        # Simple avoidance: adjust velocity based on nearest obstacle
        if obstacles:
            nearest_obstacle = min(obstacles, key=lambda o: np.linalg.norm(o[0]))
            distance_to_obstacle = np.linalg.norm(nearest_obstacle[0]) - nearest_obstacle[1]
            if distance_to_obstacle < 1.0:
                avoidance_vector = -nearest_obstacle[0] / np.linalg.norm(nearest_obstacle[0])
                self.ukf.x[3:5] += avoidance_vector * (1.0 - distance_to_obstacle)

        return obstacles

    def visual_odometry(self, current_image):
        if self.last_image is None:
            self.last_image = current_image
            return np.zeros(2), 0

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
        else:
            dx, dy, rotation = 0, 0, 0

        self.last_image = current_image
        return np.array([dx, dy]), rotation

    def update_noise_parameters(self, measurement):
        # Add current measurement to history
        self.measurement_history.append(measurement)
        if len(self.measurement_history) > self.adaptive_noise_window:
            self.measurement_history.pop(0)

        # Calculate measurement variance
        if len(self.measurement_history) == self.adaptive_noise_window:
            measurement_variance = np.var(self.measurement_history, axis=0)
            self.ukf.R = np.diag(measurement_variance)

    def publish_fused_odometry(self, position, orientation, velocity, angular_velocity):
        odom_msg = Odometry()
        odom_msg.header.stamp = rospy.Time.now()
        odom_msg.header.frame_id = "odom"
        odom_msg.child_frame_id = "base_link"

        odom_msg.pose.pose.position.x, odom_msg.pose.pose.position.y, odom_msg.pose.pose.position.z = position
        odom_msg.pose.pose.orientation.x, odom_msg.pose.pose.orientation.y, odom_msg.pose.pose.orientation.z, odom_msg.pose.pose.orientation.w = orientation
        odom_msg.twist.twist.linear.x, odom_msg.twist.twist.linear.y, odom_msg.twist.twist.linear.z = velocity
        odom_msg.twist.twist.angular.x, odom_msg.twist.twist.angular.y, odom_msg.twist.twist.angular.z = angular_velocity

        self.fused_odom_pub.publish(odom_msg)

    def publish_pose_with_covariance(self, position, orientation, covariance):
        pose_msg = PoseWithCovarianceStamped()
        pose_msg.header.stamp = rospy.Time.now()
        pose_msg.header.frame_id = "odom"

        pose_msg.pose.pose.position.x, pose_msg.pose.pose.position.y, pose_msg.pose.pose.position.z = position
        pose_msg.pose.pose.orientation.x, pose_msg.pose.pose.orientation.y, pose_msg.pose.pose.orientation.z, pose_msg.pose.pose.orientation.w = orientation
        pose_msg.pose.covariance = covariance.flatten().tolist()

        self.pose_with_cov_pub.publish(pose_msg)

    def publish_twist_with_covariance(self, velocity, angular_velocity, covariance):
        twist_msg = TwistWithCovarianceStamped()
        twist_msg.header.stamp = rospy.Time.now()
        twist_msg.header.frame_id = "base_link"

        twist_msg.twist.twist.linear.x, twist_msg.twist.twist.linear.y, twist_msg.twist.twist.linear.z = velocity
        twist_msg.twist.twist.angular.x, twist_msg.twist.twist.angular.y, twist_msg.twist.twist.angular.z = angular_velocity
        twist_msg.twist.covariance = covariance.flatten().tolist()

        self.twist_with_cov_pub.publish(twist_msg)

    def publish_obstacles(self, obstacles):
        obstacle_msg = Float32MultiArray()
        obstacle_data = []
        for center
