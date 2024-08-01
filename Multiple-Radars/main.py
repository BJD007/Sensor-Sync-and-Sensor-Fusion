#!/usr/bin/env python3

import rospy
import message_filters
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import Odometry
import numpy as np
from scipy.spatial.transform import Rotation as R
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import MerweScaledSigmaPoints
import tf2_ros
import tf2_geometry_msgs
from geometry_msgs.msg import PoseWithCovarianceStamped
import sensor_msgs.point_cloud2 as pc2
from sklearn.cluster import DBSCAN

class AdvancedRadarFusion:
    def __init__(self):
        rospy.init_node('advanced_radar_fusion_node', anonymous=True)

        # Subscribers for multiple 77GHz radars
        self.radar1_sub = message_filters.Subscriber('/radar1/pointcloud', PointCloud2)
        self.radar2_sub = message_filters.Subscriber('/radar2/pointcloud', PointCloud2)
        self.radar3_sub = message_filters.Subscriber('/radar3/pointcloud', PointCloud2)

        # Time synchronizer
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.radar1_sub, self.radar2_sub, self.radar3_sub],
            queue_size=10, slop=0.1)
        self.ts.registerCallback(self.radar_callback)

        # Publishers
        self.fused_pub = rospy.Publisher('/fused_radar', PointCloud2, queue_size=10)
        self.pose_pub = rospy.Publisher('/fused_pose', PoseWithCovarianceStamped, queue_size=10)

        # Initialize UKF for sensor fusion
        self.init_ukf()

        # TF2 Buffer for coordinate transformations
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # DBSCAN for clustering
        self.dbscan = DBSCAN(eps=0.5, min_samples=5)

        # Parameters for adaptive noise estimation
        self.adaptive_noise_window = 10
        self.measurement_history = []

    def init_ukf(self):
        # State: [x, y, z, vx, vy, vz, ax, ay, az]
        def fx(x, dt):
            # State transition function
            F = np.array([
                [1, 0, 0, dt, 0, 0, 0.5*dt**2, 0, 0],
                [0, 1, 0, 0, dt, 0, 0, 0.5*dt**2, 0],
                [0, 0, 1, 0, 0, dt, 0, 0, 0.5*dt**2],
                [0, 0, 0, 1, 0, 0, dt, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, dt, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, dt],
                [0, 0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1]
            ])
            return F @ x

        def hx(x):
            # Measurement function
            return x[:3]

        # Initialize sigma points
        sigma_points = MerweScaledSigmaPoints(n=9, alpha=0.1, beta=2., kappa=-1)

        # Initialize UKF
        self.ukf = UKF(dim_x=9, dim_z=3, fx=fx, hx=hx, dt=0.1, points=sigma_points)
        self.ukf.x = np.zeros(9)
        self.ukf.P *= 0.1
        self.ukf.R = np.eye(3) * 0.1  # Measurement noise
        self.ukf.Q = np.eye(9) * 0.01  # Process noise

    def radar_callback(self, radar1_msg, radar2_msg, radar3_msg):
        # Convert PointCloud2 messages to numpy arrays
        points1 = self.pointcloud2_to_array(radar1_msg)
        points2 = self.pointcloud2_to_array(radar2_msg)
        points3 = self.pointcloud2_to_array(radar3_msg)

        # Transform points to a common frame (e.g., base_link)
        points1 = self.transform_points(points1, radar1_msg.header.frame_id, 'base_link')
        points2 = self.transform_points(points2, radar2_msg.header.frame_id, 'base_link')
        points3 = self.transform_points(points3, radar3_msg.header.frame_id, 'base_link')

        # Perform sensor fusion
        fused_points, fused_state, covariance = self.fuse_radar_data(points1, points2, points3)

        # Publish fused data
        fused_msg = self.array_to_pointcloud2(fused_points, 'base_link')
        self.fused_pub.publish(fused_msg)

        # Publish fused pose
        self.publish_fused_pose(fused_state, covariance)

    def pointcloud2_to_array(self, cloud_msg):
        # Convert PointCloud2 message to numpy array
        return np.array(list(pc2.read_points(cloud_msg, field_names=("x", "y", "z"), skip_nans=True)))

    def array_to_pointcloud2(self, points, frame_id):
        # Convert numpy array to PointCloud2 message
        header = rospy.Header()
        header.stamp = rospy.Time.now()
        header.frame_id = frame_id
        return pc2.create_cloud_xyz32(header, points)

    def transform_points(self, points, from_frame, to_frame):
        try:
            trans = self.tf_buffer.lookup_transform(to_frame, from_frame, rospy.Time())
            return tf2_geometry_msgs.do_transform_point(points, trans)
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logwarn(f"TF Error: {e}")
            return points

    def fuse_radar_data(self, points1, points2, points3):
        all_points = np.vstack((points1, points2, points3))

        # Cluster points using DBSCAN
        clusters = self.dbscan.fit_predict(all_points)

        fused_points = []
        for cluster in set(clusters):
            if cluster != -1:  # -1 represents noise in DBSCAN
                cluster_points = all_points[clusters == cluster]
                
                # Use the mean of the cluster as the measurement
                measurement = np.mean(cluster_points, axis=0)

                # Predict
                self.ukf.predict()

                # Update
                self.ukf.update(measurement)

                # Get fused point
                fused_point = self.ukf.x[:3]
                fused_points.append(fused_point)

        # Adaptive noise estimation
        self.update_noise_parameters(fused_points)

        return np.array(fused_points), self.ukf.x, self.ukf.P

    def update_noise_parameters(self, measurements):
        # Add current measurements to history
        self.measurement_history.extend(measurements)
        if len(self.measurement_history) > self.adaptive_noise_window:
            self.measurement_history = self.measurement_history[-self.adaptive_noise_window:]

        # Calculate measurement variance
        if len(self.measurement_history) == self.adaptive_noise_window:
            measurement_variance = np.var(self.measurement_history, axis=0)
            self.ukf.R = np.diag(measurement_variance)

    def publish_fused_pose(self, state, covariance):
        pose_msg = PoseWithCovarianceStamped()
        pose_msg.header.stamp = rospy.Time.now()
        pose_msg.header.frame_id = "base_link"

        pose_msg.pose.pose.position.x = state[0]
        pose_msg.pose.pose.position.y = state[1]
        pose_msg.pose.pose.position.z = state[2]

        # Convert velocity to orientation (assuming forward motion)
        yaw = np.arctan2(state[4], state[3])
        quat = R.from_euler('xyz', [0, 0, yaw]).as_quat()
        pose_msg.pose.pose.orientation.x = quat[0]
        pose_msg.pose.pose.orientation.y = quat[1]
        pose_msg.pose.pose.orientation.z = quat[2]
        pose_msg.pose.pose.orientation.w = quat[3]

        pose_msg.pose.covariance = covariance.flatten().tolist()

        self.pose_pub.publish(pose_msg)

if __name__ == '__main__':
    try:
        fusion = AdvancedRadarFusion()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
