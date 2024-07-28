# Multiple-Radars

#!/usr/bin/env python3

import rospy
import message_filters
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import Odometry
import numpy as np
from scipy.spatial.transform import Rotation as R
from filterpy.kalman import ExtendedKalmanFilter as EKF

class RadarFusion:
    def __init__(self):
        rospy.init_node('radar_fusion_node', anonymous=True)

        # Subscribers for multiple 77GHz radars
        self.radar1_sub = message_filters.Subscriber('/radar1/pointcloud', PointCloud2)
        self.radar2_sub = message_filters.Subscriber('/radar2/pointcloud', PointCloud2)
        self.radar3_sub = message_filters.Subscriber('/radar3/pointcloud', PointCloud2)

        # Time synchronizer
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.radar1_sub, self.radar2_sub, self.radar3_sub],
            queue_size=10, slop=0.1)
        self.ts.registerCallback(self.radar_callback)

        # Publisher for fused radar data
        self.fused_pub = rospy.Publisher('/fused_radar', PointCloud2, queue_size=10)

        # Initialize EKF for sensor fusion
        self.ekf = EKF(dim_x=6, dim_z=3)  # State: [x, y, z, vx, vy, vz]
        self.ekf.F = np.array([[1, 0, 0, 1, 0, 0],
                               [0, 1, 0, 0, 1, 0],
                               [0, 0, 1, 0, 0, 1],
                               [0, 0, 0, 1, 0, 0],
                               [0, 0, 0, 0, 1, 0],
                               [0, 0, 0, 0, 0, 1]])
        self.ekf.H = np.array([[1, 0, 0, 0, 0, 0],
                               [0, 1, 0, 0, 0, 0],
                               [0, 0, 1, 0, 0, 0]])
        self.ekf.R = np.eye(3) * 0.1  # Measurement noise
        self.ekf.Q = np.eye(6) * 0.01  # Process noise

    def radar_callback(self, radar1_msg, radar2_msg, radar3_msg):
        # Convert PointCloud2 messages to numpy arrays
        points1 = self.pointcloud2_to_array(radar1_msg)
        points2 = self.pointcloud2_to_array(radar2_msg)
        points3 = self.pointcloud2_to_array(radar3_msg)

        # Perform sensor fusion
        fused_points = self.fuse_radar_data(points1, points2, points3)

        # Publish fused data
        fused_msg = self.array_to_pointcloud2(fused_points)
        self.fused_pub.publish(fused_msg)

    def pointcloud2_to_array(self, cloud_msg):
        # Convert PointCloud2 message to numpy array
        # This is a placeholder - implement actual conversion based on your message format
        return np.random.rand(100, 3)  # Placeholder: 100 random 3D points

    def array_to_pointcloud2(self, points):
        # Convert numpy array to PointCloud2 message
        # This is a placeholder - implement actual conversion based on your message format
        msg = PointCloud2()
        # Populate msg fields
        return msg

    def fuse_radar_data(self, points1, points2, points3):
        fused_points = []
        for p1, p2, p3 in zip(points1, points2, points3):
            # Predict
            self.ekf.predict()

            # Update with measurements from each radar
            self.ekf.update(p1)
            self.ekf.update(p2)
            self.ekf.update(p3)

            # Get fused point
            fused_point = self.ekf.x[:3]
            fused_points.append(fused_point)

        return np.array(fused_points)

if __name__ == '__main__':
    try:
        fusion = RadarFusion()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

Created on 2022-11-05