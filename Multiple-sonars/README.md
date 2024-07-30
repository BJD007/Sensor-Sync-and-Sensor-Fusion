# Multiple-sonars

## The code does the following:
- Sets up ROS subscribers for six sonars (front, back, left, right, up, down).
- Uses message_filters.ApproximateTimeSynchronizer to synchronize incoming sonar messages.
- Implements sensor fusion using an Extended Kalman Filter (EKF).
- Processes and fuses data from all sonars in the sonar_callback method.
- Publishes the fused sonar data as an Odometry message.

## Key features:
- Sensor Synchronization: The code uses ROS's ApproximateTimeSynchronizer to ensure that readings from all sonars are processed together.
- Extended Kalman Filter: The EKF is used to fuse the sonar measurements and estimate the drone's position and velocity in 3D space.
- State Estimation: The state vector includes position (x, y, z) and velocity (vx, vy, vz).
- Dynamic Update: The state transition matrix is updated with the time difference between measurements to account for varying update rates.
- Odometry Publishing: The fused state is published as an Odometry message, which includes position, orientation, and their covariances.

## To use this code:
- Save it as a Python file (e.g., sonar_fusion.py) in your ROS package's scripts directory.
- Make the file executable: chmod +x sonar_fusion.py
- Add the following line to your package's CMakeLists.txt:

    catkin_install_python(PROGRAMS scripts/sonar_fusion.py
      DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

- Build your ROS package: catkin_make
- Run the node: rosrun your_package_name sonar_fusion.py

Note that this code assumes a specific setup with six sonars. You may need to adjust the number of sonars and their orientations based on your drone's configuration. Also, the fusion algorithm assumes a simple motion model and might need refinement based on your drone's dynamics and the characteristics of your sonar sensors.

## For more advanced applications, we need to:
- Incorporate IMU data for better motion estimation.
- Implement outlier rejection to handle noisy or erroneous sonar readings.
- Add a mapping component to build a 3D representation of the environment.
- Implement obstacle avoidance algorithms using the fused sonar data.




Created on 2016-10-01