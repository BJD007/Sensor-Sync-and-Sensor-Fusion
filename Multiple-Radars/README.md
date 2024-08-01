# Multiple-Radars

## This code does the following:
- Sets up ROS subscribers for multiple 77GHz radars (assuming three radars in this example).
- Uses message_filters.ApproximateTimeSynchronizer to synchronize incoming radar messages.
- Implements a simple sensor fusion algorithm using an Extended Kalman Filter (EKF).
- Processes and fuses data from multiple radars in the radar_callback method.
- Publishes the fused radar data.

## To use this code:
- Save it as a Python file (e.g., radar_fusion.py) in your ROS package's scripts directory.
- Make the file executable: chmod +x radar_fusion.py
- Add the following line to your package's CMakeLists.txt:

    catkin_install_python(PROGRAMS scripts/radar_fusion.py
      DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

- Build your ROS package: catkin_make
- Run the node: rosrun your_package_name radar_fusion.py

## To adapt this code to your specific radar hardware:
- Adjust the topic names in the subscriber initialization to match your radar topics.
- Modify the pointcloud2_to_array function if your PointCloud2 messages have a different structure or additional fields.
- Tune the DBSCAN parameters (eps and min_samples) based on your radar's resolution and noise characteristics.
- Adjust the UKF parameters (process noise, measurement noise, initial state uncertainty) based on your radar's specifications and expected dynamics of the tracked objects.


Created on 2022-11-05