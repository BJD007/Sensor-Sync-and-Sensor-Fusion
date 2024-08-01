# Multiple-sonars


## To use this code:
- Save it as a Python file (e.g., sonar_fusion.py) in your ROS package's scripts directory.
- Make the file executable: chmod +x sonar_fusion.py
- Add the following line to your package's CMakeLists.txt:

    catkin_install_python(PROGRAMS scripts/sonar_fusion.py
      DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

- Build your ROS package: catkin_make
- Run the node: rosrun your_package_name sonar_fusion.py

## bAdjustments for Your Drone:
- Number and Orientation of Sonars: Adjust the number of sonars and their orientations based on your drone's configuration.
- Fusion Algorithm: The fusion algorithm assumes a simple motion model. You might need to refine it based on your drone's dynamics and the characteristics of your sonar sensors.
- Outlier Rejection Parameters: Tune the parameters of DBSCAN (eps and min_samples) based on the noise characteristics of your sonar sensors.
- Obstacle Avoidance: The obstacle avoidance algorithm is basic and might need to be enhanced based on your specific requirements.

Created on 2016-10-01