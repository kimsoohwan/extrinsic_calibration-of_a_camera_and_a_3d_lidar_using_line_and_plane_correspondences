# Calibration of Camera and Lidar with only one pose

This repository contains code for the calibration of the Camera and Lidar with only one pose. The algorithm is based of the following paper with slight changes.

> Zhou, Lipu, Zimo Li, and Michael Kaess. "Automatic extrinsic calibration of a camera and a 3d lidar using line and plane correspondences." 2018 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2018.

"Automatic extrinsic calibration of a camera and a 3d lidar using line and plane correspondences" uses line and plane correspondance of calibration target in RGB image and LiDAR point cloud to find rotation and translation matrix to map points from lidar point cloud to camera coordinate.

## Calibration Target
Our calibration target is a simple checkerboard calibration target like the paper. However, we used yellow tape around it in order to easily find the exterior edges of the calibration target.

## Small Changes
There are some small and minor changes in the algorithm. For example, the paper works for one or more poses. However, our code just implemented calibration with one pose. The code can be easily changed to incorporate more changes. Also, one of the formulations in the paper results in a rotation matrix that is not orthonormal. We changed that in a manner that keeps the property. However, both methods are implemented.

## Input/Output/Execute
An input set and its corresponding output are proved in the `example_real_img_lidar_points` folder. The input is an image, LiDAR points on the calibration target, all LiDAR points, calibration parameter of the camera. For more informatioin about format of inputs, you can see provided input example.

For running the code and testing the code, you can execute this python file: `automatic_extrinsic_calibration_of_a_camera_and_a_3D_lidar_using_line_and_plane_correspondences_2018.py`

Output of algorithm for example input:
![alt text](https://github.com/farhad-dalirani/find_normal_vector_plane_pointcloud/blob/main/example_real_img_lidar_points/10-10-2022-16-40-20/img_target_lidar_points.png?raw=true)

![alt text](https://github.com/farhad-dalirani/find_normal_vector_plane_pointcloud/blob/main/example_real_img_lidar_points/10-10-2022-16-40-20/img_scence_lidar_points.png?raw=true)


