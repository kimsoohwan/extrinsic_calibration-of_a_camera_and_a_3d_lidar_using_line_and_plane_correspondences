# Find Normal Vector of A Plane In Pointcloud

The code gets the dimension of the calibration target, a rotation, and a translation vector for moving the target with respect to the coordinate of LiDAR, characteristics of our LiDAR such as (maximum range, error, number of rays, angel between rays in the vertical and horizontal dimension, etc.). Then it calculates the intersection of LiDAR's rays and calibration target.

However, in these images, I did not add the noise yet. Also, the size and rotation in visualization may seem incorrect. However, they are correct, and it is a problem of Matplotlib because different axes have different scales.

Step to find plane and line equations of calibration target's edges:
-I generated a noisy point could with a simulator that I wrote. Figure 1
-I found the plane equation with RANSAC, and I projected points to the plane. Figure 2
-I found different lines (arrays) in the point cloud. This part is parameter sensitive and always does not work and selection of good hyper parameter is needed. Figure 3
-For each of those point arrays in the previous part, I found its line equation with RANSAC. Figure 4
-I projected the point cloud's point on the line equations. Figure 6
-I found points of edges. Figure 7
-I found points of left-lower, left-upper, right-lower, and right-upper edges of the calibration target, Figure 8
-I found line equations for each edge of the calibration target with the RANSAC algorithm.
