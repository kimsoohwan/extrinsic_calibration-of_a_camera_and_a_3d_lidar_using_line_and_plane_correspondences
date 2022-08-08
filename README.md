# Find Normal Vector of A Plane In Pointcloud

The code gets the dimension of the calibration target, a rotation, and a translation vector for moving the target with respect to the coordinate of LiDAR, characteristics of our LiDAR such as (maximum range, error, number of rays, angel between rays in the vertical and horizontal dimension, etc.). Then it calculates the intersection of LiDAR's rays and calibration target.

However, in these images, I did not add the noise yet. Also, the size and rotation in visualization may seem incorrect. However, they are correct, and it is a problem of Matplotlib because different axes have different scales.