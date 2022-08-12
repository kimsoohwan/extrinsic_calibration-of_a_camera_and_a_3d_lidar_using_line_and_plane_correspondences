import imp
import numpy as np
from generate_a_plane import generate_a_lidar_plane_in_3D, show_point_cloud
from find_plane_in_lidar import ransac_plane_in_lidar
import matplotlib.pyplot as plt

def distance_of_points_to_plane(point_cloud, plane_equation):
    # distance of points in point cloud to the plane
    point_cloud_with_one = np.hstack((point_cloud, np.ones(shape=(point_cloud.shape[0], 1))))
    distance_points_to_plane = np.dot(point_cloud_with_one, plane_equation.T)

    return distance_points_to_plane


def map_points_to_plane(point_cloud, plane_equation):
    
    # find distance of points to the plane
    distance_points_to_plane = distance_of_points_to_plane(point_cloud=point_cloud, plane_equation=plane_equation)
    distance_points_to_plane = np.reshape(distance_points_to_plane, newshape=(-1, 1))

    projected_point_cloud = -1 * np.dot(distance_points_to_plane, np.reshape(plane_equation[0:3], newshape=(1,3))) + point_cloud

    return projected_point_cloud

def find_different_lines(point_cloud, min_distance=None):
    """
    point cloud
    min_distance: minimum distance for two points to be on the same line (mm). if it be None, it find it itself.
    """
    if min_distance is None:

        all_dis = []
        for row_i in range(0, point_cloud.shape[0]):
            dis = []
            for row_j in range(0, point_cloud.shape[0]):
                if row_i == row_j:
                    continue
                dis.append(np.linalg.norm(point_cloud[row_i]-point_cloud[row_j]))

            if len(dis) != 0:
                min_1 = np.min(dis)
                dis.remove(min_1)
                min_2 = np.min(dis)
                
                if min_1 > 150:
                    continue

                if min_1 *  9 < min_2:
                    all_dis.append(min_1)
                else:
                    all_dis.append(min_2)

        print(all_dis)

        min_distance = np.max(all_dis)
    
        print(min_distance)

    lines = []
    seen_point = [False] * point_cloud.shape[0]
    
    for row_i in range(point_cloud.shape[0]):

        if seen_point[row_i] == True:
            continue
        
        for line_idx, line in enumerate(lines):
            for row_j in range(len(line)):
            
                if np.linalg.norm(point_cloud[row_i]-line[row_j]) <= min_distance:
                    lines[line_idx].append(point_cloud[row_i])
                    seen_point[row_i] = True
                    break
        
        if seen_point[row_i] == False:
            lines.append([point_cloud[row_i]])

        lines_copy = []
        for line in lines:
            if len(line) > 5:
                lines_copy.append(np.array(line))

    return lines_copy

def find_edges_of_calibration_target_in_lidar(lidar_points, plane_equation, display=False):

    # convert to numpy
    point_cloud = np.copy(lidar_points)

    # map points to plane
    projected_point_cloud = map_points_to_plane(point_cloud=point_cloud, plane_equation=plane_equation)

    # find different lines
    lines = find_different_lines(point_cloud=projected_point_cloud)

    if display == True:
        show_point_cloud(point_cloud=point_cloud)
        show_point_cloud(point_cloud=projected_point_cloud)
        show_point_cloud(point_cloud=lines)
        
    plt.show()


if __name__ == '__main__':

    # generate an plane (point cloud)
    output_dic = generate_a_lidar_plane_in_3D(
                                    rotation_vector=np.array([45.0, 0.0, 0.0]), 
                                    translation_vector=np.array([5000.0, 0.0, 0.0]),
                                    display=False
                                )

    # find plane equation
    best_ratio_plane = ransac_plane_in_lidar(lidar_point=output_dic['lidar_point_with_noise'])
    
    # find plane (calibration target) edges
    find_edges_of_calibration_target_in_lidar(lidar_points=output_dic['lidar_point_with_noise'], plane_equation=best_ratio_plane['plane_equation'], display=True)
