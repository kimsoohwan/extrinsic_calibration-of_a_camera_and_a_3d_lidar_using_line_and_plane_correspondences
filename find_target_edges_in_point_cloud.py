import imp
import numpy as np
from generate_a_plane import generate_a_lidar_plane_in_3D, show_point_cloud
from find_plane_in_lidar import ransac_plane_in_lidar
import matplotlib.pyplot as plt
from find_line_in_lidar import ransac_line_in_lidar, map_point_to_line
from copy import copy


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
                
                if min_1 > 100 or min_2 > 100:
                    continue

                if min_1 *  9 < min_2:
                    all_dis.append(min_1)
                else:
                    all_dis.append(min_2)

        #print(all_dis)

        min_distance = np.max(all_dis)
    
        #print(min_distance)

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

def find_points_on_left_right_border(lines):
    
    points_on_left_border = []
    points_on_right_border = []
    
    for line in lines:
        # sory by y
        line = sorted(line, key = lambda x: x[1])

        points_on_left_border.append(line[0])
        points_on_right_border.append(line[-1])

    all_points = points_on_left_border + points_on_right_border
    all_points = np.array(all_points)

    return {'left_points': points_on_left_border, 'right_points': points_on_right_border, 'border_point_cloud': all_points}

def find_upper_and_lower_points_on_edges(points_on_left_border, points_on_right_border):

    # sort points based in their z
    points_on_left_border = sorted(points_on_left_border, key = lambda x: x[2])
    points_on_right_border = sorted(points_on_right_border, key = lambda x: x[2])

    if len(points_on_left_border) < 4 or len(points_on_right_border) < 4:
        raise ValueError('Not enough points on borders to find lines.')

    for set_idx, points_set in enumerate([points_on_left_border, points_on_right_border]):

        upper_points = []
        lower_points = []
        
        angel_vectors = [-2]
        for i in range(1, len(points_set)-1):
            vec_1 = points_set[i+1] - points_set[i]
            vec_2 = points_set[i] - points_set[i - 1]

            cos_two_vector = np.dot(vec_1, vec_2.T)/(np.linalg.norm(vec_1) * np.linalg.norm(vec_2))

            angel_vectors.append(1-cos_two_vector)

        
        angel_vectors_copy = copy(angel_vectors)
        max_1 = np.argmax(angel_vectors_copy)
        angel_vectors_copy.remove(angel_vectors_copy[max_1])
        max_2 = np.argmax(angel_vectors_copy)
        
        if np.abs(max_1 - max_2) == 1:
            if max_1 > max_2:
                break_point = max_2
            else:
                break_point = max_1
        else:
            break_point = max_1

        for i in range(0, len(points_set)):

            if i <= break_point:
                lower_points.append(points_set[i])
            else:
                upper_points.append(points_set[i])
                
        if set_idx == 0:
            left_lower_points = np.copy(lower_points)
            left_upper_points = np.copy(upper_points)
        else:
            right_lower_points = np.copy(lower_points)
            right_upper_points = np.copy(upper_points)

    return {'left_lower_points': left_lower_points, 'left_upper_points': left_upper_points, 'right_lower_points': right_lower_points, 'right_upper_points': right_upper_points}

def find_edges_of_calibration_target_in_lidar(lidar_points, plane_equation, display=False):

    # convert to numpy
    point_cloud = np.copy(lidar_points)

    # map points to plane
    projected_point_cloud = map_points_to_plane(point_cloud=point_cloud, plane_equation=plane_equation)

    # find different lines
    lines = find_different_lines(point_cloud=projected_point_cloud)

    # find equation of each line in 3D space
    lines_equations = []
    for line in lines:
        best_ratio_line = ransac_line_in_lidar(lidar_point=line)
        lines_equations.append(best_ratio_line['line_equation'])

    # map noisy points of each line to the found line
    point_cloud_mapped_on_lines = None
    list_point_mapped_on_lines = []
    
    for line_idx in range(len(lines)):
        new_line = map_point_to_line(lines[line_idx], lines_equations[line_idx])
        
        if point_cloud_mapped_on_lines is None:
            point_cloud_mapped_on_lines = np.copy(new_line)
        else:
            point_cloud_mapped_on_lines = np.vstack((point_cloud_mapped_on_lines, new_line))
        
        list_point_mapped_on_lines.append(new_line)

    dic_point_border = find_points_on_left_right_border(list_point_mapped_on_lines)

    # find point on upper and lower edges
    edges_points = find_upper_and_lower_points_on_edges(points_on_left_border=dic_point_border['left_points'], points_on_right_border=dic_point_border['right_points'])

    if display == True:
        show_point_cloud(point_cloud=point_cloud)
        show_point_cloud(point_cloud=projected_point_cloud)
        show_point_cloud(point_cloud=lines)
        show_point_cloud(point_cloud=point_cloud_mapped_on_lines)
        show_point_cloud(point_cloud=[*lines, point_cloud_mapped_on_lines])
        show_point_cloud(point_cloud=list_point_mapped_on_lines)
        show_point_cloud(point_cloud=dic_point_border['border_point_cloud'], marker='o')
        show_point_cloud(point_cloud=[point_cloud_mapped_on_lines, dic_point_border['border_point_cloud']], marker='o')
        show_point_cloud(point_cloud=[edges_points['left_lower_points'], edges_points['left_upper_points'], edges_points['right_lower_points'], edges_points['right_upper_points']], marker='o')

    plt.show()


if __name__ == '__main__':

    # generate an plane (point cloud)
    output_dic = generate_a_lidar_plane_in_3D(
                                    rotation_vector=np.array([45.0, 0.0, 0.0]), 
                                    translation_vector=np.array([2000.0, 0.0, 0.0]),
                                    display=False
                                )

    # find plane equation
    best_ratio_plane = ransac_plane_in_lidar(lidar_point=output_dic['lidar_point_with_noise'])
    
    # find plane (calibration target) edges
    find_edges_of_calibration_target_in_lidar(lidar_points=output_dic['lidar_point_with_noise'], plane_equation=best_ratio_plane['plane_equation'], display=True)
