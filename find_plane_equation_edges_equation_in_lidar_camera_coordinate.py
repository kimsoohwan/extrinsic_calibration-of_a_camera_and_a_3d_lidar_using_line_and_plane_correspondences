import traceback

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from copy import copy

from camera_coordinae_find_edges_equation_calibration_target import \
    find_edge_equation_in_camera_coordinate
from camera_coordinate_find_plane_equation_calibration_target import \
    camera_coordinate_plane_equation_calibration_target
from camera_image_find_edges_of_calibration_target import \
    line_equation_four_edges_calibration_target_in_camera_image
from lidar_find_plane_edges_equations import \
    plane_equation_and_edges_equation_lidar_point_cloud
from read_calibration_file import read_yaml_file


def calculate_plane_equation_edges_equation_in_lidar_camera_coordinate(
    point_cloud,
    maximim_distance_two_consecutive_points_in_ray,
    calibration_data,
    rgb_image,
    num_row,
    num_col,
    square,
    display=False):
    
    # plane equation of calibration target in camera coordinate system
    try:
        image_coordinate_plane_equation = camera_coordinate_plane_equation_calibration_target(
                                rgb_img=rgb_image,
                                num_row=num_row,
                                num_col=num_col,
                                square=square,
                                camera_matrix=calibration_data['camera_matrix'],
                                distortion_coefficients=calibration_data['distortion_coefficients'],
                                display=display
                            )
    except Exception:
        traceback.print_exc()

    
    #  Finds edges equations in camera image 
    lines_equations, image_process_1 = line_equation_four_edges_calibration_target_in_camera_image(
                                            rgb_image=rgb_image, display=display)

    # calculate line equation in  camera coordinate
    lines_equation_camera_coordinate = {}
    for line_name in lines_equations:
        line_equation_camera_coordinate = find_edge_equation_in_camera_coordinate(
                line_equation_image=lines_equations[line_name],
                plane_camera_coordinate=image_coordinate_plane_equation,
                camera_matrix=calibration_data['camera_matrix']
                )
        lines_equation_camera_coordinate[line_name] = copy(line_equation_camera_coordinate)

    # unify edge directions of calibration target inside camera coordinate system
    if lines_equation_camera_coordinate['left_lower_edge_equation'][1][0] > 0:
        lines_equation_camera_coordinate['left_lower_edge_equation'][1] *= -1
    if lines_equation_camera_coordinate['left_upper_edge_equation'][1][0] < 0:
        lines_equation_camera_coordinate['left_upper_edge_equation'][1] *= -1
    if lines_equation_camera_coordinate['right_upper_edge_equation'][1][0] < 0:
        lines_equation_camera_coordinate['right_upper_edge_equation'][1] *= -1
    if lines_equation_camera_coordinate['right_lower_edge_equation'][1][0] > 0:
        lines_equation_camera_coordinate['right_lower_edge_equation'][1] *= -1
    
        

    # find plane and edges equation in LiDAR
    plane_edges_equation, image_process_2 = plane_equation_and_edges_equation_lidar_point_cloud(
        lidar_point_cloud=point_cloud,
        maximim_distance_two_consecutive_points_in_ray=maximim_distance_two_consecutive_points_in_ray,
        display=display
    )

    for key in image_process_2:
        image_process_1.append(image_process_2[key])

    return {'camera_coordinate_plane_equation': image_coordinate_plane_equation,
            'camera_coordinate_edges_equation': lines_equation_camera_coordinate,
            'lidar_plane_equation': plane_edges_equation['plane_equation'],
            'lidar_plane_centroid': plane_edges_equation['plane_centroid'],
            'lidar_edges_equation': plane_edges_equation['edges_equation'],
            'lidar_edges_centroid': plane_edges_equation['edges_centroid'],
            'lidar_denoised_edges_points': plane_edges_equation['denoised_edges_points'],
            'lidar_denoised_plane_points': plane_edges_equation['denoised_plane_points'],
            'image_process': image_process_1,
            'description': 'plane equation: ax+by+cz+d=0, each line equation: p0 a point on line and t the direction vector'}

if __name__ == '__main__':
    
    ################################################################
    # Read image
    ################################################################
    # read image
    img_bgr = cv.imread('/home/farhad-bat/code/find_normal_vector_plane_pointcloud/example_real_img_lidar_points/frame-1.png')

    # convert BGR to RGB
    rgb_image = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)

    ################################################################
    # Read Point Cloud
    ################################################################
    point_cloud = np.load('example_real_img_lidar_points/selected_points_in_lidar-1.npy')
    # convert to mm
    point_cloud *= 1000

    ################################################################
    # calibration information related to camera
    ################################################################
    path = '/home/farhad-bat/code/find_normal_vector_plane_pointcloud/example_real_img_lidar_points/left_camera_calibration_parameters.yaml'
    calibration_data = read_yaml_file(path=path)
    print("Calibration Parameters:\n", calibration_data)

    ################################################################
    # Calculate plane equation and edges equations for calibration
    # target inside lidar and camera coordinate system
    ################################################################
    plane_edges_equations_in_lidar_camera_coordinate = calculate_plane_equation_edges_equation_in_lidar_camera_coordinate(
                                                            point_cloud=point_cloud,
                                                            maximim_distance_two_consecutive_points_in_ray=100,
                                                            calibration_data=calibration_data,
                                                            rgb_image=rgb_image,
                                                            num_row=6,
                                                            num_col=8,
                                                            square=152,
                                                            display=False
                                                        )

    for key in plane_edges_equations_in_lidar_camera_coordinate:
        if key != 'image_process':
            print('>    {}'.format(key))
            print(plane_edges_equations_in_lidar_camera_coordinate[key])

    for img in plane_edges_equations_in_lidar_camera_coordinate['image_process']:
        plt.figure()
        plt.imshow(img)
    plt.show()
