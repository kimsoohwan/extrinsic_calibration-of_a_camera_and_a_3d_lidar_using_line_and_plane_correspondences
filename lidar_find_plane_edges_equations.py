import matplotlib.pyplot as plt
import numpy as np

from lidar_find_plane import ransac_plane_in_lidar
from lidar_find_target_edges_in_point_cloud import \
    find_edges_of_calibration_target_in_lidar
from lidar_generate_a_plane import generate_a_lidar_plane_in_3D


def plane_equation_and_edges_equation_lidar_point_cloud(lidar_point_cloud, maximim_distance_two_consecutive_points_in_ray=100, display=False):
    # find plane equation
    best_ratio_plane = ransac_plane_in_lidar(lidar_point=lidar_point_cloud)
    
    # find plane's (calibration target) edges equation
    dic_line_equations, denoised_plane_centroid, dic_edges_centroid, images_edges_process = find_edges_of_calibration_target_in_lidar(
                                lidar_points=lidar_point_cloud,
                                plane_equation=best_ratio_plane['plane_equation'],
                                display=display,
                                maximim_distance_two_consecutive_points_in_ray=maximim_distance_two_consecutive_points_in_ray)

    description = 'plane equation: ax+by+cz+d=0, each line equation: p0 a pont on line and t the direction vector'
    return {'plane_equation': best_ratio_plane['plane_equation'],
            'plane_centroid': denoised_plane_centroid,
            'edges_equation':dic_line_equations, 
            'edges_centroid': dic_edges_centroid, 
            'description':description}, images_edges_process

if __name__ == '__main__':

    ########################################
    # generated data example
    ########################################
    # generate an plane (point cloud)
    #output_dic = generate_a_lidar_plane_in_3D(
    #                                rotation_vector=np.array([45.0, 0.0, 0.0]), 
    #                                translation_vector=np.array([8000.0, 0.0, 0.0]),
    #                                display=False
    #                            )

    # find plane and edges equation
    #plane_edges_equation = plane_equation_and_edges_equation_lidar_point_cloud(lidar_point_cloud=output_dic['lidar_point_with_noise'],
    #                                                                           maximim_distance_two_consecutive_points_in_ray=100,
    #                                                                           display=True)

    #print('Plane and Edges equations:')
    #print(plane_edges_equation)

    #######################################
    # Real Data example 1
    #######################################
    point_cloud = np.load('example_real_img_lidar_points/selected_points_in_lidar-1.npy')
    # convert to mm
    point_cloud *= 1000

    # find plane and edges equation
    plane_edges_equation, images_edges_process = plane_equation_and_edges_equation_lidar_point_cloud(lidar_point_cloud=point_cloud,
                                                                               maximim_distance_two_consecutive_points_in_ray=100,
                                                                               display=True)
    
    print('Plane and Edges equations:')
    print(plane_edges_equation)

    for key_i in images_edges_process:
        plt.figure()
        plt.imshow(images_edges_process[key_i])
    plt.show()

    #######################################
    # Real Data example 2
    #######################################
    #point_cloud = np.load('example_real_img_lidar_points/selected_points_in_lidar-2.npy')
    # convert to mm
    #point_cloud *= 1000

    # find plane and edges equation
    #plane_edges_equation = plane_equation_and_edges_equation_lidar_point_cloud(lidar_point_cloud=point_cloud,
    #                                                                           maximim_distance_two_consecutive_points_in_ray=100,
    #                                                                           display=True)
    
    #print('Plane and Edges equations:')
    #print(plane_edges_equation)
