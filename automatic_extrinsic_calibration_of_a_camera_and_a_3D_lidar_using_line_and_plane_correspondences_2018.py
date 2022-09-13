import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import math

from find_plane_equation_edges_equation_in_lidar_camera_coordinate import \
    calculate_plane_equation_edges_equation_in_lidar_camera_coordinate
from read_calibration_file import read_yaml_file
from utils_display import get_img_from_fig

def is_rotation_matrix(R) :
    """
    Checks if a matrix is a valid rotation matrix.
    """
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


def rotation_matrix_to_euler_angles(R):
    """
    Calculates rotation matrix to euler angles
    The result is the same as MATLAB except the order
    of the euler angles ( x and z are swapped ).
    """
    assert(is_rotation_matrix(R))

    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])

    singular = sy < 1e-6

    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0

    return np.array([x, y, z])

def init_estimate_R_one_pose(lidar_plane_equation, lidar_edges_equation, camera_coordinate_plane_equation, camera_coordinate_edges_equation):
    n_l = lidar_plane_equation.tolist()[0:3]
    l1_l =  lidar_edges_equation['line_equation_left_lower'][1].tolist()
    l2_l =  lidar_edges_equation['line_equation_left_upper'][1].tolist()
    l3_l =  lidar_edges_equation['line_equation_right_upper'][1].tolist()
    l4_l =  lidar_edges_equation['line_equation_right_lower'][1].tolist()
    
    m_l = np.stack((n_l, l1_l, l2_l, l3_l, l4_l)).T
    
    n_c = camera_coordinate_plane_equation.tolist()[0:3]
    l1_c =  camera_coordinate_edges_equation['left_lower_edge_equation'][1].tolist()
    l2_c =  camera_coordinate_edges_equation['left_upper_edge_equation'][1].tolist()
    l3_c =  camera_coordinate_edges_equation['right_upper_edge_equation'][1].tolist()
    l4_c =  camera_coordinate_edges_equation['right_lower_edge_equation'][1].tolist()

    m_c = np.stack((n_c, l1_c, l2_c, l3_c, l4_c)).T

    m = np.dot(m_l, m_c.T)

    u, s, v_t = np.linalg.svd(a=m)

    r_estimate_matrix = np.dot(v_t.T, u.T)
    r_estimate_vector_radian = rotation_matrix_to_euler_angles(r_estimate_matrix)
    r_estimate_vector_degree = r_estimate_vector_radian * (180/np.math.pi)

    return {'rotation_matrix': r_estimate_matrix, 'rotation_vector_radian':r_estimate_vector_radian, 'rotation_vector_degree':r_estimate_vector_degree}

def calculate_A(line_direction):
    # direction of line
    line_direction = np.copy(line_direction)
    line_direction = np.reshape(line_direction, newshape=(-1, 1))
    if line_direction.shape[0] != 3:
        raise ValueError('The shape of direction vector for line equation is not correct.')

    matrix_a = np.identity(n=3) - np.dot(line_direction, line_direction.T)

    return matrix_a

def init_estimate_t_one_pose(camera_coordinate_plane_equation, camera_coordinate_edges_equation,
                             estimated_rotation_matrix, lidar_plane_centroid, lidar_edges_centroid):
    
    # normal vector and d of plane in camera coordinate (ax+by+cz+d=0)
    n_c = camera_coordinate_plane_equation.tolist()[0:3]
    d_c = camera_coordinate_plane_equation.tolist()[3]
    
    # points on calibration target edges
    p1_c =  camera_coordinate_edges_equation['left_lower_edge_equation'][0].tolist()
    p2_c =  camera_coordinate_edges_equation['left_upper_edge_equation'][0].tolist()
    p3_c =  camera_coordinate_edges_equation['right_upper_edge_equation'][0].tolist()
    p4_c =  camera_coordinate_edges_equation['right_lower_edge_equation'][0].tolist()

    # direction of calibration target edges
    l1_c =  camera_coordinate_edges_equation['left_lower_edge_equation'][1].tolist()
    l2_c =  camera_coordinate_edges_equation['left_upper_edge_equation'][1].tolist()
    l3_c =  camera_coordinate_edges_equation['right_upper_edge_equation'][1].tolist()
    l4_c =  camera_coordinate_edges_equation['right_lower_edge_equation'][1].tolist()

    centroid_l1_c = lidar_edges_centroid['left_lower_points'].tolist()
    centroid_l2_c = lidar_edges_centroid['left_upper_points'].tolist()
    centroid_l3_c = lidar_edges_centroid['right_upper_points'].tolist()
    centroid_l4_c = lidar_edges_centroid['right_lower_points'].tolist()

    # A = I-d.d
    matrix_A_1 = calculate_A(line_direction=l1_c)
    matrix_A_2 = calculate_A(line_direction=l2_c)
    matrix_A_3 = calculate_A(line_direction=l3_c)
    matrix_A_4 = calculate_A(line_direction=l4_c)

    # convert matrixes to proper size
    n_c = np.reshape(n_c, newshape=(-1, 1))
    d_c = np.reshape(d_c, newshape=(-1, 1))
    p1_c = np.reshape(p1_c, newshape=(-1, 1))
    p2_c = np.reshape(p2_c, newshape=(-1, 1))
    p3_c = np.reshape(p3_c, newshape=(-1, 1))
    p4_c = np.reshape(p4_c, newshape=(-1, 1))
    lidar_plane_centroid = np.reshape(lidar_plane_centroid, newshape=(-1, 1))
    centroid_l1_c = np.reshape(centroid_l1_c, newshape=(-1, 1))
    centroid_l2_c = np.reshape(centroid_l2_c, newshape=(-1, 1))
    centroid_l3_c = np.reshape(centroid_l3_c, newshape=(-1, 1))
    centroid_l4_c = np.reshape(centroid_l4_c, newshape=(-1, 1))

    # create linear system, euqation Matrix_left * t= Vector_right 
    matrix_left = np.vstack((n_c.T, matrix_A_1, matrix_A_2, matrix_A_3, matrix_A_4))

    matrix_right = np.zeros(shape=(matrix_left.shape[0], 1))
    matrix_right[0, 0] = -np.dot(n_c.T, np.dot(estimated_rotation_matrix, lidar_plane_centroid)) - d_c
    matrix_right[1:4] = -np.dot(matrix_A_1, np.dot(estimated_rotation_matrix, centroid_l1_c) - p1_c)
    matrix_right[4:7] = -np.dot(matrix_A_2, np.dot(estimated_rotation_matrix, centroid_l2_c) - p2_c)
    matrix_right[7:10] = -np.dot(matrix_A_3, np.dot(estimated_rotation_matrix, centroid_l3_c) - p3_c)
    matrix_right[10:13] = -np.dot(matrix_A_4, np.dot(estimated_rotation_matrix, centroid_l4_c) - p4_c) 

    # solve least squre problem
    estimated_t = np.dot(np.dot(np.linalg.inv(np.dot(matrix_left.T, matrix_left)), matrix_left.T), matrix_right)

    return estimated_t

def automatic_extrinsic_calibration_of_a_camera_and_a_3D_lidar_using_line_and_plane_correspondences(
    calibration_data,
    camera_coordinate_plane_equation,
    camera_coordinate_edges_equation,
    lidar_plane_equation,
    lidar_plane_centroid,
    lidar_edges_equation,
    lidar_edges_centroid
):
    
    # initial estimate for R
    estimated_rotation_results = init_estimate_R_one_pose(
                                        lidar_plane_equation=lidar_plane_equation,
                                        lidar_edges_equation=lidar_edges_equation,
                                        camera_coordinate_plane_equation=camera_coordinate_plane_equation,
                                        camera_coordinate_edges_equation=camera_coordinate_edges_equation
                                        )
    estimated_rotation_matrix = estimated_rotation_results['rotation_matrix']

    # initial estimate for t
    estimated_translation= init_estimate_t_one_pose(
        camera_coordinate_plane_equation=camera_coordinate_plane_equation,
        camera_coordinate_edges_equation=camera_coordinate_edges_equation,
        estimated_rotation_matrix=estimated_rotation_matrix, 
        lidar_plane_centroid=lidar_plane_centroid, 
        lidar_edges_centroid=lidar_edges_centroid
        )

    print('Estimated Rotation Matrix:')
    print(estimated_rotation_matrix)
    print('Estimated Translation Matrix:')
    print(estimated_translation)

    return estimated_rotation_matrix, estimated_translation

def lidar_points_in_image(
    rgb_image, 
    point_cloud,
    calibration_data,
    r_lidar_to_camera_coordinate,
    t_lidar_to_camera_coordinate, 
):

    # keep points that are in front of LiDAR
    points_in_front_of_lidar = []
    for point_i in point_cloud:
        if point_i[0] >= 0:
            points_in_front_of_lidar.append(point_i)
    point_cloud = np.array(points_in_front_of_lidar)
    
    # translate lidar points to camera coordinate system
    points_in_camera_coordinate = np.dot(r_lidar_to_camera_coordinate, point_cloud.T) + t_lidar_to_camera_coordinate

    # project points form camera coordinate to image
    points_in_image = np.dot(calibration_data['camera_matrix'], points_in_camera_coordinate)
    points_in_image = points_in_image / points_in_image[2, :]
    points_in_image = points_in_image[0:2, :]
    points_in_image = points_in_image.T

    # keep points that are inside image
    points_inside_image = []
    for point_i in points_in_image:
        if (0 <= point_i[0] < rgb_image.shape[1]) and (0 <= point_i[1] < rgb_image.shape[0]):
            points_inside_image.append(point_i)
    points_in_image = np.array(points_inside_image)

    fig = plt.figure()
    plt.imshow(rgb_image)
    plt.scatter(points_in_image[:, 0].tolist(), points_in_image[:, 1].tolist(), s=3)
    ax = plt.gca()
    ax.axes.xaxis.set_ticks([])
    ax.axes.yaxis.set_ticks([])
    img_lidar_points = get_img_from_fig(fig=fig, dpi=500)

    return points_in_image, img_lidar_points
    

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
    
    # point cloud of calibration target
    point_cloud_target = np.load('example_real_img_lidar_points/selected_points_in_lidar-1.npy')
    # convert to mm
    point_cloud_target *= 1000

    # point cloud of whole scence
    point_cloud_scene = np.load('example_real_img_lidar_points/selected_points_in_lidar-1_whole_scene.npy')
    # convert to mm
    point_cloud_scene *= 1000

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
                                                            point_cloud=point_cloud_target,
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

    #for img in plane_edges_equations_in_lidar_camera_coordinate['image_process']:
    #    plt.figure()
    #    plt.imshow(img)
    #plt.show()

    ###################################################################
    #       Calculate R and t
    ###################################################################
    r_lidar_to_camera_coordinate, t_lidar_to_camera_coordinate = automatic_extrinsic_calibration_of_a_camera_and_a_3D_lidar_using_line_and_plane_correspondences(
        calibration_data=calibration_data,
        camera_coordinate_plane_equation=plane_edges_equations_in_lidar_camera_coordinate['camera_coordinate_plane_equation'],
        camera_coordinate_edges_equation=plane_edges_equations_in_lidar_camera_coordinate['camera_coordinate_edges_equation'],
        lidar_plane_equation=plane_edges_equations_in_lidar_camera_coordinate['lidar_plane_equation'],
        lidar_plane_centroid=plane_edges_equations_in_lidar_camera_coordinate['lidar_plane_centroid'],
        lidar_edges_equation=plane_edges_equations_in_lidar_camera_coordinate['lidar_edges_equation'],
        lidar_edges_centroid=plane_edges_equations_in_lidar_camera_coordinate['lidar_edges_centroid']
    )

    # point clould points of calibrariotion target on image
    points_in_image, img_lidar_points = lidar_points_in_image(
        rgb_image=rgb_image,
        point_cloud=point_cloud_target,
        calibration_data=calibration_data,
        r_lidar_to_camera_coordinate=r_lidar_to_camera_coordinate,
        t_lidar_to_camera_coordinate=t_lidar_to_camera_coordinate
    )
    
    # point clould points of whole scene on image
    points_in_image, img_lidar_points = lidar_points_in_image(
        rgb_image=rgb_image,
        point_cloud=point_cloud_scene,
        calibration_data=calibration_data,
        r_lidar_to_camera_coordinate=r_lidar_to_camera_coordinate,
        t_lidar_to_camera_coordinate=t_lidar_to_camera_coordinate
    )

    plt.figure()
    plt.imshow(img_lidar_points)
    plt.show()

