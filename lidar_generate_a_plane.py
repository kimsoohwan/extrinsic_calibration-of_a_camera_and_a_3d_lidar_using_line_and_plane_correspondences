import warnings

import matplotlib.pyplot as plt
import numpy as np

from utils_display import show_point_cloud

warnings.filterwarnings("ignore")

# intersection function
def isect_line_plane_v3(p0, p1, p_co, p_no, epsilon=1e-6):
    """
    p0, p1: Define the line.
    p_co, p_no: define the plane:
        p_co Is a point on the plane (plane coordinate).
        p_no Is a normal vector defining the plane direction;
             (does not need to be normalized).

    Return a Vector or None (when the intersection can't be found).
    """

    u = sub_v3v3(p1, p0)
    dot = dot_v3v3(p_no, u)

    if abs(dot) > epsilon:
        # The factor of the point between p0 -> p1 (0 - 1)
        # if 'fac' is between (0 - 1) the point intersects with the segment.
        # Otherwise:
        #  < 0.0: behind p0.
        #  > 1.0: infront of p1.
        w = sub_v3v3(p0, p_co)
        fac = -dot_v3v3(p_no, w) / dot
        u = mul_v3_fl(u, fac)
        return add_v3v3(p0, u)

    # The segment is parallel to plane.
    return None

def add_v3v3(v0, v1):
    return (
        v0[0] + v1[0],
        v0[1] + v1[1],
        v0[2] + v1[2],
    )


def sub_v3v3(v0, v1):
    return (
        v0[0] - v1[0],
        v0[1] - v1[1],
        v0[2] - v1[2],
    )


def dot_v3v3(v0, v1):
    return (
        (v0[0] * v1[0]) +
        (v0[1] * v1[1]) +
        (v0[2] * v1[2])
    )


def len_squared_v3(v0):
    return dot_v3v3(v0, v0)


def mul_v3_fl(v0, f):
    return (
        v0[0] * f,
        v0[1] * f,
        v0[2] * f,
    )

def distance_two_points(point1, point2):
    return np.math.sqrt(np.sum((point1-point2)**2))

def generate_a_lidar_plane_in_3D(
        target_width=1000, 
        target_height=1000,  
        translation_vector=np.array([0.0, 0.0, 0.0]),
        rotation_vector=np.array([0.0, 0.0, 0.0]),
        liadar_range=100000,
        horizontal_resolution=0.2,
        vertical_resolution=2,
        range_accuracy=30,
        horizontal_field_of_view = 360,
        vertical_field_of_view = 30,
        display=False
    ):
    """
    target_width: Calibration width in 3D scence (mm)
    target_height: Calibration height in 3D scence (mm)
    max_number_of_ray: maximum number of lidar ray that hit the target
    hor_distance: horizontal distance between two consequitive points on a ray 
    lidar_noise_type_x_dir: noise type
    lidar_uniform_error_x_dir: error for uniform distribution
    lidar_noise_type_y_dir: noise type
    lidar_uniform_error_y_dir: error for uniform distribution
    lidar_noise_type_z_dir: noise type
    lidar_uniform_error_z_dir: error for uniform distribution
    translation_vector: translate target in 3D world from the origin
    rotation_vector: rotate calibration target along x, y, and z axis (clockwise rotation, degree)
    lidar_range: lidar maximum range
    horizontal_resolution: in degree degree, usually between 0.1 degree and 0.4 degree
    vertical_resolution: in degree degree, usually between 2 degree and 3 degree
    range_accuracy: accuracy in x direction in mm
    horizontal_field_of_view: in degree
    vertical_field_of_view: in degree

    coordinate systerm: Coordinate System: X Forward, Y Left, Z Up
    width is in Z direction and height is in Y direction
    """

    plt_images = {}

    # corners of target before translating and rotating
    target_init_corners = np.array([[0, target_width/2.0, target_height/2.0], [0, -target_width/2.0, target_height/2.0], [0, -target_width/2.0, -target_height/2.0], [0, target_width/2.0, -target_height/2.0]])
    
    # normal vector of plane
    plane_normal = np.array([1, 0, 0])
    
    # display plane (calibration target)
    if display:
        plt_img = show_point_cloud(point_cloud=target_init_corners, normal_vector=plane_normal, title='Target at origin')
        plt_images['calibration_target_orgin'] = np.copy(plt_img)

    # get rotation marix
    rotation_matrix = get_rotation_matrix(rotation_vector=rotation_vector)
    
    # rotate target by rotation angels
    target_rotated_corners = np.dot(rotation_matrix, target_init_corners.T).T
    
    # rotate normal vector of plane (calibration target)
    plane_normal = np.dot(rotation_matrix, plane_normal.T).T
    plane_normal /= np.linalg.norm(plane_normal)

    if display:
        plt_img = show_point_cloud(point_cloud=target_rotated_corners, normal_vector=plane_normal, title='Target after rotaion: {}'.format(rotation_vector))
        plt_images['calibration_target_rotated'] = np.copy(plt_img)

    # transform the target by tarnsform vector
    target_rotated_and_translated_corners = target_rotated_corners + translation_vector
    if display:
        plt_img = show_point_cloud(point_cloud=target_rotated_and_translated_corners, normal_vector=plane_normal, title='Target after translation: {}'.format(translation_vector))
        plt_images['calibration_target_rotate_and_translated'] = np.copy(plt_img)

    d = -1 * (plane_normal[0] * target_rotated_and_translated_corners[0, 0] + plane_normal[1] * target_rotated_and_translated_corners[0, 1] + plane_normal[2] * target_rotated_and_translated_corners[0, 2])
    plane_eqiotion = np.array([plane_normal[0], plane_normal[1], plane_normal[2], d])

    # all points on the palne (calibration target)
    all_intersection = []
    all_noisy_intersection = []

    # calculate intersection of LiDAR rays to plane (calibation target)
    all_vertical_range = [-vertical_field_of_view/2 + vertical_resolution * v_r for v_r in range(int(np.math.ceil(vertical_field_of_view/vertical_resolution))+1)]
    all_horizontal_range = [-horizontal_field_of_view/2 + horizontal_resolution * v_r for v_r in range(int(np.math.ceil(horizontal_field_of_view/horizontal_resolution))+1)]
    
    for z_range in all_vertical_range:
        for y_range in all_horizontal_range:
            
            if y_range < 0:
                continue

            # start of ray: lidar position
            start_ray = np.array([0, 0, 0])

            # another point of ray
            direction_ray = np.array([1, 0, 0])
            ray_rotation_matrix = get_rotation_matrix(rotation_vector=np.array([0, z_range, y_range-90]))
            direction_ray = np.dot(ray_rotation_matrix, direction_ray.T).T            
            another_ray = start_ray + direction_ray

            intersection_pos = isect_line_plane_v3(p0=start_ray, 
                                                   p1=another_ray,
                                                   p_co=target_rotated_and_translated_corners[0, :],
                                                   p_no=plane_normal,
                                                   epsilon=1e-6)

            if intersection_pos is not None:
                # check intersection is on plane (calibration target)
                if np.dot(target_rotated_and_translated_corners[1, :]-target_rotated_and_translated_corners[0, :], intersection_pos-target_rotated_and_translated_corners[0, :]) >= 0:
                    if np.dot(target_rotated_and_translated_corners[2, :]-target_rotated_and_translated_corners[1, :], intersection_pos-target_rotated_and_translated_corners[1, :]) >= 0:
                        if np.dot(target_rotated_and_translated_corners[3, :]-target_rotated_and_translated_corners[2, :], intersection_pos-target_rotated_and_translated_corners[2, :]) >= 0:
                            if np.dot(target_rotated_and_translated_corners[0, :]-target_rotated_and_translated_corners[3, :], intersection_pos-target_rotated_and_translated_corners[3, :]) >= 0:
                                all_intersection.append(intersection_pos)

                                # add noise to the intersection
                                lidar_intersection_line = intersection_pos - start_ray
                                lidar_intersection_line /= np.linalg.norm(lidar_intersection_line)
                                intersection_pos_noisy = np.random.choice([-1, 1]) * lidar_intersection_line * np.random.uniform(low=-range_accuracy, high=range_accuracy) + intersection_pos

                                all_noisy_intersection.append(intersection_pos_noisy)

    # reshape intersection points
    if len(all_intersection) != 0 and display:
        all_intersection = np.array(all_intersection)
        all_intersection = np.reshape(all_intersection, newshape=(-1, 3))

        all_noisy_intersection = np.array(all_noisy_intersection)
        all_noisy_intersection = np.reshape(all_noisy_intersection, newshape=(-1, 3))

        if display == True:
            plt_img = show_point_cloud(point_cloud=target_rotated_and_translated_corners, normal_vector=plane_normal, intersection_points=all_intersection, title='Intersection of LiDAR and Target, No Noise')
            plt_images['intersection_rays_calibration_target_no_noise'] = np.copy(plt_img)
            plt_img = show_point_cloud(point_cloud=target_rotated_and_translated_corners, normal_vector=plane_normal, intersection_points=all_noisy_intersection, title='Intersection of LiDAR and Target, With Noise')
            plt_images['intersection_rays_calibration_target_noise'] = np.copy(plt_img)

    

    return {'calibration_target_corners':target_rotated_and_translated_corners,
            'normal_plane': plane_normal,
            'plane_equation': plane_eqiotion,
            'lidar_point_without_noise':all_intersection, 
            'lidar_point_with_noise':all_noisy_intersection}, plt_images

if __name__ == '__main__':

    output_dic, plt_images = generate_a_lidar_plane_in_3D(
                                    rotation_vector=np.array([45.0, 0.0, 0.0]), 
                                    translation_vector=np.array([5000.0, 0.0, 0.0]),
                                    display=True
                                )

    print('Generated calibration target and point cloud:')
    print(output_dic)

    for key_i in plt_images:
        plt.figure()
        plt.imshow(plt_images[key_i])
    plt.show()
