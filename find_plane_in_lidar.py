import numpy as np
from generate_a_plane import generate_a_lidar_plane_in_3D

def calculate_plane_equition(three_points):
    # calculate plane equition ax+by+cz+d = 0
    vec_1 = three_points[1, :] - three_points[0, :]
    vec_2 = three_points[2, :] - three_points[0, :] 
    normal = np.cross(vec_1, vec_2)
    if normal[0] < 0:
        normal *= -1
    normal /= np.linalg.norm(normal)
    d = -1 * (normal[0] * three_points[0, 0] + normal[1] * three_points[0, 1] + normal[2] * three_points[0, 2])
    plane_eqiotion = np.array([normal[0], normal[1], normal[2], d])

    return plane_eqiotion

def distance_of_points_to_plane(point_cloud, plane_eqiotion):
    # distance of points in point cloud to the plane
    point_cloud_with_one = np.hstack((point_cloud, np.ones(shape=(point_cloud.shape[0], 1))))
    distance_points_to_plane = np.abs(np.dot(point_cloud_with_one, plane_eqiotion.T))

    return distance_points_to_plane

def find_inliers(distance_points_to_plane, distance_to_be_inlier):
    # find inliers
    inliers_index = np.argwhere(distance_points_to_plane <= distance_to_be_inlier)
    inliers_index = np.reshape(inliers_index, newshape=(-1))

    return inliers_index

def ransac_plane_in_lidar(lidar_point, maximum_iteration=50000, inlier_ratio=0.9, distance_to_be_inlier=10):

    point_cloud_orginal = np.copy(lidar_point)
    
    best_ratio_plane = [0, None]

    for _ in range(maximum_iteration):
        
        # randomly select three points
        three_index = np.random.choice([idx for idx in range(point_cloud_orginal.shape[0])], size=3, replace=False)
        three_points = point_cloud_orginal[three_index]

        # calculate plane equation ax+by+cz+d = 0
        plane_eqiotion = calculate_plane_equition(three_points=three_points)

        # distance of points in point cloud to the plane
        distance_points_to_plane_all_set = distance_of_points_to_plane(point_cloud=point_cloud_orginal, plane_eqiotion=plane_eqiotion)

        # find inliers
        inliers_index_all_set = find_inliers(distance_points_to_plane=distance_points_to_plane_all_set, distance_to_be_inlier=distance_to_be_inlier)

        # find inliers ratio
        inlier_to_all_points_all_set = inliers_index_all_set.shape[0]/distance_points_to_plane_all_set.shape[0]

        if inlier_to_all_points_all_set > best_ratio_plane[0]:
            best_ratio_plane[0] = inlier_to_all_points_all_set
            best_ratio_plane[1] = plane_eqiotion

            if inlier_ratio <= inlier_to_all_points_all_set:
                break

    return {'inlier_to_all_data_ratio':best_ratio_plane[0], 'plane_equation':best_ratio_plane[1]}

if __name__ == '__main__':

    output_dic = generate_a_lidar_plane_in_3D(
                                    rotation_vector=np.array([45.0, 0.0, 0.0]), 
                                    translation_vector=np.array([5000.0, 0.0, 0.0]),
                                    display=False
                                )

    best_ratio_plane = ransac_plane_in_lidar(lidar_point=output_dic['lidar_point_with_noise'])
    print(best_ratio_plane)