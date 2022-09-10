import matplotlib.pyplot as plt
import numpy as np

from utils_display import show_point_cloud


def calculate_line_equition(two_points):
    # calculate line equation x = x0 + kt
    t = two_points[1, :] - two_points[0, :]
    x0 = two_points[0, :] 
    
    t = t / np.linalg.norm(t)
    
    return [x0, t]

def distance_of_points_to_line(point_cloud, line_eqiotion):
    # distance of points in point cloud to the line
    distance_points_to_line = []

    for point in point_cloud:

        vec = line_eqiotion[0] - point
        cross_product = np.cross(vec, line_eqiotion[1])

        distance_point_line = np.linalg.norm(cross_product) / np.linalg.norm(line_eqiotion[1]) 

        distance_points_to_line.append(distance_point_line)

    distance_points_to_line = np.reshape(distance_points_to_line, newshape=(-1, 1))

    return distance_points_to_line


def map_point_to_line(point_cloud, line_eqiotion):
    # distance of points in point cloud to the line
    points_on_line = []

    for point in point_cloud:

        vec_1 = point - line_eqiotion[0]
        point_on_line = line_eqiotion[0] + np.dot(vec_1, line_eqiotion[1])/np.dot(line_eqiotion[1], line_eqiotion[1].T) * line_eqiotion[1]

        points_on_line.append(point_on_line)

    points_on_line = np.array(points_on_line)

    return points_on_line

def find_inliers(distance_points_to_line, distance_to_be_inlier):
    # find inliers
    inliers_index = np.argwhere(distance_points_to_line <= distance_to_be_inlier)
    if inliers_index.shape[1] > 1:
        inliers_index = inliers_index[:, 0]
    inliers_index = np.reshape(inliers_index, newshape=(-1))

    return inliers_index

def ransac_line_in_lidar(lidar_point, maximum_iteration=8000, inlier_ratio=0.9, distance_to_be_inlier=10):
    """
    lidar_point: numpy array with shape of (n, 3), all measurements are in mm.
    maximum_iteration: maximum iteration before halting the program.
    inlier_ratio: it will stop algorithm if the 90% or more of data in point cloud considered as inliers. 
    distance_to_be_inlier: if a point has a distance equal or less than this value, it will considered as inliers.
    """
    
    point_cloud_orginal = np.copy(lidar_point)
    
    best_ratio_line = [0, None]

    for _ in range(maximum_iteration):
        
        # randomly select three points
        two_index = np.random.choice([idx for idx in range(point_cloud_orginal.shape[0])], size=2, replace=False)
        two_points = point_cloud_orginal[two_index]

        # calculate line equation x = x0 + kt
        line_eqiotion = calculate_line_equition(two_points=two_points)

        # distance of points in point cloud to the line
        distance_points_to_line_all_set = distance_of_points_to_line(point_cloud=point_cloud_orginal, line_eqiotion=line_eqiotion)

        # find inliers
        inliers_index_all_set = find_inliers(distance_points_to_line=distance_points_to_line_all_set, distance_to_be_inlier=distance_to_be_inlier)

        # find inliers ratio
        inlier_to_all_points_all_set = inliers_index_all_set.shape[0]/distance_points_to_line_all_set.shape[0]

        if inlier_to_all_points_all_set > best_ratio_line[0]:
            best_ratio_line[0] = inlier_to_all_points_all_set
            best_ratio_line[1] = line_eqiotion

            if inlier_ratio <= inlier_to_all_points_all_set:
                break

    return {'inlier_to_all_data_ratio':best_ratio_line[0], 'line_equation':best_ratio_line[1]}

if __name__ == '__main__':

    # generate a line (point cloud)
    t = np.random.randint(low=0, high=100, size=3)
    t = np.reshape(t, newshape=(1, 3))
    t = t / np.linalg.norm(t)
    x0 = np.random.randint(low=200, high=400, size=3)
    x0 = np.reshape(x0, newshape=(1, 3))
    
    point_cloud = []
    for step in np.linspace(start=-200, stop=200, num=50):
        point = x0 + step * t
        point = point[0]
        point_cloud.append(point)

    point_cloud = np.array(point_cloud)
    point_cloud = point_cloud + np.random.uniform(low=-4, high=4, size=point_cloud.shape)

    

    # call function to calculate line equation
    best_ratio_line = ransac_line_in_lidar(lidar_point=point_cloud)
    
    print('Ground Truth Line:')
    print('x0: {}, t: {}'.format(x0, t))
    print('Calculated line Equation:')
    print(best_ratio_line)

    point_cloud2 = []
    for step in np.linspace(start=-200, stop=200, num=50):
        point = best_ratio_line['line_equation'][0] + step * best_ratio_line['line_equation'][1]
        point_cloud2.append(point)
    point_cloud2 = np.array(point_cloud2)
    
    point_cloud3 = map_point_to_line(point_cloud, best_ratio_line['line_equation'])

    show_point_cloud([point_cloud, point_cloud2])
    show_point_cloud(point_cloud3)
    show_point_cloud([point_cloud, point_cloud3])
    

    plt.show()
