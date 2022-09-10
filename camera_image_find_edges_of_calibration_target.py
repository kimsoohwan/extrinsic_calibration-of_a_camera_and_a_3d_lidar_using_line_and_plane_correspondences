from copy import copy

import cv2
import matplotlib.pyplot as plt
import numpy as np

from camera_image_find_line_equation import ransac_line_in_image
from utils_display import get_img_from_fig


def segment_yellow_color(img):
    """
    img: it gets an image in HSV format
    """    
    # yellow
    frame_threshold = cv2.inRange(img, (20, 100, 100), (30, 255, 255))

    return frame_threshold

def find_biggest_connected_component(img):

    # Apply the Component analysis function
    analysis = cv2.connectedComponentsWithStats(img, 4, cv2.CV_32S)
    (totalLabels, label_ids, values, centroid) = analysis
    
    max_id = None
    max_id_members = 0

    for i in range(1, totalLabels):
        num_pix = np.sum(label_ids==i)

        if num_pix > max_id_members:
            max_id_members = num_pix
            max_id = i

    biggest_component = np.where(label_ids == max_id, 1, 0)

    return biggest_component


def find_points_on_edges(img):

    # find smallest and biggest row number for pixels with value 1
    index = np.argwhere(img==1)
    
    min_row, min_col = np.min(index, axis=0)
    max_row, max_col = np.max(index, axis=0)

    # point on left edges
    point_on_left_edges = []
    for row_i in range(min_row, max_row):
        for col_i in range(min_col, max_col):
            if img[row_i, col_i] == 1:
                point_on_left_edges.append([row_i, col_i])
                break
    # sort point of left edges according to row number
    point_on_left_edges = sorted(point_on_left_edges, key = lambda x: x[0])

    # point on right edges
    point_on_right_edges = []
    for row_i in range(min_row, max_row):
        for col_i in range(max_col, min_col, -1):
            if img[row_i, col_i] == 1:
                point_on_right_edges.append([row_i, col_i])
                break
    # sort point of right edges according to row number
    point_on_right_edges = sorted(point_on_right_edges, key = lambda x: x[0])
    
    # lower and upper bound of left edge
    left_lower_edge_points = []
    left_upper_edge_points = []
    on_upper = True
    for point in point_on_left_edges:
        if on_upper == False:
            left_lower_edge_points.append(point)
        else:
            left_upper_edge_points.append(point)
        if point[1] == min_col:
            on_upper = False

    # lower and upper bound of right edge
    right_lower_edge_points = []
    right_upper_edge_points = []
    on_upper = True
    for point in point_on_right_edges:
        if on_upper == False:
            right_lower_edge_points.append(point)
        else:
            right_upper_edge_points.append(point)
        if point[1] == max_col:
            on_upper = False

    return {'left_lower_edge_points': left_lower_edge_points, 'left_upper_edge_points': left_upper_edge_points, 
            'right_lower_edge_points': right_lower_edge_points, 'right_upper_edge_points': right_upper_edge_points,}


def points_on_four_edges_calibration_target_camera_image(rgb_image, display=False):
    # convert RGB to HSV
    hsvImage = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
        
    # separate yellow color from other parts
    color_masked_img = segment_yellow_color(img=hsvImage)
        
    # find the biggest connected  component
    bigest_component = find_biggest_connected_component(img=color_masked_img)

    points_on_edges = find_points_on_edges(img=bigest_component)

    img_edges = np.copy(bigest_component)
    img_edges = img_edges / 4
    for key_i in points_on_edges:
        for point in points_on_edges[key_i]:
            img_edges[point[0], point[1]] = 1

    if display == True:
        plt.figure()
        plt.imshow(hsvImage, cmap = 'gray', interpolation = 'bicubic')
        plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis

        plt.figure()
        plt.imshow(color_masked_img, cmap = 'gray', interpolation = 'bicubic')
        plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis

        plt.figure()
        plt.imshow(bigest_component, cmap = 'gray', interpolation = 'bicubic')
        plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis

        plt.figure()
        plt.imshow(img_edges, cmap = 'gray', interpolation = 'bicubic')
        plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis

        plt.show()


    # images during process
    images_process = [hsvImage, color_masked_img, bigest_component, img_edges]

    return points_on_edges, images_process

def line_equation_four_edges_calibration_target_in_camera_image(rgb_image, display=False):
    """
    the return lines equations are in opencv format
    """
    # extract points of four edges
    edges_points, images_process = points_on_four_edges_calibration_target_camera_image(rgb_image=rgb_image, display=display)

    lines_equations = {} 
    for edge_name in edges_points:
        # points on an edge
        points = edges_points[edge_name]

        # convert (row, col) points to (x, y) in OpenCV
        points = np.array(points)
        points[:, [1, 0]] = points[:, [0, 1]]

        # find line eqution (point of line, direction)
        best_ratio_line = ransac_line_in_image(lidar_point=points, 
                                               maximum_iteration=800, 
                                               inlier_ratio=0.9,
                                               distance_to_be_inlier=1)

        line_equation = best_ratio_line['line_equation']

        if edge_name == 'left_lower_edge_points':
            lines_equations['left_lower_edge_equation'] = copy(line_equation)
        elif edge_name == 'left_upper_edge_points':
            lines_equations['left_upper_edge_equation'] = copy(line_equation)
        elif edge_name == 'right_lower_edge_points':
            lines_equations['right_lower_edge_equation'] = copy(line_equation)
        elif edge_name == 'right_upper_edge_points':
            lines_equations['right_upper_edge_equation'] = copy(line_equation)
        else:
            raise ValueError('Name of edge is not correct')

    # generate images forline equations
    fig = plt.figure()
    plt.imshow(rgb_image)
    for line_name in lines_equations:
        point_cloud2 = []
        for step in np.linspace(start=-400, stop=400, num=50):
            point = lines_equations[line_name][0] + step * lines_equations[line_name][1]
            point_cloud2.append(point)

        point_cloud2 = np.array(point_cloud2)
        plt.plot(point_cloud2[:, 0], point_cloud2[:, 1])
        
    numpy_img = get_img_from_fig(fig=fig)
    plt.close(fig)
    images_process.append(numpy_img)

    return lines_equations, images_process

def conver_2d_line_equation_to_homogenous_format(line_equation):
    """
    It gets a 2d line equation in format (point, direction) to  
    homogenous format ax+by+c=0: (a, b, c)
    """
    point_1 = line_equation[0] + 1 * line_equation[1]
    point_2 = line_equation[0] + 2 * line_equation[1]

    m = (point_2[1]-point_1[1]) / (point_2[0]-point_1[0])

    a = -m
    b = 1
    c = m * point_1[0] - point_1[1]

    homogeneous_equation = np.array([[a], [b], [c]])
    homogeneous_equation = homogeneous_equation / np.linalg.norm(homogeneous_equation)

    return homogeneous_equation


if __name__ == '__main__':

    for img_path in ['/home/farhad-bat/code/find_normal_vector_plane_pointcloud/example_real_img_lidar_points/frame-1.png']:
        # read image
        img_bgr = cv2.imread(img_path)

        # convert BGR to RGB
        rgb_image = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


        lines_equations, images_process = line_equation_four_edges_calibration_target_in_camera_image(
                                            rgb_image=rgb_image,
                                            display=True)

        print('all line equations for four edges of calibration target')
        for line_name in lines_equations:
            print('=' * 100)
            print('Line name: {}'.format(line_name))
            print('Line equation (point, direction): {}'.format(lines_equations[line_name]))
            print('Line equation (homogenous format):\n {}'.format(
                                                            conver_2d_line_equation_to_homogenous_format(line_equation=lines_equations[line_name]))
                                                        )
        
        for img in images_process:
            plt.figure()
            plt.imshow(img)
        plt.show()
