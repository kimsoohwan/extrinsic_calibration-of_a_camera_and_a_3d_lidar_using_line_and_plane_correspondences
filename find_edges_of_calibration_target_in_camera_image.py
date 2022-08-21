import cv2
import numpy as np
import matplotlib.pyplot as plt

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

if __name__ == '__main__':

    
    for img_path in ['./sample_imgs/yellow-2.jpg', '/home/farhad-bat/code/find_normal_vector_plane_pointcloud/example_real_img_lidar_points/frame-1.png', '/home/farhad-bat/code/find_normal_vector_plane_pointcloud/example_real_img_lidar_points/frame-2.png']:
        # read image
        img_bgr = cv2.imread(img_path)

        # convert BGR to HSV
        hsvImage = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        
        # separate yellow color from other parts
        color_masked_img = segment_yellow_color(img=hsvImage)
        
        # find the biggest connected  component
        bigest_component = find_biggest_connected_component(img=color_masked_img)

        points_on_edges = find_points_on_edges(img=bigest_component)

        #plt.figure()
        #plt.imshow(hsvImage, cmap = 'gray', interpolation = 'bicubic')
        #plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis

        #plt.figure()
        #plt.imshow(color_masked_img, cmap = 'gray', interpolation = 'bicubic')
        #plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis

        plt.figure()
        plt.imshow(bigest_component, cmap = 'gray', interpolation = 'bicubic')
        plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis

    plt.show()

