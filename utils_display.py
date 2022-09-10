import io
import warnings

import cv2
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore")

def show_point_cloud(point_cloud, normal_vector=None, intersection_points=None, title=None, marker=None):
    
    if  isinstance(point_cloud, list):
        point_could_temp = []
        for sub_list in point_cloud:
            point_could_temp.append(sub_list)
    else:
        point_could_temp = np.vstack((point_cloud, point_cloud[0,:]))
    
    fig = plt.figure()
    plt.autoscale(False)
    ax = plt.axes(projection='3d')

    if  isinstance(point_cloud, list):
        for  sub_list in point_could_temp:
            ax.plot3D(sub_list[:, 0], sub_list[:, 1], sub_list[:, 2], label='calibration target')
    else:
        # plot calibration target
        if marker is None:
            ax.plot3D(point_could_temp[:, 0], point_could_temp[:, 1], point_could_temp[:, 2], 'gray', label='calibration target')
        else:
            ax.plot3D(point_could_temp[:, 0], point_could_temp[:, 1], point_could_temp[:, 2], 'gray', label='calibration target', marker=marker)

    # plot lidar
    ax.scatter3D(0, 0, 0, 'green', label='LiDAR')
    
    # plot normal vector
    if normal_vector is not None:
        middle = np.mean(point_cloud, axis=0)
        ax.scatter3D(middle[0], middle[1], middle[2], 'purple')
        ax.plot3D([middle[0], middle[0]+normal_vector[0]*100],
                  [middle[1], middle[1]+normal_vector[1]*100], [middle[2], middle[2]+normal_vector[2]*100],
                  'red', label='Normal Vector')

    if intersection_points is not None:
        ax.scatter3D(intersection_points[:, 0], intersection_points[:, 1], intersection_points[:, 2], 'blue', label='intersecion points')


    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    
    if normal_vector is None:
        plt.title(title)
    else:
        if  isinstance(point_cloud, list):
            plt.title("{}\nNormal: {}, Num Lines".format(title, normal_vector, len(point_cloud)))
        else:
            plt.title("{}\nNormal: {}".format(title, normal_vector))

    plt.legend()

    numpy_img = get_img_from_fig(fig=fig)

    # close opend figure
    plt.close(fig)

    return numpy_img


# define a function which returns an image as numpy array from figure
def get_img_from_fig(fig, dpi=180):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img
