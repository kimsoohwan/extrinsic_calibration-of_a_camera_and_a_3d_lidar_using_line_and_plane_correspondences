import numpy as np


point_cloud = np.random.randint(low=0, high=100, size=(10, 3))


all_points = point_cloud.tolist()

# sort according to y
all_points = sorted(all_points, key = lambda x: x[1])

# sort according to z
all_points = sorted(all_points, key = lambda x: x[2])

print(all_points)
