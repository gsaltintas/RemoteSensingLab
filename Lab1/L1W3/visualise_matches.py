import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

SCENE_NUMBER = 2 # 1 or 2
RADIUS = 3 # 3 or 7 for search window of size 7 or 15

if not os.path.exists(f'outputs/points_scene_{SCENE_NUMBER}_{2*RADIUS+1}x{2*RADIUS+1}.npz'):
    print('\n\nPoints must me precomputed.\n\n')
    raise Exception

X = np.load(f'outputs/points_scene_{SCENE_NUMBER}_{2*RADIUS+1}x{2*RADIUS+1}.npz')['arr_0']

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[0,:], X[1,:], X[2,:], marker='.')
plt.show()