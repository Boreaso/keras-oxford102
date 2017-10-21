from scipy.io import loadmat
import numpy as np

matrix = loadmat("data/distancematrices102.mat")
array = np.array(matrix)
print(array.shape)