import numpy as np
import h5py 

f = h5py.File('example_oco2_linear_model_201510_lamont.h5', 'r')

Fmatrix = f['model_matrix'][:,:]
