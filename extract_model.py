import numpy as np
import h5py 

filepath = '/Users/Camila/Desktop/OCO-2 UROP/spatiotemp-oco2/linear_oco_model/example_oco2_linear_model_201706_wollongong.h5'
with h5py.File(filepath, 'r') as f:
    Fmatrix = f['model_matrix'][:,:]

np.save('example_oco2_linear_model_201706_wollongong.npy', Fmatrix)
