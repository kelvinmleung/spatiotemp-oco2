import numpy as np
import h5py 

filepath = '/Users/Camila/Desktop/OCO-2 UROP/spatiotemp-oco2/linear_oco_model/example_oco2_linear_model_201510_lamont.h5'
with h5py.File(filepath, 'r') as f:
    Fmatrix = f['model_matrix'][:,:]
    true_x = f['state_true_mean_vector'][:]
    prior_mean_x = f['operational_prior_mean_vector'][:]
    prior_cov_x = f['operational_prior_covariance_matrix'][:,:]
    error_variance = f['error_variance_diagonal'][:]

    # Print the dimensions of each key
    for key in f.keys():
        dataset = f[key]
        print(f"Key: {key}, Shape: {dataset.shape}")

    print(f['state_vector_names'][:])

# np.save('example_oco2_linear_model_201706_wollongong.npy', Fmatrix)
np.save('true_state_vector_2015-10_lamont.npy', true_x)
np.save('prior_mean_2015-10_lamont.npy', prior_mean_x)
np.save('prior_cov_matrix_2015-10_lamont.npy', prior_cov_x)
np.save('error_variance_2015-10_lamont.npy', error_variance)