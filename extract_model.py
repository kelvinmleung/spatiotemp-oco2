import numpy as np
import h5py 

filepath = '/Users/Camila/Desktop/OCO-2_UROP/spatiotemp-oco2/Lamont-2015/example_oco2_linear_model_201510_lamont.h5'
with h5py.File(filepath, 'r') as f:
    Fmatrix = f['model_matrix'][:,:]
    true_x = f['state_true_mean_vector'][:]
    prior_mean_x = f['operational_prior_mean_vector'][:]
    prior_cov_x = f['operational_prior_covariance_matrix'][:,:]
    error_variance = f['error_variance_diagonal'][:]
    weighting_func = f['pressure_weighting_function'][:]
    wavelength = f['wavelength'][:]

    # Print the dimensions of each key
    for key in f.keys():
        dataset = f[key]
        print(f"Key: {key}, Shape: {dataset.shape}")

# np.save("wavelengths.npy", wavelength)
# np.save("Wollongong-2017/linear_model_Wollongong2017.npy", Fmatrix)
# np.save("Wollongong-2017/weighting_func_Wollongong2017.npy", weighting_func)
# np.save('Wollongong-2017/true_state_vector_Wollongong2017.npy', true_x)
# np.save('Wollongong-2017/prior_mean_Wollongong2017.npy', prior_mean_x)
# np.save('Wollongong-2017/prior_cov_matrix_Wollongong2017.npy', prior_cov_x)
# np.save('Wollongong-2017/error_variance_Wollongong2017.npy', error_variance)
