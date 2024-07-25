using HDF5
using Random
using Plots
default()
using Statistics

CO2_file = h5open("/Users/Camila/Desktop/OCO-2_UROP/spatiotemp-oco2/GRF/Lamont2015_CO2_covGRF.h5", "r")
diff_CO2_tensor = read(CO2_file["CO2tensor-DiffCov"])

SP_file = h5open("/Users/Camila/Desktop/OCO-2_UROP/spatiotemp-oco2/GRF/Lamont2015_SP_GRF.h5", "r")
SP_matrix = read(SP_file["SPmatrix"])

np_truex = numpy_true_x = np.load("/Users/Camila/Desktop/OCO-2_UROP/spatiotemp-oco2/SampleState-Lamont2015/true_state_vector_2015-10_lamont.npy")
albedos = convert(Array{Float64}, numpy_true_x)[22:27]
albedo_tensor = repeat(reshape(albedos, 1, 1, 6), 8, 8, 1)