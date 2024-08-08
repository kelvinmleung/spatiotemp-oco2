using HDF5
using Random
using Plots
default()
using Statistics
using LinearAlgebra
include("/Users/Camila/Desktop/OCO-2_UROP/spatiotemp-oco2/bayesian_helpers.jl")
include("/Users/Camila/Desktop/OCO-2_UROP/spatiotemp-oco2/plot_helpers.jl")
include("/Users/Camila/Desktop/OCO-2_UROP/spatiotemp-oco2/GaussianProcessRegression/GPR_helpers.jl")

np_truex = numpy_true_x = np.load("/Users/Camila/Desktop/OCO-2_UROP/spatiotemp-oco2/SampleState-Lamont2015/true_state_vector_2015-10_lamont.npy")
state_vec = convert(Array{Float64}, numpy_true_x)
SPAlbAero_tensor = repeat(reshape(state_vec[21:39],1,1,19),8,8,1)

#Build State Vector Spatial Field

CO2_file = h5open("/Users/Camila/Desktop/OCO-2_UROP/spatiotemp-oco2/GRF/Lamont2015_CO2_covGRF.h5", "r")
diff_CO2_tensor = read(CO2_file["CO2tensor-DiffCov"])
fixed_SPAlbAero_tensor = cat(diff_CO2_tensor, SPAlbAero_tensor, dims=3)

# h5write(joinpath("GRF","Lamont2015_State_GRF.h5"), "FixedSPAlbAero", fixed_SPAlbAero_tensor)

#Construct Simulated Radiances
numpy_model = np.load("/Users/Camila/Desktop/OCO-2_UROP/spatiotemp-oco2/SampleState-Lamont2015/linear_model_2015-10_lamont.npy")
F_matrix = transpose(convert(Array{Float64}, numpy_model))

#Load error variance and make diagonal matrix
numpy_error_var = np.load("/Users/Camila/Desktop/OCO-2_UROP/spatiotemp-oco2/SampleState-Lamont2015/error_variance_2015-10_lamont.npy")
error_variance = convert(Array{Float64}, numpy_error_var)
#check if any of the elements of error_var are 0
@assert all(x -> x != 0, error_variance) "variance vector should not contain any zero elements."
error_cov_matrix = Diagonal(error_variance)

# Create error distribution
n_y = size(F_matrix)[1]
error_mean = zeros(Float64, n_y)
error_dist = MvNormal(error_mean, error_cov_matrix)

#Apply forward model to calculate radiance
rad_tensor = zeros(64,3048)
i = 1
for y in 1:8
    for x in 1:8
        state_vec = fixed_SPAlbAero_tensor[x,y,:]
        radiance = F_matrix*state_vec
        error = rand(error_dist)
        radiance += error
        rad_tensor[i,:] = radiance
        i += 1
    end
end 
        
rad_tensor

#Load prior cov and mean
numpy_prior_mean = np.load("/Users/Camila/Desktop/OCO-2_UROP/spatiotemp-oco2/SampleState-Lamont2015/prior_mean_2015-10_lamont.npy")
prior_mean = convert(Array{Float64}, numpy_prior_mean)
numpy_prior_cov = np.load("SampleState-Lamont2015/prior_cov_matrix_2015-10_lamont.npy")
prior_cov_matrix = convert(Array{Float64}, numpy_prior_cov)

# Define the grid points
s_pts_x = range(0.65, step=1.3, length=8)
s_pts_y = range(1.15, step=2.3, length=8)
all_pixels = [(sx,sy) for sx in s_pts_x, sy in s_pts_y]
S = fill((0.0,0.0), 64)
for (idx,(x,y)) in enumerate(all_pixels)
    S[idx] = (x,y)
end
S

# Simulate observed locations
using Random
obs_indices = sample(1:64, 32, replace=false)
obs_coords = S[obs_indices, :]

#Simulate observed values
observed_state = zeros(32,39)

for (idx,(loc)) in enumerate(obs_indices)

    sample_radiance = rad_tensor[loc,:]

    #Find analytical MAP
    true_map, true_posterior_cov = calc_true_posterior(prior_cov_matrix, 
                                    prior_mean, 
                                    F_matrix, 
                                    error_cov_matrix, 
                                    sample_radiance)

    #estimate MAP
    function apply_forward_model(x::Vector{<:Real})
        return F_matrix*x
    end
    neg_log_posterior(x) = -calc_log_posterior(x,sample_radiance,prior_cov_matrix,error_cov_matrix,prior_mean,apply_forward_model)
    initial = prior_mean
    map_estimate,objective_plot, gradient_plot = find_map(initial,neg_log_posterior)

    #plot vertical profile
    row = div(loc - 1, 8) + 1
    col = mod(loc - 1, 8) + 1
    true_x = fixed_SPAlbAero_tensor[row,col,:]
    vertical_profile = plot_vertical_profile(map_estimate,prior_mean, "Lamont 2015", true_map, true_x)
    display(vertical_profile)
    observed_state[idx,:] = map_estimate
end

#Slice to retain only the CO2 profile
observed_state = observed_state[:,1:20]

# Normalize output data
X_I_mean = mean(