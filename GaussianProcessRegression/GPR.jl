using HDF5
using Random
using Plots
default()
using Statistics
using LinearAlgebra
include("/Users/Camila/Desktop/OCO-2_UROP/spatiotemp-oco2/bayesian_helpers.jl")
include("/Users/Camila/Desktop/OCO-2_UROP/spatiotemp-oco2/plot_helpers.jl")
include("/Users/Camila/Desktop/OCO-2_UROP/spatiotemp-oco2/GaussianProcessRegression/GPR_helpers.jl")

#Load true state vector
np_truex = numpy_true_x = np.load("/Users/Camila/Desktop/OCO-2_UROP/spatiotemp-oco2/SampleState-Lamont2015/true_state_vector_2015-10_lamont.npy")
state_vec = convert(Array{Float64}, numpy_true_x)

#Build State Vector Spatial Field
SPAlbAero_tensor = repeat(reshape(state_vec[21:39],1,1,19),8,8,1)
CO2_file = h5open("/Users/Camila/Desktop/OCO-2_UROP/spatiotemp-oco2/GRF/Lamont2015_CO2_covGRF.h5", "r")
CO2GRF_tensor = read(CO2_file["CO2tensor-SampleCov_SEKernel"])
fixed_SPAlbAero_tensor = cat(CO2GRF_tensor, SPAlbAero_tensor, dims=3)


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
using StatsBase
num_obs = 8
obs_indices = StatsBase.sample(1:64, 8, replace=false)
obs_coords = S[obs_indices, :]

#Simulate observed values
observed_state = zeros(n,39)

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
X_I_mean = mean(observed_state,dims=1)
X_I_std = std(observed_state, dims=1)
norm_X_I = (observed_state .- X_I_mean) ./ X_I_std


#Level 1 GPR
using GaussianProcesses
mZero = MeanZero()
kern = SE(0.0,0.0)
obs_coords
S_I = reshape(hcat(map(t -> collect(t), obs_coords)...), 2, 8)
norm_X_I_L1 = norm_X_I[:,1]
gp = GP(S_I,norm_X_I_L1,mZero,kern)
optimize!(gp, kern=true)
gp.kernel
opt_lambda = exp(-1.816117049367534)
opt_sigma = exp(-0.0783378382633143)

