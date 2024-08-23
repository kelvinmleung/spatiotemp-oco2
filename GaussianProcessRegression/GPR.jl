using HDF5
using PyCall
np = pyimport("numpy")
using Random
using Plots
default()
using Statistics
using LinearAlgebra
include("/Users/Camila/Desktop/OCO-2_UROP/spatiotemp-oco2/bayesian_helpers.jl")
include("/Users/Camila/Desktop/OCO-2_UROP/spatiotemp-oco2/plot_helpers.jl")

plots_dir = "/Users/Camila/Desktop/OCO-2_UROP/spatiotemp-oco2/Plots/CO2_GPR"
sample_dir = "/Users/Camila/Desktop/OCO-2_UROP/spatiotemp-oco2/GaussianProcessRegression"

#Load true state vector
np_truex = numpy_true_x = np.load("/Users/Camila/Desktop/OCO-2_UROP/spatiotemp-oco2/SampleState-Lamont2015/true_state_vector_2015-10_lamont.npy")
state_vec = convert(Array{Float64}, numpy_true_x)

#Build State Vector Spatial Field
SPAlbAero_tensor = repeat(reshape(state_vec[21:39],1,1,19),8,8,1)
CO2_file = h5open("/Users/Camila/Desktop/OCO-2_UROP/spatiotemp-oco2/GRF/Lamont2015_CO2_covGRF.h5", "r")
CO2GRF_tensor = read(CO2_file["CO2tensor-SampleCov_Std_SEKernel"])
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
S_tuple = fill((0.0,0.0), 64)
for (idx,(x,y)) in enumerate(all_pixels)
    S_tuple[idx] = (x,y)
end
S_tuple


# Simulate observed locations
using StatsBase
num_obs = 16
obs_indices = StatsBase.sample(1:64, num_obs, replace=false)
obs_coords = S_tuple[obs_indices, :]

#Simulate observed values
observed_state = zeros(num_obs,39)

for (idx,(loc)) in enumerate(obs_indices)
    println("idx $idx")
    println("loc $loc")
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
    # map_estimate,objective_plot, gradient_plot = find_map(initial,neg_log_posterior)

    #plot vertical profile
    row = div(loc - 1, 8) + 1
    col = mod(loc - 1, 8) + 1
    true_x = fixed_SPAlbAero_tensor[row,col,:]
    # vertical_profile = plot_vertical_profile(map_estimate,prior_mean, "Lamont 2015", true_map, true_x)
    # display(vertical_profile)
    # observed_state[idx,:] = map_estimate
end

#Slice to retain only the CO2 profile
observed_state = observed_state[:,1:20]

# Normalize output data
X_I_mean = mean(observed_state,dims=1)
norm_X_I = (observed_state .- X_I_mean)

#Load sample variances
sample_cov = read(h5open("/Users/Camila/Desktop/OCO-2_UROP/spatiotemp-oco2/OCO2_Data-Lamont2015/SampleCov.h5", "r")["sample_cov_matrix"])
sample_variances = diag(sample_cov)






#Use Gaussian process regression to make predictions at each level 
using GaussianProcesses
level = 1
mZero = MeanZero()
true_var = sample_variances[level]
sigma = sqrt(true_var) - 0.05
lambda = 1.05
log(lambda)
log(sigma)
kern = SE(log(lambda), log(sigma))
obs_coords
S_I = reshape(hcat(map(t -> collect(t), obs_coords)...), 2, num_obs)
S = hcat(map(collect, S_tuple)...)
norm_XI = norm_X_I[:,level]
true_grid = CO2GRF_tensor[:,:,level]
gp = GP(S_I,norm_XI, mZero,kern)
heatmap(gp)
μ1,var1 = predict_y(gp,S)

pred1 = fill(0.0,(8,8))
for (idx,val) in enumerate(μ1)
    row = div(idx - 1, 8) + 1
    col = mod(idx - 1, 8) + 1
    pred1[row,col] = val
end
pred1
pred1 = pred1 .+ X_I_mean[level] 
pred1_plt = heatmap(pred1, 
            title="Level $level GP Predictions Pre-Optimization, n=$num_obs, λ=$(round(lambda,digits=3)), σ=$(round(sigma,digits=3))",
            titlefontsize=10,
            clims=(325,475))
savefig(pred1_plt, joinpath(plots_dir, "Level$(level)_Prediction1.png"))

err1 = true_grid-pred1
err1_plt = heatmap(err1, 
            color=:balance, 
            title="Level $level Pre-Optimization Net Error",
            clims=(-50.0,50.0))
savefig(err1_plt, joinpath(plots_dir, "Level$(level)_Err1.png"))
total_err1 = (sum(err1 .^2))/64
# h5write(joinpath(sample_dir,"Lamont2015_CO2GPR_MSE.h5"), "Level$(level)_err1", total_err1)


#Optimize GP Kernel parameters
using Optim
optimize!(gp,kern=true, domean=false; method=BFGS(), g_tol = 1e-20)
gp.kernel
opt_lambda = exp(0.266171932086272166)
opt_sigma = exp(-2.4349208358988106)
μ2,var2 = predict_y(gp,S)
pred2 = fill(0.0,(8,8))
for (idx,val) in enumerate(μ2)
    row = div(idx - 1, 8) + 1
    col = mod(idx - 1, 8) + 1
    pred2[row,col] = val
end
pred2
pred2 = pred2 .+ X_I_mean[level]
pred2_plt = heatmap(pred2, 
            title="GP Predictions Post-Optimization n=$num_obs, λ=$(round(opt_lambda,digits=3)), σ=$(round(opt_sigma,digits=3))",
            titlefontsize=10,
            clims=(325,475))
savefig(pred2_plt, joinpath(plots_dir, "Level$(level)_Prediction2.png"))

err2 = true_grid-pred2
err2_plt = heatmap(err2, 
            color=:balance, 
            title="Level $level Post-Optimization Error n=$num_obs",
            clims=(-50.0,50.0))
savefig(err2_plt, joinpath(plots_dir, "Level$(level)_Err2.png"))
total_err2 = (sum(err2 .^2))/64
# h5write(joinpath(sample_dir,"Lamont2015_CO2GPR_MSE.h5"), "Level$(level)_err2", total_err2)


single_pixel_tensor = fill(0.0, 8,8,20)
#Single pixel retrieval on same field
for idx in 1:64
    sample_radiance = rad_tensor[idx,:]
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
    row = div(idx - 1, 8) + 1
    col = mod(idx - 1, 8) + 1
    true_x = fixed_SPAlbAero_tensor[row,col,:]
    vertical_profile = plot_vertical_profile(map_estimate,prior_mean, "Lamont 2015", true_map, true_x)
    display(vertical_profile)
    single_pixel_tensor[row,col,:] = map_estimate[1:20]
end
single_pixel_tensor

#Calculate MSE at each level
err = CO2GRF_tensor .- single_pixel_tensor
for level in 1:20
    grid = err[:,:,level]
    mse = (sum(grid .^2))/64
    println(mse)
    # h5write(joinpath(sample_dir,"Lamont2015_CO2SP_MSE.h5"), "Level$(level)", mse)
end

#Plot spatial pred1 pred 2 and single pred
GPR_MSE_file = h5open("/Users/Camila/Desktop/OCO-2_UROP/spatiotemp-oco2/GaussianProcessRegression/Lamont2015_CO2GPR_MSE.h5", "r")
SP_MSE_file = h5open("/Users/Camila/Desktop/OCO-2_UROP/spatiotemp-oco2/GaussianProcessRegression/Lamont2015_CO2SP_MSE.h5", "r")
GPR_Pred1_MSE = fill(0.0,20)
GPR_Pred2_MSE = fill(0.0,20)
SP_MSE = fill(0.0,20)
for level in 1:20
    pred1 = read(GPR_MSE_file["Level$(level)_err1"])
    GPR_Pred1_MSE[level] = pred1

    pred2 = read(GPR_MSE_file["Level$(level)_err2"])
    GPR_Pred2_MSE[level] = pred2

    sp_pred = read(SP_MSE_file["Level$level"])
    SP_MSE[level] = sp_pred
end

mse_plt = plot(1:20, GPR_Pred1_MSE, label="GPR MSE Pre-Optimization", lw=2, marker=:circle)
plot!(1:20, GPR_Pred2_MSE, label="GPR MSE Post-Optimization", lw=2, marker=:square)
plot!(1:20, SP_MSE, label="Single Pixel", lw=2, marker=:diamond)
xlabel!("Levels")
ylabel!("Mean Squared Error")
title!("Lamont 2015 MSE Comparison")
savefig(mse_plt, joinpath(plots_dir, "Lamont2015_MSEComp.png"))