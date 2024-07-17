include("/Users/Camila/Desktop/OCO-2_UROP/spatiotemp-oco2/bayesian_helpers.jl")
include("/Users/Camila/Desktop/OCO-2_UROP/spatiotemp-oco2/plot_helpers.jl")
using PyCall
np = pyimport("numpy")
using LinearAlgebra
using Distributions
using Serialization

save_dir = "/Users/Camila/Desktop/OCO-2_UROP/spatiotemp-oco2/Plots"

#Load forward model 
numpy_model = np.load("/Users/Camila/Desktop/OCO-2_UROP/spatiotemp-oco2/Wollongong-2017/linear_model_Wollongong2017.npy")
F_matrix = transpose(convert(Array{Float64}, numpy_model))

#Load true_x
numpy_true_x = np.load("/Users/Camila/Desktop/OCO-2_UROP/spatiotemp-oco2/Wollongong-2017/true_state_vector_Wollongong2017.npy")
true_x = convert(Array{Float64}, numpy_true_x)

#Load error variance and make diagonal matrix
numpy_error_var = np.load("/Users/Camila/Desktop/OCO-2_UROP/spatiotemp-oco2/Wollongong-2017/error_variance_Wollongong2017.npy")
error_variance = convert(Array{Float64}, numpy_error_var)
#check if any of the elements of error_var are 0
@assert all(x -> x != 0, error_variance) "variance vector should not contain any zero elements."
error_cov_matrix = Diagonal(error_variance)

#Load prior cov and mean
numpy_prior_mean = np.load("/Users/Camila/Desktop/OCO-2_UROP/spatiotemp-oco2/Wollongong-2017/prior_mean_Wollongong2017.npy")
prior_mean = convert(Array{Float64}, numpy_prior_mean)
numpy_prior_cov = np.load("Wollongong-2017/prior_cov_matrix_Wollongong2017.npy")
prior_cov_matrix = convert(Array{Float64}, numpy_prior_cov)

#Load weighting function
numpy_h = np.load("/Users/Camila/Desktop/OCO-2_UROP/spatiotemp-oco2/Wollongong-2017/weighting_func_Wollongong2017.npy")
h_matrix = convert(Array{Float64}, numpy_h)



# Sample from the error distribution
n_y = size(F_matrix)[1]
error_mean = zeros(Float64, n_y)
error_dist = MvNormal(error_mean, error_cov_matrix)
# Check if the file exists before loading sample_error
if isfile("/Users/Camila/Desktop/OCO-2_UROP/spatiotemp-oco2/Wollongong-2017/sample_error_Wollongong2017.jls")
    # Load sample_error from the file
    sample_error = open(deserialize, "/Users/Camila/Desktop/OCO-2_UROP/spatiotemp-oco2/Wollongong-2017/sample_error_Wollongong2017.jls")
else
    # Generate new sample_error if the file doesn't exist
    sample_error = rand(error_dist)
    
    # Save sample_error to a file
    open("/Users/Camila/Desktop/OCO-2_UROP/spatiotemp-oco2/Wollongong-2017/sample_error_Wollongong2017.jls", "w") do io
        serialize(io, sample_error)
    end
end

# Generate the sample radiance
sample_Y = F_matrix * true_x + sample_error

#Calc true posterior
true_map, true_posterior_cov = calc_true_posterior(prior_cov_matrix, prior_mean, F_matrix, error_cov_matrix, sample_Y)



#find MAP
function apply_forward_model(x::Vector{<:Real})
    return F_matrix*x
end

neg_log_posterior(x) = -calc_log_posterior(x,sample_Y,prior_cov_matrix,error_cov_matrix,prior_mean,apply_forward_model)
initial = prior_mean

map_estimate,objective_plot, gradient_plot = find_map(initial,neg_log_posterior)
post_cov_estimate = laplace_approx(map_estimate, neg_log_posterior)
savefig(objective_plot, joinpath(save_dir, "Wollongong2017-NegLogPosterior-vs-Iterations.png"))
savefig(gradient_plot, joinpath(save_dir, "Wollongong2017-Gradient-vs-Iterations.png"))



#Compare MAP estimate to True MAP
log_post_map = calc_log_posterior(true_map,sample_Y, prior_cov_matrix,error_cov_matrix,prior_mean, apply_forward_model)
log_post_estimate = calc_log_posterior(map_estimate, sample_Y, prior_cov_matrix, error_cov_matrix, prior_mean, apply_forward_model)
println("norm of difference with map ", norm(map_estimate .- true_map))
println("difference in log posterior of map vs. estimate ", log_post_map - log_post_estimate)



#Plot true CO2 conc vs. map CO2 conc vs. map estimate CO2 conc
col_avg_plot = plot_col_avg_co2(h_matrix, map_estimate, post_cov_estimate, true_map, true_posterior_cov, true_x)
display(col_avg_plot)
savefig(col_avg_plot, joinpath(save_dir, "Wollongong2017-ColAvgCO2.png"))

#Plot state vector elements of true vs map vs estimate
CO2byLevel_plot = plot_CO2_by_level(true_x, true_map, true_posterior_cov, map_estimate, post_cov_estimate)
display(CO2byLevel_plot)
savefig(CO2byLevel_plot, joinpath(save_dir, "Wollongong2017-CO2byLevel.png"))

#plot vertical profile
vertical_profile = plot_vertical_profile(map_estimate, prior_mean,"Wollongong 2017", true_map, true_x)
display(vertical_profile)
savefig(vertical_profile, joinpath(save_dir, "Wollongong2017-VerticalProfile.png"))