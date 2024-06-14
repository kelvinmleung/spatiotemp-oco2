include("bayesian_helpers.jl")
using PyCall
np = pyimport("numpy")
using LinearAlgebra
using Distributions
using Serialization

#Load forward model 
numpy_model = np.load("/Users/Camila/Desktop/OCO-2 UROP/spatiotemp-oco2/linear_model_2015-10_lamont.npy")
F_matrix = transpose(convert(Array{Float64}, numpy_model))

#Load true x
numpy_true_x = np.load("/Users/Camila/Desktop/OCO-2 UROP/spatiotemp-oco2/true_state_vector_2015-10_lamont.npy")
true_x = convert(Array{Float64}, numpy_true_x)

#Load error variance and make diagonal matrix
numpy_error_var = np.load("/Users/Camila/Desktop/OCO-2 UROP/spatiotemp-oco2/error_variance_2015-10_lamont.npy")
error_variance = convert(Array{Float64}, numpy_error_var)
#check if any of the elements of error_var are 0
@assert all(x -> x != 0, error_variance) "variance vector should not contain any zero elements."
error_cov_matrix = Diagonal(error_variance)

#Load prior cov and mean
numpy_prior_mean = np.load("/Users/Camila/Desktop/OCO-2 UROP/spatiotemp-oco2/prior_mean_2015-10_lamont.npy")
prior_mean = convert(Array{Float64}, numpy_prior_mean)
numpy_prior_cov = np.load("/Users/Camila/Desktop/OCO-2 UROP/spatiotemp-oco2/prior_cov_matrix_2015-10_lamont.npy")
prior_cov_matrix = convert(Array{Float64}, numpy_prior_cov)


# Sample from the error distribution
n_y = size(F_matrix)[1]
println(n_y)
error_mean = zeros(Float64, n_y)
error_dist = MvNormal(error_mean, error_cov_matrix)
# Check if the file exists before loading sample_error
if isfile("sample_error_data.jls")
    # Load sample_error from the file
    sample_error = open(deserialize, "sample_error_data.jls")
else
    # Generate new sample_error if the file doesn't exist
    sample_error = rand(error_dist)
    
    # Save sample_error to a file
    open("sample_error_data.jls", "w") do io
        serialize(io, sample_error)
    end
end



# Generate the sample radiance
sample_Y = F_matrix * true_x + sample_error


# Manual Posterior Calc 
manual_posterior_cov_matrix = inv(inv(prior_cov_matrix) + F_matrix'*inv(error_cov_matrix)*F_matrix)
manual_posterior_mean = manual_posterior_cov_matrix*(inv(prior_cov_matrix)*prior_mean + F_matrix'*inv(error_cov_matrix)*sample_Y)
println(manual_posterior_mean)

#Inference using helpers
function apply_forward_model(x::Vector{Float64})
    return F_matrix*x
end



objective(x) = -calc_log_posterior(x,sample_Y,prior_cov_matrix,error_cov_matrix,prior_mean,apply_forward_model)
diff = ones(39)
initial = manual_posterior_mean .- diff
println(initial)
log_post_map = calc_log_posterior(manual_posterior_mean,sample_Y, prior_cov_matrix,error_cov_matrix,prior_mean, apply_forward_model)


map_estimate = find_map(initial,objective)
println("estimate ",map_estimate)
println("map", manual_posterior_mean)
println("norm of difference with map ", norm(map_estimate .- manual_posterior_mean))
log_post_map_estimate = calc_log_posterior(map_estimate, sample_Y,prior_cov_matrix,error_cov_matrix,prior_mean,apply_forward_model)
println("difference in log posterior of map vs. estimate ", log_post_map - log_post_map_estimate)