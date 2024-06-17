include("bayesian_helpers.jl")
include("plot_helpers.jl")
using PyCall
np = pyimport("numpy")
using LinearAlgebra
using Distributions
using Serialization

save_dir = "/Users/Camila/Desktop/OCO-2_UROP/spatiotemp-oco2/Plots"

#Load forward model 
numpy_model = np.load("/Users/Camila/Desktop/OCO-2_UROP/spatiotemp-oco2/linear_model_2015-10_lamont.npy")
F_matrix = transpose(convert(Array{Float64}, numpy_model))

#Load true x
numpy_true_x = np.load("/Users/Camila/Desktop/OCO-2_UROP/spatiotemp-oco2/true_state_vector_2015-10_lamont.npy")
true_x = convert(Array{Float64}, numpy_true_x)

#Load error variance and make diagonal matrix
numpy_error_var = np.load("/Users/Camila/Desktop/OCO-2_UROP/spatiotemp-oco2/error_variance_2015-10_lamont.npy")
error_variance = convert(Array{Float64}, numpy_error_var)
#check if any of the elements of error_var are 0
@assert all(x -> x != 0, error_variance) "variance vector should not contain any zero elements."
error_cov_matrix = Diagonal(error_variance)

#Load prior cov and mean
numpy_prior_mean = np.load("/Users/Camila/Desktop/OCO-2_UROP/spatiotemp-oco2/prior_mean_2015-10_lamont.npy")
prior_mean = convert(Array{Float64}, numpy_prior_mean)
numpy_prior_cov = np.load("/Users/Camila/Desktop/OCO-2_UROP/spatiotemp-oco2/prior_cov_matrix_2015-10_lamont.npy")
prior_cov_matrix = convert(Array{Float64}, numpy_prior_cov)

#Load weighting function
numpy_h = np.load("/Users/Camila/Desktop/OCO-2_UROP/spatiotemp-oco2/weighting_func_2015-10_lamont.npy")
h_matrix = convert(Array{Float64}, numpy_h)


# Sample from the error distribution
n_y = size(F_matrix)[1]
error_mean = zeros(Float64, n_y)
error_dist = MvNormal(error_mean, error_cov_matrix)
# Check if the file exists before loading sample_error
if isfile("sample_error_lamont2015.jls")
    # Load sample_error from the file
    sample_error = open(deserialize, "sample_error_lamont2015.jls")
else
    # Generate new sample_error if the file doesn't exist
    sample_error = rand(error_dist)
    
    # Save sample_error to a file
    open("sample_error_lamont2015.jls", "w") do io
        serialize(io, sample_error)
    end
end



# Generate the sample radiance
sample_Y = F_matrix * true_x + sample_error


true_map, true_posterior_cov = calc_true_posterior(prior_cov_matrix, prior_mean, F_matrix, error_cov_matrix, sample_Y)

#find MAP
function apply_forward_model(x::Vector{<:Real})
    return F_matrix*x
end

neg_log_posterior(x) = -calc_log_posterior(x,sample_Y,prior_cov_matrix,error_cov_matrix,prior_mean,apply_forward_model)
initial = prior_mean

map_estimate,objective_plot, gradient_plot = find_map(initial,neg_log_posterior)
post_cov_estimate = laplace_approx(map_estimate, neg_log_posterior)
savefig(objective_plot, joinpath(save_dir, "Lamont2015-NegLogPosterior-vs-Iterations.pdf"))
savefig(gradient_plot, joinpath(save_dir, "Lamont2015-Gradient-vs-Iterations.pdf"))

#Evaluate MAP estimate
log_post_map = calc_log_posterior(true_map,sample_Y, prior_cov_matrix,error_cov_matrix,prior_mean, apply_forward_model)
log_post_estimate = calc_log_posterior(map_estimate, sample_Y, prior_cov_matrix, error_cov_matrix, prior_mean, apply_forward_model)
println("norm of difference with map ", norm(map_estimate .- true_map))
println("difference in log posterior of map vs. estimate ", log_post_map - log_post_map_estimate)


#Plot true CO2 conc vs. map CO2 conc vs. map estimate CO2 conc
map_col_avg_co2 = h_matrix'*true_map[1:20]
post_std_col_avg_co2 = sqrt(h_matrix'*true_posterior_cov_matrix[1:20,1:20]*h_matrix)

map_estimate_col_avg_co2 = h_matrix'*map_estimate[1:20]
post_estimate_std_col_avg_co2 = sqrt(h_matrix'*post_cov_estimate[1:20,1:20]*h_matrix)

true_col_avg_co2 = h_matrix'*true_x[1:20]
true_std_col_avg_co2 = 0

values = [map_col_avg_co2, map_estimate_col_avg_co2, true_col_avg_co2]
std_devs = [post_std_col_avg_co2, post_estimate_std_col_avg_co2, true_std_col_avg_co2]

labels = ["Map Col Avg CO2", "Map Estimate Col Avg CO2", "True Col Avg CO2"]

# Create a bar chart with error bars
bar_width = 0.5  
ColAvgCO2_plot = bar(labels, values, yerr=std_devs, label="CO2 Concentration", bar_width=bar_width, title="CO2 Concentration Comparison", xlabel="Categories", ylabel="CO2 Concentration")
ylims!(350,450)
display(ColAvgCO2_plot)
savefig(ColAvgCO2_plot, joinpath(save_dir, "Lamont2015-ColAvgCO2.pdf"))

#Plot state vector elements of true vs map vs estimate
true_co2 = true_x[1:20]
true_co2_stds = zeros(20)

post_co2 = true_map[1:20]
post_co2_stds = sqrt.(diag(true_posterior_cov_matrix)[1:20])

est_co2 = map_estimate[1:20]
est_co2_stds = sqrt.(diag(post_cov_estimate)[1:20])

CO2byLevel_plot = plot_CO2_by_level(true_co2, post_co2, post_co2_stds, est_co2, est_co2_stds)
display(CO2byLevel_plot)
savefig(CO2byLevel_plot, joinpath(save_dir, "Lamont2015-CO2byLevel.pdf"))

#plot vertical profile
vertical_profile = plot_vertical_profile(est_co2, post_co2, true_co2)