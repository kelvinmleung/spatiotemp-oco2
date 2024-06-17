using LinearAlgebra
using Optim
using ForwardDiff
using Distributions
using Plots


save_dir = "/Users/Camila/Desktop/OCO-2_UROP/spatiotemp-oco2/Plots"

identity_func(x) = x


function calc_log_posterior(x,y, cov_x, cov_y, mu_x, f=identity_func)
    n_x = length(x)
    n_y = length(y)
    mu_y = f(x)
    # println("mu_y", mu_y)
    @assert length(mu_x) == n_x "mu_x has size $(length(mu_x)) not $(n_x)"
    @assert length(mu_y) == n_y "mu_y has size $(length(mu_y)) not $(n_y)"
    @assert size(cov_x) == (n_x,n_x) "cov_x matrix has size $(size(cov_x)) not $(n_x), $(n_x)"
    @assert size(cov_y) == (n_y,n_y) "cov_y matrix has size $(size(cov_y)) not $(n_y), $(n_y)"
    log_prior  = -0.5*(n_x*log(2π) + logdet(cov_x) + transpose((x .- mu_x))*inv(cov_x)*(x .- mu_x))
    # println("log_prior ", log_prior)
    log_likelihood = -0.5*(n_y*log(2π) + logdet(cov_y) + transpose((y .- mu_y))*inv(cov_y)*(y .- mu_y))
    # println("log_likelihood ", log_likelihood)
   return log_prior + log_likelihood
end  



# Find MAP 
function find_map(initial_guess, objective)
    opt = optimize(objective, initial_guess, LBFGS(m=20), Optim.Options(store_trace = true, iterations = 1000, g_tol =0.25, allow_f_increases=false, ))
    println(opt)
    trace = opt.trace
    # println(trace)
    trace_objective = []
    trace_gradient = []
    iterations = 5:length(trace)
    
    for i in iterations
        append!(trace_objective, parse(Float64, split(string(trace[i]))[2]))
        append!(trace_gradient, parse(Float64, split(string(trace[i]))[3]))
    end

    # Plotting the objective values vs. iterations
    objective_plot = plot(iterations, trace_objective, seriestype = :line, marker = :auto, label = "Objective Value")
    xlabel!("Iteration")
    ylabel!("Objective Value")

    gradient_plot = plot(iterations, trace_gradient, seriestype = :line, marker = :auto, label = "Gradient")
    xlabel!("Iteration")
    ylabel!("Gradient")

    # Display the plot
    display(objective_plot)
    display(gradient_plot)
    
    # println(opt.trace)
    map = Optim.minimizer(opt)
    return map, objective_plot, gradient_plot 
end






function laplace_approx(map, neg_log_posterior)
    # Compute Hessian at the MAP 
    hess = ForwardDiff.hessian(neg_log_posterior, map)
    cov_matrix = inv(hess)
    return cov_matrix 
end 




function calc_true_posterior(prior_cov, prior_mean, F_matrix, error_cov, Y)
    posterior_cov = inv(inv(prior_cov) + F_matrix'*inv(error_cov)*F_matrix)
    MAP = posterior_cov*(inv(prior_cov)*prior_mean + F_matrix'*inv(error_cov)*Y)
    return MAP, posterior_cov
end
########################## Univariate Test cases ##############################
#test calc log posterior
x = [1.0]
y = [1.35]
cov_x = reshape([1.0], 1, 1)  # Define cov_x as a 1x1 matrix
cov_y = reshape([2.0], 1, 1)  # Define cov_y as a 1x1 matrix
mu_x = [0.5]
println(calc_log_posterior(x, y, cov_x, cov_y, mu_x))


#test MAP estimate
initial = [1.0]
neg_log_posterior(x) = -calc_log_posterior(x,y,cov_x,cov_y,mu_x)
map_estimate = find_map(initial, neg_log_posterior)[1]

println("MAP: ", map_estimate)
# @assert calc_log_posterior(map_estimate,y,cov_x,cov_y,mu_x) == neg_log_posterior(map_estimate)

#test Laplace approximation 
cov_matrix = laplace_approx(map_estimate, neg_log_posterior)
println("laplace approx cov matrix ", cov_matrix)

#visualize Laplace approximation and true posterior

x_range = range(-10, stop=10, length=500)

posterior_values = [calc_log_posterior([x_val], y, cov_x, cov_y, mu_x) for x_val in x_range]

println("log posterior at MAP ", calc_log_posterior(map_estimate[1], y, cov_x, cov_y, mu_x))
println("log Laplace approx at MAP ", log(pdf(Normal(map_estimate[1], sqrt(cov_matrix[1,1])), map_estimate[1])))

gaussian_approx = [log(pdf(Normal(map_estimate[1], sqrt(cov_matrix[1,1])), x_val)) for x_val in x_range]

log_post_vs_estimate_plot = plot(x_range, posterior_values, label="True Log Posterior", xlabel="x", legend=:bottom)
plot!(x_range, gaussian_approx, label="Log Gaussian Approximation (Laplace)")
vline!([map_estimate[1]], label="MAP estimate", linestyle=:dash, color=:red)
savefig(log_post_vs_estimate_plot, joinpath(save_dir, "1D-LogPosterior-vs-Estimate.pdf"))
