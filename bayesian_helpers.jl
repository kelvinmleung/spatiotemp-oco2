using LinearAlgebra

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
    log_prior  = -0.5*(n_x*log(2π) + log(det(cov_x)) + transpose((x .- mu_x))*inv(cov_x)*(x .- mu_x))
    # println("log_prior ", log_prior)
    log_likelihood = -0.5*(n_y*log(2π) + log(det(cov_y)) + transpose((y .- mu_y))*inv(cov_y)*(y .- mu_y))
    # println("log_likelihood ", log_likelihood)
   return log_prior + log_likelihood
end  

# Test cases
x = [1.0, 2.0, 3.0]
y = [4.0, 5.0, 6.0]
cov_x = [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]
cov_y = [2.0 0.0 0.0; 0.0 2.0 0.0; 0.0 0.0 2.0]
mu_x = [0.0, 0.0, 0.0]
println(calc_log_posterior(x,y,cov_x,cov_y,mu_x))


using Optim
function find_map(initial_guess, x,y,cov_x,cov_y, mu_x, f=identity_func)
    objective(x) = calc_log_posterior(x,y,cov_x,cov_y,mu_x,f)
    map = optimize(objective,initial_guess, LBFGS())
    return map
end


