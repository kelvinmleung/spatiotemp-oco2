function squared_exp_covariance(x_i, x_j, l, σ_squared)
    return σ_squared * exp((-(x_i - x_j)^2)/(2*l^2))
end

function calc_K(all_pixels, cov_kernel)
   n = length(all_pixels)
   K = zeros(Float64,n,n)
   for i in 1:n
    for j in 1:n
        K[i,j] = cov_kernel(all_pixels[i], all_pixels[j])
    end
   end
   return K
end

function calc_K_star(all_pixels, pixels_obs, cov_kernel)
    n = length(all_pixels)
    m = length(pixels_obs)
    K_star = zeros(Float64, n,m)
    for i in 1:n
        for j in 1:m
            K[i,j] = cov_kernel(all_pixels[i], pixels_obs[j])
        end
    end
    return K_star
end

function calc_K_star_star(pixel_obs, cov_kernel)
    m = length(pixel_obs)
    K_star_star = zeros(Float64, m,m)
    for i in 1:m
        for j in 1:m
            K_star_star[i,j] = cov_kernel(pixel_obs[i], pixel_obs[j])
        end
    end
    return K_star_star
end

function get_posterior_distribution(prior_mean,cov_kernel, all_pixels, pixel_obs, state_obs)
    K = calc_K(all_pixels, cov_kernel)
    K_star = calc_K_star(all_pixels, pixel_obs, cov_kernel)
    K_star_star = calc_K_star_star(pixel_obs, cov_kernel)
    μ_star = K_star'*inv(K)*state_obs
    Σ_star = K_star_star - K_star'*inv(K)*K_star
    return μ_star, Σ_star
end 


