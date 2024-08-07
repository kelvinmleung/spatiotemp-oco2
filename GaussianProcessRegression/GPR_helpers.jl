function squared_exp_covariance(p1, p2, l, σ_squared)
    dist_squared = (p1[1] - p2[1])^2 + (p1[2] - p2[2])^2
    return σ_squared * exp((-dist_squared)/(2*l^2))
end

function calc_K(obs_pixels, cov_kernel)
   n = length(obs_pixels)
   K = zeros(Float64,n,n)
   for i in 1:n
    for j in 1:n
        K[i,j] = cov_kernel(all_pixels[i], all_pixels[j])
    end
   end
   return K
end

function calc_K_star(all_pixels, pixels_obs, cov_kernel)
    n = length(pixels_obs)
    m = length(all_pixels)
    K_star = zeros(Float64, n,m)
    for i in 1:n
        for j in 1:m
            K_star[i,j] = cov_kernel(pixels_obs[i], all_pixels[j])
        end
    end
    return K_star
end

function calc_K_star_star(all_pixels, cov_kernel)
    m = length(all_pixels)
    K_star_star = zeros(Float64, m,m)
    for i in 1:m
        for j in 1:m
            K_star_star[i,j] = cov_kernel(all_pixels[i], all_pixels[j])
        end
    end
    return K_star_star
end

function get_posterior_distribution(cov_kernel, all_pixels, pixel_obs, state_obs)
    K = calc_K(pixel_obs, cov_kernel)
    println("row 1 K: $(K[1,:]) \n")
    K_star = calc_K_star(all_pixels, pixel_obs, cov_kernel)
    println("row 1 K_star: $(K_star[1,:]) \n")
    K_star_star = calc_K_star_star(all_pixels, cov_kernel)
    inverse_K = inv(K)
    println("row 1 inverse_K: $(inverse_K[1,:]) \n")
    product = K_star'*inverse_K
    println("row 1 produvct: $(product[1,:]) \n")
    μ_star = K_star'*inverse_K*state_obs
    Σ_star = K_star_star - K_star'*inv(K)*K_star
    return μ_star, Σ_star
end 


