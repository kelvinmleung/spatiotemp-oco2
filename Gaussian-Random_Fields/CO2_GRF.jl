using PyCall
np = pyimport("numpy")
using LinearAlgebra
using Distributions
using Serialization
using GaussianRandomFields
using Plots
using SpecialFunctions
using HDF5

plots_dir = "/Users/Camila/Desktop/OCO-2_UROP/spatiotemp-oco2/Plots/GRF"
sample_dir = "/Users/Camila/Desktop/OCO-2_UROP/spatiotemp-oco2/Gaussian-Random_Fields"


#Import known values
#Lamont 2015
numpy_true_x = np.load("/Users/Camila/Desktop/OCO-2_UROP/spatiotemp-oco2/Lamont-2015/true_state_vector_2015-10_lamont.npy")
Lamont2015_true_xCO2 = convert(Array{Float64}, numpy_true_x)[1:20]
Lamont_lambdas = [18.5, 15.5, 10, 27.5, 100,
                100,100,100,100,100,
                100,100,100,100, 2.5,
                2.5, 40, 100,100,100]

# Wollongong 2016
numpy_true_x = np.load("Wollongong-2016/true_state_vector_Wollongong2016.npy")
Wollongong2016_true_xCO2 = convert(Array{Float64}, numpy_true_x)[1:20]
Wollongong2016_lambdas = [68.75, 31.25, 35, 18.5, 57.5,
                27.5,27.5,31.25,36,35,
                37,41,50,75, 87,
                22, 9.5, 100,72,53]

#Wollongong 2017
numpy_true_x = np.load("Wollongong-2017/true_state_vector_Wollongong2017.npy")
Wollongong2017_true_xCO2 = convert(Array{Float64}, numpy_true_x)[1:20]
Wollongong2017_lambdas = [40.5, 20, 30, 42.5, 100,
                79,81, 68.75,59.5,37.5,
                28.5,20,12.5,17.5, 38,
                7.5, 12, 100,100,100]


labels = ["Lamont2015", "Wollongong2016", "Wollongong2017"]
true_CO2s = [Lamont2015_true_xCO2, Wollongong2016_true_xCO2, Wollongong2017_true_xCO2]
lambdas = [Lamont_lambdas, Wollongong2016_lambdas, Wollongong2017_lambdas]


#Cov and Correlation matrix
numpy_C = np.load("/Users/Camila/Desktop/OCO-2_UROP/spatiotemp-oco2/Lamont-2015/prior_cov_matrix_2015-10_lamont.npy")
C = convert(Array{Float64}, numpy_C)[1:20, 1:20]
symmetric_C = tril(C) + tril(C,-1)'
variances = diag(symmetric_C)
D = sqrt.(diagm(variances))
correlation_matrix = inv(D)* symmetric_C * inv(D)
correlation_matrix = tril(correlation_matrix) + tril(correlation_matrix,-1)'


#Constants
diffusion_coef = 16.0 / (10^6)  
fixed_lambda = sqrt(diffusion_coef)  # km
fixed_nu = 2.0
x_pts = range(0.65,step=1.3, length=8)
y_pts = range(1.15, step=2.3, length=8)
z_pts = 1:20




function matern_kernel(p1,p2,lambda, nu; sigma=1.0)
    d = abs(p1-p2)
    if iszero(d)
        return 1.0
    end 
    scale = sigma^2 * 2^(1 - nu) / gamma(nu)
    factor = (sqrt(2 * nu) * d / lambda)
    result = scale * factor^nu * besselk(nu, factor)
    # @printf("x1: %.4f, x2: %.4f, d: %.4f, factor: %.4f, besselk: %.4e, result: %.4e\n", p1, p2, d, factor, besselk(nu, factor), result)
    return result
end

function cov_matrix_entry(z1,z2,C)
    @assert C == C' "C is not symmetric"
    i = Int(z1)
    j = Int(z2)
    return C[i,j]
end


# Construct individual covariance matrices
function construct_cov_matrix(x_pts, y_pts, z_pts, lambda, nu, C)

    n = length(x_pts)
    m = length(y_pts)
    k = length(z_pts)

    Kx = [matern_kernel(x_pts[i],x_pts[j], lambda, nu) for i in 1:n, j in 1:n]
    Ky = [matern_kernel(y_pts[i],y_pts[j], lambda, nu) for i in 1:m, j in 1:m]
    Kz = [cov_matrix_entry(z_pts[i], z_pts[j], C) for i in 1:k, j in 1:k]

    # Ensure Kx, Ky, and Kz are positive definite
    Kx += 1e-10 * I(n)
    eigvals_Kx = eigen(Kx).values
    @assert all(eigvals_Kx .> 0) "Kx is not positive definite"
    Ky += 1e-10 * I(m)
    eigvals_Ky = eigen(Ky).values
    @assert all(eigvals_Ky .> 0) "Ky is not positive definite"
    Kz += 1e-10 * I(k)
    eigvals_Kz = eigen(Kz).values
    @assert all(eigvals_Kz .> 0) "Kx is not positive definite"

    # Combine covariance matrices using Kronecker product
    K = kron(kron(Kx, Ky), Kz)

    return K
end


cov_K = construct_cov_matrix(x_pts, y_pts, z_pts, lambda, nu, symmetric_C)
eigvals_cov_K = eigen(cov_K).values 
@assert all(eigvals_cov_K .> 0) "cov K is not positive definite"

cor_K = construct_cov_matrix(x_pts,y_pts,z_pts, lambda, nu, correlation_matrix)
eigvals_cor_K = eigen(cor_K).values 
@assert all(eigvals_cor_K .> 0) "cor K is not positive definite"


#For each location
for loc in 1:3
    #Make mean vector
    true_x = true_CO2s[loc]
    mean = zeros(1280)

    for i in 1:1280
        idx = i % 20
        if iszero(idx)
            mean[i] = true_x[20]
        else
            mean[i] = true_x[idx]
        end
    end


    grf = MvNormal(mean, cor_K)
    sample_vec = rand(grf)

    #Construct tensor from sample & save it
    n = length(x_pts)
    m = length(y_pts)
    k = length(z_pts)
    sample_tensor = zeros(n,m,k)

    j = 1
    for x in 1:n
        for y in 1:m
            for z in 1:k
                sample_tensor[x,y,z] = sample_vec[j]
                j+=1
            end
        end
    end

    # h5write(joinpath(sample_dir,"$(labels[loc])_CO2_corGRF.h5"), "CO2tensor", sample_tensor)

    #Generate plots from tensor
    for level in 1:k
        level_plt = heatmap(sample_tensor[:,:,level],clims=(375,425), title="$(labels[loc]) GRF Sounding Level $(level)", xlabel="X", ylabel="Y", colorbar_title="CO2 Concentration (ppm)")
        display(level_plt)
        savefig(level_plt, joinpath(plots_dir, "$(labels[loc])_CO2_corGRF_Level$(level)"))
    end

    for x in 1:n
        x_plt = heatmap(sample_tensor[x,:,:], clims=(375,425), title="$(labels[loc]) GRF Sounding X = $(x)", xlabel="Level", ylabel="Y", colorbar_title="CO2 Concentration (ppm)")
        display(x_plt)
        savefig(x_plt, joinpath(plots_dir, "$(labels[loc])_CO2_corGRF_X$(x)"))
    end

    for y in 1:m
        y_plt = heatmap(sample_tensor[:,y,:], clims=(375,425), title="$(labels[loc]) GRF Sounding Y = $(y)", xlabel="Level", ylabel="X", colorbar_title="CO2 Concentration (ppm)")
        display(y_plt)
        savefig(y_plt, joinpath(plots_dir, "$(labels[loc])_CO2_corGRF_Y$(y)"))
    end
end