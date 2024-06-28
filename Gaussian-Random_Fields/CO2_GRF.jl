using PyCall
np = pyimport("numpy")
using LinearAlgebra
using Distributions
using Serialization
using GaussianRandomFields
using Plots
using SpecialFunctions
using Printf

plots_dir = "/Users/Camila/Desktop/OCO-2_UROP/spatiotemp-oco2/Plots"
sample_dir = "/Users/Camila/Desktop/OCO-2_UROP/spatiotemp-oco2/Gaussian-Random_Fields"


#Import known values
#Lamont 2015
numpy_true_x = np.load("/Users/Camila/Desktop/OCO-2_UROP/spatiotemp-oco2/Lamont-2015/true_state_vector_2015-10_lamont.npy")
Lamont2015_true_xCO2 = convert(Array{Float64}, numpy_true_x)[1:20]

# Wollongong 2016
numpy_true_x = np.load("Wollongong-2016/true_state_vector_Wollongong2016.npy")
Wollongong2016_true_xCO2 = convert(Array{Float64}, numpy_true_x)[1:20]

#Wollongong 2017
numpy_true_x = np.load("Wollongong-2017/true_state_vector_Wollongong2017.npy")
Wollongong2017_true_xCO2 = convert(Array{Float64}, numpy_true_x)[1:20]

#Cov matrix
numpy_C = np.load("/Users/Camila/Desktop/OCO-2_UROP/spatiotemp-oco2/Lamont-2015/prior_cov_matrix_2015-10_lamont.npy")
C = convert(Array{Float64}, numpy_C)[1:20, 1:20]


#Constants
cov_matrix  = zeros(1280,1280)
diffusion_coef = 16.0 / (10^6)  # km^2 / s^-1
t = 1  # interval between soundings in seconds
lambda = sqrt(diffusion_coef * t)  # km
nu = 2.0
x_pts = collect(range(1.3,step=1.3, length=8))


function matern_kernel(p1,p2,lambda, nu; sigma=1.0)
    println("p1 $(p1)")
    println("p2 $(p2)")
    d = abs(p1-p2)
    if iszero(d)
        return 1.0
    end 
    scale = sigma^2 * 2^(1 - nu) / gamma(nu)
    factor = (sqrt(2 * nu) * d / lambda)
    result = scale * factor^nu * besselk(nu, factor)
    @printf("x1: %.4f, x2: %.4f, d: %.4f, factor: %.4f, besselk: %.4e, result: %.4e\n", p1, p2, d, factor, besselk(nu, factor), result)
    return result
end

#fill cov matrix with matern covariance for x and y
for i in eachindex(x_pts)
    for j in eachindex(x_pts)
        x1 = x_pts[i]
        x2 = x_pts[j]
        cov = matern_kernel(x1, x2, lambda, nu)
        cov_matrix[i, j] = cov
    end
end

cov_matrix[1:8,1:8]   