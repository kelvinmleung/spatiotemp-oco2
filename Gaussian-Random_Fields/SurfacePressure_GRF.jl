using PyCall
np = pyimport("numpy")
using LinearAlgebra
using Distributions
using Serialization
using GaussianRandomFields
using Plots
pyplot()


plots_dir = "/Users/Camila/Desktop/OCO-2_UROP/spatiotemp-oco2/Plots/SP_GRF"
sample_dir = "/Users/Camila/Desktop/OCO-2_UROP/spatiotemp-oco2/Gaussian-Random_Fields"

#Lamont 2015
numpy_true_x = np.load("/Users/Camila/Desktop/OCO-2_UROP/spatiotemp-oco2/Lamont-2015/true_state_vector_2015-10_lamont.npy")
Lamont2015_true_SP = convert(Array{Float64}, numpy_true_x)[21]
Lamont_lambda = 43.75


# Wollongong 2016
numpy_true_x = np.load("Wollongong-2016/true_state_vector_Wollongong2016.npy")
Wollongong2016_true_SP = convert(Array{Float64}, numpy_true_x)[21]
Wollongong2016_lambda = 10.5


#Wollongong 2017
numpy_true_x = np.load("Wollongong-2017/true_state_vector_Wollongong2017.npy")
Wollongong2017_true_SP = convert(Array{Float64}, numpy_true_x)[21]
Wollongong2017_lambda = 12.5

labels = ["Lamont2015", "Wollongong2016", "Wollongong2017"]
true_sps = [Lamont2015_true_SP, Wollongong2016_true_SP, Wollongong2017_true_SP]
lambdas = [Lamont_lambda, Wollongong2016_lambda, Wollongong2017_lambda]

footprints = 8
x_pts = range(0.65,step=1.3, length=8)
y_pts = range(1.15, step=2.3, length=8)

for loc in 1:3
    true_SP = true_sps[loc]
    l = lambdas[loc]
    cov_func = CovarianceFunction(2, Matern(l, 2.0)) # length scale l, smoothness 1
    grf = GaussianRandomField(true_SP, cov_func, Spectral(), x_pts, y_pts)
    sample_matrix = GaussianRandomFields.sample(grf)
    h5write(joinpath(sample_dir,"$(labels[loc])_SP_GRF.h5"), "SPmatrix", sample_matrix)
    plt = heatmap(sample_matrix, 
    colorbar=:bottom, 
    colorbar_title="Surface Pressure (hPa)", 
    plot_title="$(labels[loc]) Surface Pressure GRF", 
    size=(600,500))
    display(plt)
    savefig(plt, joinpath(plots_dir, "$(labels[loc])_SP_GRF.png"))

end


# sp = true_sps[1]
# l = lambdas[1]
# cov_func = CovarianceFunction(2, Matern(l, 2.0))
# grf = GaussianRandomField(sp, cov_func, Spectral(), x_pts,y_pts)
# sample = GaussianRandomFields.sample(grf)
# plt = heatmap(sample, colorbar=:bottom, colorbar_title="Surface Pressure (hPa)", plot_title="Surface Pressure GRF Lamont2015")
# display(plt)