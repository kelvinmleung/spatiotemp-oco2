using PyCall
np = pyimport("numpy")
using LinearAlgebra
using Distributions
using Serialization
using GaussianRandomFields
using Plots

plots_dir = "/Users/Camila/Desktop/OCO-2_UROP/spatiotemp-oco2/Plots/GRF"
sample_dir = "/Users/Camila/Desktop/OCO-2_UROP/spatiotemp-oco2/Gaussian-Random_Fields"

#Lamont 2015
numpy_true_x = np.load("/Users/Camila/Desktop/OCO-2_UROP/spatiotemp-oco2/Lamont-2015/true_state_vector_2015-10_lamont.npy")
Lamont2015_true_SP = convert(Array{Float64}, numpy_true_x)[21]

# Wollongong 2016
numpy_true_x = np.load("Wollongong-2016/true_state_vector_Wollongong2016.npy")
Wollongong2016_true_SP = convert(Array{Float64}, numpy_true_x)[21]

#Wollongong 2017
numpy_true_x = np.load("Wollongong-2017/true_state_vector_Wollongong2017.npy")
Wollongong2017_true_SP = convert(Array{Float64}, numpy_true_x)[21]

labels = ["Lamont2015", "Wollongong2016", "Wollongong2017"]
true_sps = [Lamont2015_true_SP, Wollongong2016_true_SP, Wollongong2017_true_SP]

l = 10
footprints = 8
cov_func = CovarianceFunction(2, Matern(l, 2.0)) # length scale l, smoothness 1
x_pts = range(0.65,step=1.3, length=8)
y_pts = range(1.15, step=2.3, length=8)

for loc in 1:3
    true_SP = true_sps[loc]
    grf = GaussianRandomField(true_SP, cov_func, Spectral(), x_pts, y_pts)
    sample_matrix = GaussianRandomFields.sample(grf)
    # h5write(joinpath(sample_dir,"$(labels[loc])_SP_GRF.h5"), "SPmatrix", sample_matrix)
    plt = heatmap(grf, title="$(labels[loc]) GRF Sounding", colorbar_title="Surface Pressure (hPa)")
    display(plt)
    savefig(plt, joinpath(plots_dir, "$(labels[loc])_simulated_SP.png"))

end