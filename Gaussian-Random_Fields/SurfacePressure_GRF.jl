using PyCall
np = pyimport("numpy")
using LinearAlgebra
using Distributions
using Serialization
using GaussianRandomFields
using Plots

plots_dir = "/Users/Camila/Desktop/OCO-2_UROP/spatiotemp-oco2/Plots"
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

labels = ["Lamont-2015", "Wollongong-2016", "Wollongong-2017"]
true_sps = [Lamont2015_true_SP, Wollongong2016_true_SP, Wollongong2017_true_SP]

l = 10
footprints = 100
cov = CovarianceFunction(1, Matern(l, 1.25)) # length scale l, smoothness 1
pts = range(0,10.4, footprints)

for loc in 1:3
    true_SP = true_sps[loc]
    grf = GaussianRandomField(true_SP, cov, Spectral(), pts)
    grf_samples = zeros(footprints)
    sample_vector = sample(grf)
    for j in 1:footprints
        grf_samples[j] = sample_vector[j]
    end
    x_axis = 1:footprints
    plt = plot(x_axis, grf_samples, ylabel="Surface Pressure hPa", xlabel="Footprint", title="$(labels[loc]) Simulated Surface Pressure Sounding", legend=false)
    display(plt)
    savefig(plt, joinpath(plots_dir, "$(labels[loc])_simulated_SP.png"))

    #Save grf samples
    file_path = joinpath(sample_dir, "$(labels[loc])_simulated_SP.jls")
    open(file_path, "w") do io
        serialize(io, grf_samples)
    end

end