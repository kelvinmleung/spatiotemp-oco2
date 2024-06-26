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
#Load true x
numpy_true_x = np.load("/Users/Camila/Desktop/OCO-2_UROP/spatiotemp-oco2/Lamont-2015/true_state_vector_2015-10_lamont.npy")
Lamont2015_true_xCO2 = convert(Array{Float64}, numpy_true_x)[1:20]

# Wollongong 2016
numpy_true_x = np.load("Wollongong-2016/true_state_vector_Wollongong2016.npy")
Wollongong2016_true_xCO2 = convert(Array{Float64}, numpy_true_x)[1:20]

#Wollongong 2017
numpy_true_x = np.load("Wollongong-2017/true_state_vector_Wollongong2017.npy")
Wollongong2017_true_xCO2 = convert(Array{Float64}, numpy_true_x)[1:20]

labels = ["Lamont-2015", "Wollongong-2016", "Wollongong-2017"]
true_xs = [Lamont2015_true_xCO2, Wollongong2016_true_xCO2, Wollongong2017_true_xCO2]


diffusion_coef = 16.0/(10^6) # km^2 / s^-1
t = 1/3 #interval between soundings in seconds
l = sqrt(diffusion_coef * t) #km
levels = 20
footprints = 100
cov = CovarianceFunction(1, Matern(l, 1)) # length scale l, smoothness 1
pts = range(0,10.4, footprints)


for loc in 1:3
    true_xCO2 = true_xs[loc] 
    #Create Gaussian random field for each level 
    grf_samples = zeros(levels, footprints)

    for i in 1:levels
        grf = GaussianRandomField(true_xCO2[i],cov, CirculantEmbedding(), pts, minpadding=201)
        sample_vector = sample(grf)
        for j in 1:footprints
            grf_samples[i,j] = sample_vector[j]
        end 
    end

    #Visualize samples at each of the 8 locations
    plt = heatmap(grf_samples, xlabel="Footprint", ylabel="Vertical Level", title="$(labels[loc]) Simulated Sounding", colorbar=true)
    display(plt)
    savefig(plt, joinpath(plots_dir, "$(labels[loc])_simulated_xCO2.png"))

    #Save grf samples
    file_path = joinpath(sample_dir, "$(labels[loc])_simulated_xCO2.jls")

    open(file_path, "w") do io
        serialize(io, grf_samples)
    end
end