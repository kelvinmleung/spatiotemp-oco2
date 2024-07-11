using PyCall
np = pyimport("numpy")
using LinearAlgebra
using Distributions
using Serialization
using GaussianRandomFields
using Plots

plots_dir = "/Users/Camila/Desktop/OCO-2_UROP/spatiotemp-oco2/Plots/GRF"
sample_dir = "/Users/Camila/Desktop/OCO-2_UROP/spatiotemp-oco2/Gaussian-Random_Fields"

aod_indices = [27, 30, 33, 36]

#Lamont 2015
numpy_true_x = np.load("/Users/Camila/Desktop/OCO-2_UROP/spatiotemp-oco2/Lamont-2015/true_state_vector_2015-10_lamont.npy")
Lamont2015_x = convert(Array{Float64}, numpy_true_x)
Lamont2015_AODs = Lamont2015_x[aod_indices]

# Wollongong 2016

numpy_true_x = np.load("Wollongong-2016/true_state_vector_Wollongong2016.npy")
Wollongong2016_x = convert(Array{Float64}, numpy_true_x)
Wollongong2016_AODs = Wollongong2016_x[aod_indices]

#Wollongong 2017
numpy_true_x = np.load("Wollongong-2017/true_state_vector_Wollongong2017.npy")
Wollongong2017_x = convert(Array{Float64}, numpy_true_x)
Wollongong2017_AODs = Wollongong2017_x[aod_indices]