using PyCall
np = pyimport("numpy")
using LinearAlgebra
using Distributions
using Serialization
using GaussianRandomFields
using Plots
# pyplot()
using HDF5

plots_dir = "/Users/Camila/Desktop/OCO-2_UROP/spatiotemp-oco2/Plots/PH_GRF"
sample_dir = "/Users/Camila/Desktop/OCO-2_UROP/spatiotemp-oco2/Gaussian-Random_Fields"
ph_indices = [29, 32, 35, 38]


#Lamont 2015
numpy_true_x = np.load("/Users/Camila/Desktop/OCO-2_UROP/spatiotemp-oco2/SampleState-Lamont2015/true_state_vector_2015-10_lamont.npy")
Lamont2015_x = convert(Array{Float64}, numpy_true_x)
Lamont2015_phs = Lamont2015_x[ph_indices]
Lamont2015_lambdas = [20,7.5,6,15.5]

# Wollongong 2016

numpy_true_x = np.load("Wollongong-2016/true_state_vector_Wollongong2016.npy")
Wollongong2016_x = convert(Array{Float64}, numpy_true_x)
Wollongong2016_phs = Wollongong2016_x[ph_indices]
Wollongong2016_lambdas = [10,30.5,18.5,32.5]

#Wollongong 2017
numpy_true_x = np.load("Wollongong-2017/true_state_vector_Wollongong2017.npy")
Wollongong2017_x = convert(Array{Float64}, numpy_true_x)
Wollongong2017_phs = Wollongong2017_x[ph_indices]
Wollongong2017_lambdas = [10,31.5,3,19]

phs = [Lamont2015_phs, Wollongong2016_phs, Wollongong2017_phs]
lambdas = [Lamont2015_lambdas, Wollongong2016_lambdas, Wollongong2017_lambdas]
loc_labels = ["Lamont2015", "Wollongong2016", "Wollongong2017"]
Lamont_aerosol_labels = ["Sulfate", "Dust", "Cloud", "Ice"]
Wollongong2016_aerosol_labels = ["Sulfate", "Sea Salt", "Cloud", "Ice"]
Wollongong2017_aerosol_labels = ["Sulfate", "Sea Salt", "Cloud", "Ice"]
aerosol_labels = [Lamont_aerosol_labels, Wollongong2016_aerosol_labels, Wollongong2017_aerosol_labels]

footprints = 8
x_pts = range(0.65,step=1.3, length=8)
y_pts = range(1.15, step=2.3, length=8)

for loc in 1:3
    loc_phs = phs[loc]
    loc_lambdas = lambdas[loc]
    loc_aero_labels = aerosol_labels[loc]
    for aerosol in 1:4
        ph = loc_phs[aerosol]
        l = loc_lambdas[aerosol]
        cov_func = CovarianceFunction(2, Matern(l, 2.0)) # length scale l, smoothness 1
        grf = GaussianRandomField(ph, cov_func, Spectral(), x_pts, y_pts)
        sample_matrix = GaussianRandomFields.sample(grf)
        h5write(joinpath(sample_dir,"$(loc_labels[loc])_PH_$(loc_aero_labels[aerosol])_GRF.h5"), "PHmatrix", sample_matrix)
        plt = heatmap(sample_matrix, 
        colorbar=:bottom, 
        colorbar_title="Profile Height (km)", 
        plot_title="$(loc_labels[loc]) $(loc_aero_labels[aerosol]) Profile Height GRF", 
        size=(600,500))
        display(plt)
        savefig(plt, joinpath(plots_dir, "$(loc_labels[loc])_PH_$(loc_aero_labels[aerosol])_GRF.png"))
    end
end


