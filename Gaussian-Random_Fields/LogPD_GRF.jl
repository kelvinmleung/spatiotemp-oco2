using PyCall
np = pyimport("numpy")
using LinearAlgebra
using Distributions
using Serialization
using GaussianRandomFields
using Plots
# pyplot()
using HDF5

plots_dir = "/Users/Camila/Desktop/OCO-2_UROP/spatiotemp-oco2/Plots/pd_GRF"
sample_dir = "/Users/Camila/Desktop/OCO-2_UROP/spatiotemp-oco2/Gaussian-Random_Fields"
pd_indices = [30, 33, 36, 39]


#Lamont 2015
numpy_true_x = np.load("/Users/Camila/Desktop/OCO-2_UROP/spatiotemp-oco2/Lamont-2015/true_state_vector_2015-10_lamont.npy")
Lamont2015_x = convert(Array{Float64}, numpy_true_x)
Lamont2015_pds = Lamont2015_x[pd_indices]
Lamont2015_lambdas = [39,100,22,8]

# Wollongong 2016

numpy_true_x = np.load("Wollongong-2016/true_state_vector_Wollongong2016.npy")
Wollongong2016_x = convert(Array{Float64}, numpy_true_x)
Wollongong2016_pds = Wollongong2016_x[pd_indices]
Wollongong2016_lambdas = [9.5,17.5, 17,28]

#Wollongong 2017
numpy_true_x = np.load("Wollongong-2017/true_state_vector_Wollongong2017.npy")
Wollongong2017_x = convert(Array{Float64}, numpy_true_x)
Wollongong2017_pds = Wollongong2017_x[pd_indices]
Wollongong2017_lambdas = [10,7.5,37.5,30]

pds = [Lamont2015_pds, Wollongong2016_pds, Wollongong2017_pds]
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
    loc_pds = pds[loc]
    loc_lambdas = lambdas[loc]
    loc_aero_labels = aerosol_labels[loc]
    for aerosol in 1:4
        pd = loc_pds[aerosol]
        l = loc_lambdas[aerosol]
        cov_func = CovarianceFunction(2, Matern(l, 2.0)) # length scale l, smoothness 1
        grf = GaussianRandomField(pd, cov_func, Spectral(), x_pts, y_pts)
        sample_matrix = GaussianRandomFields.sample(grf)
        h5write(joinpath(sample_dir,"$(loc_labels[loc])_PD_$(loc_aero_labels[aerosol])_GRF.h5"), "PDmatrix", sample_matrix)
        plt = heatmap(sample_matrix, 
        colorbar=:bottom, 
        colorbar_title="Log Profile Depth (km)", 
        plot_title="$(loc_labels[loc]) $(loc_aero_labels[aerosol]) Log Profile Depth GRF", 
        size=(600,500))
        display(plt)
        savefig(plt, joinpath(plots_dir, "$(loc_labels[loc])_pd_$(loc_aero_labels[aerosol])_GRF.png"))
    end
end
