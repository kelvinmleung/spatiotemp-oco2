using PyCall
np = pyimport("numpy")
using LinearAlgebra
using Distributions
using Serialization
using GaussianRandomFields
using Plots
default()
using HDF5

plots_dir = "/Users/Camila/Desktop/OCO-2_UROP/spatiotemp-oco2/Plots/Aerosol_Param1_GRF"
sample_dir = "/Users/Camila/Desktop/OCO-2_UROP/spatiotemp-oco2/GRF"

aod_indices = [28, 31, 34, 37]

#Lamont 2015
numpy_true_x = np.load("/Users/Camila/Desktop/OCO-2_UROP/spatiotemp-oco2/SampleState-Lamont2015/true_state_vector_2015-10_lamont.npy")
Lamont2015_x = convert(Array{Float64}, numpy_true_x)
Lamont2015_AODs = (Lamont2015_x[aod_indices])
Lamont2015_lambdas = [36.5, 100, 44, 37.5]


footprints = 8
x_pts = range(0.65,step=1.3, length=8)
y_pts = range(1.15, step=2.3, length=8)

# for loc in 1:3
#     loc_AODs = AODs[loc]
#     loc_lambdas = lambdas[loc]
#     loc_aero_labels = aerosol_labels[loc]
#     for aerosol in 1:4
#         log_opt_depth = loc_AODs[aerosol]
#         l = loc_lambdas[aerosol]
#         cov_func = CovarianceFunction(2, Matern(l, 1.0)) # length scale l, smoothness 1
#         grf = GaussianRandomField(log_opt_depth, cov_func, Spectral(), x_pts, y_pts)
#         sample_matrix = GaussianRandomFields.sample(grf)
#         # h5write(joinpath(sample_dir,"$(loc_labels[loc])_OD_$(loc_aero_labels[aerosol])_GRF.h5"), "ODmatrix", sample_matrix)
#         plt = heatmap(sample_matrix, 
#         plot_title="Lamont2015 $(Lamont_aerosol_labels[aerosol]) Param $aer GRF", 
#         size=(700,500))
#         display(plt)
#         # savefig(plt, joinpath(plots_dir, "$(loc_labels[loc])_OD_$(loc_aero_labels[aerosol])_GRF.png"))
#     end
# end


SO_logOD = Lamont2015_AODs[1]
SO_cov_func = CovarianceFunction(2, Matern(125, 0.5)) # length scale l, smoothness 1
SO_grf = GaussianRandomField(SO_logOD, SO_cov_func, Spectral(), x_pts, y_pts)
SO_sample = GaussianRandomFields.sample(SO_grf)
# h5write(joinpath(sample_dir,"Lamont2015_SO_GRF.h5"), "Param1", SO_sample)
SO_plt = heatmap(SO_sample, 
    plot_title="Lamont2015 SO Aerosol Param 1 GRF Sample", 
    size=(700,500))
display(SO_plt)
# savefig(SO_plt, joinpath(plots_dir, "Lamont2015_SO_Param1_GRF.png"))

OC_logOD = Lamont2015_AODs[2]
OC_cov_func = CovarianceFunction(2, Matern(125, 0.5)) # length scale l, smoothness 1
OC_grf = GaussianRandomField(OC_logOD, OC_cov_func, Spectral(), x_pts, y_pts)
OC_sample = GaussianRandomFields.sample(OC_grf)
# h5write(joinpath(sample_dir,"Lamont2015_OC_GRF.h5"), "Param1", OC_sample)
OC_plt = heatmap(OC_sample, 
    plot_title="Lamont2015 OC Aerosol Param 1 GRF Sample", 
    size=(700,500))
display(OC_plt)
# savefig(OC_plt, joinpath(plots_dir, "Lamont2015_OC_Param1_GRF.png"))


Ice_logOD = Lamont2015_AODs[3]
Ice_cov_func = CovarianceFunction(2, Matern(44, 0.1)) # length scale l, smoothness 1
Ice_grf = GaussianRandomField(Ice_logOD, Ice_cov_func, Spectral(), x_pts, y_pts)
Ice_sample = GaussianRandomFields.sample(Ice_grf)
# h5write(joinpath(sample_dir,"Lamont2015_Ice_GRF.h5"), "Param1", Ice_sample)
Ice_plt = heatmap(Ice_sample, 
    plot_title="Lamont2015 Ice Aerosol Param 1 GRF Sample", 
    size=(700,500), 
    clims=(-8,-5))
display(Ice_plt)
# savefig(Ice_plt, joinpath(plots_dir, "Lamont2015_Ice_Param1_GRF.png"))

Water_logOD = Lamont2015_AODs[4]
Water_cov_func = CovarianceFunction(2, Matern(37.5, 0.1)) # length scale l, smoothness 1
Water_grf = GaussianRandomField(Water_logOD, Water_cov_func, Spectral(), x_pts, y_pts)
Water_sample = GaussianRandomFields.sample(Water_grf)
# h5write(joinpath(sample_dir,"Lamont2015_Water_GRF.h5"), "Param1", Water_sample)
Water_plt = heatmap(Water_sample, 
    title="Lamont2015 Water Aerosol Param 1 GRF Sample", 
    size=(700,500), 
    clims=(-6,-2))
display(Water_plt)
savefig(Water_plt, joinpath(plots_dir, "Lamont2015_Water_Param1_GRF.png"))



