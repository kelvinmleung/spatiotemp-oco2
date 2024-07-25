using PyCall
np = pyimport("numpy")
using LinearAlgebra
using Distributions
using Serialization
using GaussianRandomFields
using Plots
default()
using HDF5

plots_dir = "/Users/Camila/Desktop/OCO-2_UROP/spatiotemp-oco2/Plots/Aerosol_GRF"
sample_dir = "/Users/Camila/Desktop/OCO-2_UROP/spatiotemp-oco2/GRF"


#Lamont 2015
SO_file = h5open("/Users/Camila/Desktop/OCO-2_UROP/spatiotemp-oco2/OCO2_Data-Lamont2015/Lamont2015_SO_Params.h5", "r")

OC_file = h5open("/Users/Camila/Desktop/OCO-2_UROP/spatiotemp-oco2/OCO2_Data-Lamont2015/Lamont2015_OC_Params.h5", "r")

Ice_file = h5open("/Users/Camila/Desktop/OCO-2_UROP/spatiotemp-oco2/OCO2_Data-Lamont2015/Lamont2015_Ice_Params.h5", "r")

Water_file = h5open("/Users/Camila/Desktop/OCO-2_UROP/spatiotemp-oco2/OCO2_Data-Lamont2015/Lamont2015_water_Params.h5", "r")

Lamont2015_Lambdas = [20, 5,8,28]


footprints = 8
x_pts = range(0.65,step=1.3, length=8)
y_pts = range(1.15, step=2.3, length=8)

# for loc in 1:3
#     loc_pds = pds[loc]
#     loc_lambdas = lambdas[loc]
#     loc_aero_labels = aerosol_labels[loc]
#     for aerosol in 1:4
#         pd = loc_pds[aerosol]
#         l = loc_lambdas[aerosol]
#         cov_func = CovarianceFunction(2, Matern(l, 2.0)) # length scale l, smoothness 1
#         grf = GaussianRandomField(pd, cov_func, Spectral(), x_pts, y_pts)
#         sample_matrix = GaussianRandomFields.sample(grf)
#         h5write(joinpath(sample_dir,"$(loc_labels[loc])_PD_$(loc_aero_labels[aerosol])_GRF.h5"), "PDmatrix", sample_matrix)
#         plt = heatmap(sample_matrix, 
#         colorbar=:bottom, 
#         colorbar_title="Log Profile Depth (km)", 
#         plot_title="$(loc_labels[loc]) $(loc_aero_labels[aerosol]) Log Profile Depth GRF", 
#         size=(600,500))
#         display(plt)
#         savefig(plt, joinpath(plots_dir, "$(loc_labels[loc])_pd_$(loc_aero_labels[aerosol])_GRF.png"))
#     end
# end

SO_param2_mean = mean(read(SO_file["ParamTensor"])[:,:,2])
SO_cov_func = CovarianceFunction(2, Matern(Lamont2015_Lambdas[1], 0.25, σ = 0.1)) # length scale l, smoothness 1
SO_grf = GaussianRandomField(SO_cov_func, Spectral(), x_pts, y_pts)
SO_grf_sample = GaussianRandomFields.sample(SO_grf)
SO_sample = SO_grf_sample .+ SO_param2_mean
# h5write(joinpath(sample_dir,"Lamont2015_SO_GRF.h5"), "Param2-Final", SO_sample)
SO_plt = heatmap(SO_sample, 
    title="Lamont2015 SO Aerosol Param 2 GRF Sample", 
    size=(700,500),
    clims=(0.75, 1))
display(SO_plt)
# savefig(SO_plt, joinpath(plots_dir, "Lamont2015_SO_Param2_GRF.png"))


OC_param2_mean = mean(read(OC_file["ParamTensor"])[:,:,2])
OC_cov_func = CovarianceFunction(2, Matern(Lamont201