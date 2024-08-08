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
numpy_state_vec = np.load("SampleState-Lamont2015/true_state_vector_2015-10_lamont.npy")
state_vec = convert(Array{Float64}, numpy_state_vec)
param2_indices = [29,32,35,38]
param2 = state_vec[param2_indices]




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
OC_cov_func = CovarianceFunction(2, Matern(Lamont2015_Lambdas[2], 0.5, σ = 0.05)) # length scale l, smoothness 1
OC_grf = GaussianRandomField(OC_cov_func, Spectral(), x_pts, y_pts)
OC_grf_sample = GaussianRandomFields.sample(OC_grf)
OC_sample = OC_grf_sample .+ OC_param2_mean
# h5write(joinpath(sample_dir,"Lamont2015_OC_GRF.h5"), "Param2", OC_sample)
OC_plt = heatmap(OC_sample, 
    title="Lamont2015 OC Aerosol Param 2 GRF Sample", 
    size=(700,500),
    clims=(0.8,0.93))
display(OC_plt)
# savefig(OC_plt, joinpath(plots_dir, "Lamont2015_OC_Param2_GRF.png"))


Ice_param2_mean = mean(read(Ice_file["ParamTensor"])[:,:,2])
Ice_cov_func = CovarianceFunction(2, Matern(Lamont2015_Lambdas[3], 0.5, σ = 0.15)) # length scale l, smoothness 1
Ice_grf = GaussianRandomField(Ice_cov_func, Spectral(), x_pts, y_pts)
Ice_grf_sample = GaussianRandomFields.sample(Ice_grf)
Ice_sample = Ice_grf_sample .+ Ice_param2_mean
# h5write(joinpath(sample_dir,"Lamont2015_Ice_GRF.h5"), "Param2", Ice_sample)
Ice_plt = heatmap(Ice_sample, 
    title="Lamont2015 Ice Aerosol Param 2 GRF Sample", 
    size=(700,500), 
    clims=(-0.1,0.2))
display(Ice_plt)
# savefig(Ice_plt, joinpath(plots_dir, "Lamont2015_Ice_Param2_GRF.png"))

Water_param2_mean = mean(read(Water_file["ParamTensor"])[:,:,2])
Water_cov_func = CovarianceFunction(2, Matern(Lamont2015_Lambdas[4], 0.1, σ = 0.05)) # length scale l, smoothness 1
Water_grf = GaussianRandomField(Water_cov_func, Spectral(), x_pts, y_pts)
Water_grf_sample = GaussianRandomFields.sample(Water_grf)
Water_sample = Water_grf_sample .+ Water_param2_mean
# h5write(joinpath(sample_dir,"Lamont2015_Water_GRF.h5"), "Param2", Water_sample)
Water_plt = heatmap(Water_sample, 
    title="Lamont2015 Water Aerosol Param 2 GRF Sample", 
    size=(700,500), 
    clims=(0,1.25))
display(Water_plt)
# savefig(Water_plt, joinpath(plots_dir, "Lamont2015_Water_Param2_GRF.png"))
