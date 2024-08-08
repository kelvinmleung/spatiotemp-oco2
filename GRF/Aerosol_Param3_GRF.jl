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

Lamont2015_lambdas = [20,7.5,6,15.5]

footprints = 8
x_pts = range(0.65,step=1.3, length=8)
y_pts = range(1.15, step=2.3, length=8)


SO_param3_mean = mean(read(SO_file["ParamTensor"])[:,:,3])
SO_cov_func = CovarianceFunction(2, Matern(Lamont2015_Lambdas[1], 0.5, σ = 0.0005)) # length scale l, smoothness 1
SO_grf = GaussianRandomField(SO_cov_func, Spectral(), x_pts, y_pts)
SO_grf_sample = GaussianRandomFields.sample(SO_grf)
SO_sample = SO_grf_sample .+ SO_param3_mean
# h5write(joinpath(sample_dir,"Lamont2015_SO_GRF.h5"), "Param3", SO_sample)
SO_plt = heatmap(SO_sample, 
    title="Lamont2015 SO Aerosol Param 3 GRF Sample", 
    size=(700,500),
    clims=(0.049, 0.051))
display(SO_plt)
# savefig(SO_plt, joinpath(plots_dir, "Lamont2015_SO_Param3_GRF.png"))


OC_param3_mean = mean(read(OC_file["ParamTensor"])[:,:,3])
OC_cov_func = CovarianceFunction(2, Matern(Lamont2015_Lambdas[2], 0.5, σ = 0.0001)) # length scale l, smoothness 1
OC_grf = GaussianRandomField(OC_cov_func, Spectral(), x_pts, y_pts)
OC_grf_sample = GaussianRandomFields.sample(OC_grf)
OC_sample = OC_grf_sample .+ OC_param3_mean
# h5write(joinpath(sample_dir,"Lamont2015_OC_GRF.h5"), "Param3", OC_sample)
OC_plt = heatmap(OC_sample, 
    title="Lamont2015 OC Aerosol Param 2 GRF Sample", 
    size=(700,500),
    clims=(0.0497,0.0503))
display(OC_plt)
# savefig(OC_plt, joinpath(plots_dir, "Lamont2015_OC_Param3_GRF.png"))


Ice_param3_mean = mean(read(Ice_file["ParamTensor"])[:,:,3])
Ice_cov_func = CovarianceFunction(2, Matern(Lamont2015_Lambdas[3], 0.1, σ = 0.00025)) # length scale l, smoothness 1
Ice_grf = GaussianRandomField(Ice_cov_func, Spectral(), x_pts, y_pts)
Ice_grf_sample = GaussianRandomFields.sample(Ice_grf)
Ice_sample = Ice_grf_sample .+ Ice_param3_mean
# h5write(joinpath(sample_dir,"Lamont2015_Ice_GRF.h5"), "Param3", Ice_sample)
Ice_plt = heatmap(Ice_sample, 
    title="Lamont2015 Ice Aerosol Param 2 GRF Sample", 
    size=(700,500), 
    clims=(0.0395,0.0402))
display(Ice_plt)
# savefig(Ice_plt, joinpath(plots_dir, "Lamont2015_Ice_Param3_GRF.png"))

Water_param3_mean = mean(read(Water_file["ParamTensor"])[:,:,3])
Water_cov_func = CovarianceFunction(2, Matern(Lamont2015_Lambdas[4], 0.1, σ = 0.00035)) # length scale l, smoothness 1
Water_grf = GaussianRandomField(Water_cov_func, Spectral(), x_pts, y_pts)
Water_grf_sample = GaussianRandomFields.sample(Water_grf)
Water_sample = Water_grf_sample .+ Water_param3_mean
# h5write(joinpath(sample_dir,"Lamont2015_Water_GRF.h5"), "Param3", Water_sample)
Water_plt = heatmap(Water_sample, 
    title="Lamont2015 Water Aerosol Param 3 GRF Sample", 
    size=(700,500), 
    clims=(0.0986,0.1007))
display(Water_plt)
# savefig(Water_plt, joinpath(plots_dir, "Lamont2015_Water_Param3_GRF.png"))

