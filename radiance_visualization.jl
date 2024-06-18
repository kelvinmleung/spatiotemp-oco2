include("plot_helpers.jl")
using PyCall
np = pyimport("numpy")
using LinearAlgebra
using Distributions
using Serialization
using Accessors

save_dir = "/Users/Camila/Desktop/OCO-2_UROP/spatiotemp-oco2/Plots/radiance-vs-params"

#Load forward model 
numpy_model = np.load("/Users/Camila/Desktop/OCO-2_UROP/spatiotemp-oco2/Lamont-2015/linear_model_2015-10_lamont.npy")
F_matrix = transpose(convert(Array{Float64}, numpy_model))

#Load true state vector from Lamont 2015
numpy_true_x = np.load("/Users/Camila/Desktop/OCO-2_UROP/spatiotemp-oco2/Lamont-2015/true_state_vector_2015-10_lamont.npy")
true_x = convert(Array{Float64}, numpy_true_x)

#Load error covariance from Lamont 2015
numpy_error_var = np.load("/Users/Camila/Desktop/OCO-2_UROP/spatiotemp-oco2/Lamont-2015/error_variance_2015-10_lamont.npy")
error_variance = convert(Array{Float64}, numpy_error_var)
#check if any of the elements of error_var are 0
@assert all(x -> x != 0, error_variance) "variance vector should not contain any zero elements."
error_cov_matrix = Diagonal(error_variance)

#Load wavelengths from Lamont 2015
numpy_wavs = np.load("/Users/Camila/Desktop/OCO-2_UROP/spatiotemp-oco2/wavelengths.npy")
wavelengths = convert(Array{Float64}, numpy_wavs)

# Sample from the error distribution
n_y = size(F_matrix)[1]
error_mean = zeros(Float64, n_y)
error_dist = MvNormal(error_mean, error_cov_matrix)
# Check if the file exists before loading sample_error
if isfile("/Users/Camila/Desktop/OCO-2_UROP/spatiotemp-oco2/Lamont-2015/sample_error_lamont2015.jls")
    # Load sample_error from the file
    sample_error = open(deserialize, "/Users/Camila/Desktop/OCO-2_UROP/spatiotemp-oco2/Lamont-2015/sample_error_lamont2015.jls")
else
    # Generate new sample_error if the file doesn't exist
    sample_error = rand(error_dist)
    
    # Save sample_error to a file
    open("/Users/Camila/Desktop/OCO-2_UROP/spatiotemp-oco2/Lamont-2015/sample_error_lamont2015.jls", "w") do io
        serialize(io, sample_error)
    end
end

#Baseline Radiances
Fx = F_matrix*true_x

strong_CO2_plt_Fx,weak_CO2_plt_Fx,O2_plt_Fx = plot_radiances(wavelengths,Fx)
display(strong_CO2_plt_Fx)
savefig(strong_CO2_plt_Fx, joinpath(save_dir, "StrongCO2Band_Fx.png"))
display(weak_CO2_plt_Fx)
savefig(weak_CO2_plt_Fx, joinpath(save_dir, "WeakCO2Band_Fx.png"))
display(O2_plt)
savefig(O2_plt_Fx, joinpath(save_dir, "O2Band_Fx.png"))



# Add errors 
sample_Y = Fx + sample_error

strong_CO2_plt_Y,weak_CO2_plt_Y,O2_plt_Y = plot_modified_radiances(wavelengths, Fx, sample_Y, "Fx + ϵ")
strongCO2_diff_Y, weakCO2_diff_Y, O2_diff_Y = plot_radiance_differences(wavelengths, Fx, sample_Y, "Fx + ϵ")
display(strong_CO2_plt_Y)
display(strongCO2_diff_Y)
savefig(strong_CO2_plt_Y, joinpath(save_dir, "StrongCO2Band_Y.png"))
savefig(strongCO2_diff_Y, joinpath(save_dir, "StrongCO2_diff_Y.png"))
display(weak_CO2_plt_Y)
display(weakCO2_diff_Y)
savefig(weak_CO2_plt_Y, joinpath(save_dir, "WeakCO2Band_Y.png"))
savefig(weakCO2_diff_Y, joinpath(save_dir, "WeakCO2_diff_Y.png"))
display(O2_plt_Y)
display(O2_diff_Y)
savefig(O2_plt_Y, joinpath(save_dir, "O2Band_Y.png"))
savefig(O2_diff_Y, joinpath(save_dir, "O2_diff_Y.png"))



#decrease the conc of CO2 by 100 ppm
diff = [100*ones(20); zeros(19)]

low_CO2 = true_x .- diff
lowCO2_Y = F_matrix * low_CO2

strong_CO2_plt_lowCO2,weak_CO2_plt_lowCO2, O2_plt_lowCO2 = plot_modified_radiances(wavelengths, Fx, lowCO2_Y, "-100 ppm CO2")
strongCO2_diff_lowCO2, weakCO2_diff_lowCO2, O2_diff_lowCO2 = plot_radiance_differences(wavelengths, Fx, lowCO2_Y, "-100 ppm CO2")
display(strong_CO2_plt_lowCO2)
display(strongCO2_diff_lowCO2)
savefig(strong_CO2_plt_lowCO2, joinpath(save_dir, "StrongCO2Band_lowCO2.png"))
savefig(strongCO2_diff_lowCO2, joinpath(save_dir, "StrongCO2_diff_lowCO2.png"))
display(weak_CO2_plt_lowCO2)
display(weakCO2_diff_lowCO2)
savefig(weak_CO2_plt_lowCO2, joinpath(save_dir, "WeakCO2Band_lowCO2.png"))
savefig(weakCO2_diff_lowCO2, joinpath(save_dir, "WeakCO2_diff_lowCO2.png"))
display(O2_plt_lowCO2)
display(O2_diff_lowCO2)
savefig(O2_plt_lowCO2, joinpath(save_dir, "O2Band_lowCO2.png"))
savefig(O2_diff_lowCO2, joinpath(save_dir, "O2_diff_lowCO2.png"))


#Increase conc of CO2 by 100 ppm
high_CO2 = true_x .+ diff
highCO2_Y = F_matrix * high_CO2

strong_CO2_plt_highCO2, weak_CO2_plt_highCO2, O2_plt_highCO2 = plot_modified_radiances(wavelengths, Fx, highCO2_Y, "+100 ppm CO2")
strongCO2_diff_highCO2, weakCO2_diff_highCO2, O2_diff_highCO2 = plot_radiance_differences(wavelengths, Fx, highCO2_Y, "+ 100 ppm CO2")
display(strong_CO2_plt_highCO2)
display(strongCO2_diff_highCO2)
savefig(strong_CO2_plt_highCO2, joinpath(save_dir, "StrongCO2Band_highCO2.png"))
savefig(strongCO2_diff_highCO2, joinpath(save_dir, "StrongCO2_diff_highCO2.png"))
display(weak_CO2_plt_highCO2)
display(weakCO2_diff_highCO2)
savefig(weak_CO2_plt_highCO2, joinpath(save_dir, "WeakCO2Band_highCO2.png"))
savefig(weakCO2_diff_highCO2, joinpath(save_dir, "WeakCO2_diff_highCO2.png"))
display(O2_plt_highCO2)
display(O2_diff_highCO2)
savefig(O2_plt_highCO2, joinpath(save_dir, "O2Band_highCO2.png"))
savefig(O2_diff_highCO2, joinpath(save_dir, "O2_diff_highCO2.png"))




#Increase surface pressure by 300 
diff = zeros(39)
diff[21] = 300
high_surface_pressure_x = true_x .+ diff
high_surface_pressure_Y = F_matrix * high_surface_pressure_x

strong_CO2_plt_highpressure,weak_CO2_plt_highpressure, O2_plt_highpressure = plot_modified_radiances(wavelengths,Fx, high_surface_pressure_Y, "Surface pressure +300 hPa")
strongCO2_diff_highpressure, weakCO2_diff_highpressure, O2_diff_highpressure = plot_radiance_differences(wavelengths, Fx, high_surface_pressure_Y, "Surface pressure +300 hPa")
display(strong_CO2_plt_highpressure)
display(strongCO2_diff_highpressure)
savefig(strong_CO2_plt_highpressure, joinpath(save_dir, "StrongCO2Band_highpressure.png"))
savefig(strongCO2_diff_highpressure, joinpath(save_dir, "StrongCO2_diff_highpressure.png"))
display(weak_CO2_plt_highpressure)
display(weakCO2_diff_highpressure)
savefig(weak_CO2_plt_highpressure, joinpath(save_dir, "WeakCO2Band_highpressure.png"))
savefig(weakCO2_diff_highpressure, joinpath(save_dir, "WeakCO2_diff_highpressure.png"))
display(O2_plt_highpressure)
display(O2_diff_highpressure)
savefig(O2_plt_highpressure, joinpath(save_dir, "O2Band_highpressure.png"))
savefig(O2_diff_highpressure, joinpath(save_dir, "O2_diff_highpressure.png"))





#Decrease surface pressure by 300 
low_surface_pressure_x = true_x - diff
println(low_surface_pressure_x[21])
low_surface_pressure_Y = F_matrix * low_surface_pressure_x

strong_CO2_plt_lowpressure,weak_CO2_plt_lowpressure, O2_plt_lowpressure = plot_modified_radiances(wavelengths,Fx, low_surface_pressure_Y, "Surface pressure -300 hPa")
strongCO2_diff_lowpressure, weakCO2_diff_lowpressure, O2_diff_lowpressure = plot_radiance_differences(wavelengths, Fx, low_surface_pressure_Y, "Surface pressure -300 hPa")
display(strong_CO2_plt_lowpressure)
display(strongCO2_diff_lowpressure)
savefig(strong_CO2_plt_lowpressure, joinpath(save_dir, "StrongCO2Band_lowpressure.png"))
savefig(strongCO2_diff_lowpressure, joinpath(save_dir, "StrongCO2_diff_lowpressure.png"))
display(weak_CO2_plt_lowpressure)
display(weakCO2_diff_lowpressure)
savefig(weak_CO2_plt_lowpressure, joinpath(save_dir, "WeakCO2Band_lowpressure.png"))
savefig(weakCO2_diff_lowpressure, joinpath(save_dir, "WeakCO2_diff_lowpressure.png"))
display(O2_plt_lowpressure)
display(O2_diff_lowpressure)
savefig(O2_plt_lowpressure, joinpath(save_dir, "O2Band_lowpressure.png"))
savefig(O2_diff_lowpressure, joinpath(save_dir, "O2_diff_lowpressure.png"))



#Halving and doubling albedo all 3 bands
println(true_x[22:27])
mult = ones(39)
mult[22:27] .= 0.5
half_albedo_x = true_x .* mult
println(half_albedo_x[22:27])
half_albedo_Y = F_matrix * half_albedo_x

strong_CO2_plt_half_albedo,weak_CO2_plt_half_albedo, O2_plt_half_albedo = plot_modified_radiances(wavelengths, Fx, half_albedo_Y, "1/2 Albedo")
strongCO2_diff_halfalbedo, weakCO2_diff_halfalbedo, O2_diff_halfalbedo = plot_radiance_differences(wavelengths, Fx, half_albedo_Y, "1/2 Albedo")
display(strong_CO2_plt_half_albedo)
display(strongCO2_diff_halfalbedo)
savefig(strong_CO2_plt_half_albedo, joinpath(save_dir, "StrongCO2Band_halfalbedo.png"))
savefig(strongCO2_diff_halfalbedo, joinpath(save_dir, "StrongCO2_diff_halfalbedo.png"))
display(weak_CO2_plt_half_albedo)
display(weakCO2_diff_halfalbedo)
savefig(weak_CO2_plt_half_albedo, joinpath(save_dir, "WeakCO2Band_half_albedo.png"))
savefig(weakCO2_diff_halfalbedo, joinpath(save_dir, "WeakCO2_diff_halfalbedo.png"))
display(O2_plt_half_albedo)
display(O2_diff_halfalbedo)
savefig(O2_plt_half_albedo, joinpath(save_dir, "O2Band_half_albedo.png"))
savefig(O2_diff_halfalbedo, joinpath(save_dir, "O2_diff_halfalbedo.png"))

mult[22:27] .= 2
double_albedo_x = true_x .* mult
double_albedo_Y = F_matrix * double_albedo_x

strong_CO2_plt_double_albedo,weak_CO2_plt_double_albedo, O2_plt_double_albedo = plot_modified_radiances(wavelengths, Fx, double_albedo_Y, "2x Albedo")
strongCO2_diff_doublealbedo, weakCO2_diff_doublealbedo, O2_diff_doublealbedo = plot_radiance_differences(wavelengths, Fx, double_albedo_Y, "2x Albedo")
display(strong_CO2_plt_double_albedo)
display(strongCO2_diff_doublealbedo)
savefig(strong_CO2_plt_double_albedo, joinpath(save_dir, "StrongCO2Band_doublealbedo.png"))
savefig(strongCO2_diff_doublealbedo, joinpath(save_dir, "StrongCO2_diff_doublealbedo.png"))
display(weak_CO2_plt_double_albedo)
display(weakCO2_diff_doublealbedo)
savefig(weak_CO2_plt_double_albedo, joinpath(save_dir, "WeakCO2Band_double_albedo.png"))
savefig(weakCO2_diff_doublealbedo, joinpath(save_dir, "WeakCO2_diff_doublealbedo.png"))
display(O2_plt_double_albedo)
display(O2_diff_doublealbedo)
savefig(O2_plt_double_albedo, joinpath(save_dir, "O2Band_double_albedo.png"))
savefig(O2_diff_doublealbedo, joinpath(save_dir, "O2_diff_doublealbedo.png"))


#Halving and doubling SO Aerosol Params
println(true_x[28:39])
mult = ones(39)
mult[28:30] .= 0.5
half_SOAerosol_x = true_x .* mult
half_SOAerosol_Y = F_matrix * half_SOAerosol_x

strong_CO2_plt_half_SOaerosol,weak_CO2_plt_half_SOaerosol, O2_plt_half_SOaerosol = plot_modified_radiances(wavelengths, Fx, half_SOAerosol_Y, "1/2 SO Aerosol Params")
strongCO2_diff_half_SOaerosol, weakCO2_diff_half_SOaerosol, O2_diff_half_SOaerosol = plot_radiance_differences(wavelengths, Fx, half_SOAerosol_Y, "1/2 SO Aerosol Params")
display(strong_CO2_plt_half_SOaerosol)
display(strongCO2_diff_half_SOaerosol)
savefig(strong_CO2_plt_half_SOaerosol, joinpath(save_dir, "StrongCO2Band_half_SOaerosol.png"))
savefig(strongCO2_diff_half_SOaerosol, joinpath(save_dir, "StrongCO2_diff_half_SOaerosol.png"))
display(weak_CO2_plt_half_SOaerosol)
display(weakCO2_diff_half_SOaerosol)
savefig(weak_CO2_plt_half_SOaerosol, joinpath(save_dir, "WeakCO2Band_half_SOaerosol.png"))
savefig(weakCO2_diff_half_SOaerosol, joinpath(save_dir, "WeakCO2_diff_half_SOaerosol.png"))
display(O2_plt_half_SOaerosol)
display(O2_diff_half_SOaerosol)
savefig(O2_plt_half_SOaerosol, joinpath(save_dir, "O2Band_half_SOaerosol.png"))
savefig(O2_diff_half_SOaerosol, joinpath(save_dir, "O2_diff_half_SOaerosol.png"))

mult[28:30] .= 2
double_SOAerosol_x = true_x .* mult
double_SOAerosol_Y = F_matrix * double_SOAerosol_x

strong_CO2_plt_double_SOaerosol,weak_CO2_plt_double_SOaerosol, O2_plt_double_SOaerosol = plot_modified_radiances(wavelengths, Fx, double_SOAerosol_Y, "2x SO Aerosol Params")
strongCO2_diff_double_SOaerosol, weakCO2_diff_double_SOaerosol, O2_diff_double_SOaerosol = plot_radiance_differences(wavelengths, Fx, double_SOAerosol_Y, "2x SO Aerosol Params")
display(strong_CO2_plt_double_SOaerosol)
display(strongCO2_diff_double_SOaerosol)
savefig(strong_CO2_plt_double_SOaerosol, joinpath(save_dir, "StrongCO2Band_double_SOaerosol.png"))
savefig(strongCO2_diff_double_SOaerosol, joinpath(save_dir, "StrongCO2_diff_double_SOaerosol.png"))
display(weak_CO2_plt_double_SOaerosol)
display(weakCO2_diff_double_SOaerosol)
savefig(weak_CO2_plt_double_SOaerosol, joinpath(save_dir, "WeakCO2Band_double_SOaerosol.png"))
savefig(weakCO2_diff_double_SOaerosol, joinpath(save_dir, "WeakCO2_diff_double_SOaerosol.png"))
display(O2_plt_double_SOaerosol)
display(O2_diff_double_SOaerosol)
savefig(O2_plt_double_SOaerosol, joinpath(save_dir, "O2Band_double_SOaerosol.png"))
savefig(O2_diff_double_SOaerosol, joinpath(save_dir, "O2_diff_double_SOaerosol.png"))

#Halving and doubling DU Aerosol Params
println(true_x[28:39])
mult = ones(39)
mult[31:33] .= 0.5
half_DUAerosol_x = true_x .* mult
half_DUAerosol_Y = F_matrix * half_DUAerosol_x

strong_CO2_plt_half_DUaerosol,weak_CO2_plt_half_DUaerosol, O2_plt_half_DUaerosol = plot_modified_radiances(wavelengths, Fx, half_DUAerosol_Y, "1/2 DU Aerosol Params")
strongCO2_diff_half_DUaerosol, weakCO2_diff_half_DUaerosol, O2_diff_half_DUaerosol = plot_radiance_differences(wavelengths, Fx, half_DUAerosol_Y, "1/2 DU Aerosol Params")
display(strong_CO2_plt_half_DUaerosol)
display(strongCO2_diff_half_DUaerosol)
savefig(strong_CO2_plt_half_DUaerosol, joinpath(save_dir, "StrongCO2Band_half_DUaerosol.png"))
savefig(strongCO2_diff_half_DUaerosol, joinpath(save_dir, "StrongCO2_diff_half_DUaerosol.png"))
display(weak_CO2_plt_half_DUaerosol)
display(weakCO2_diff_half_DUaerosol)
savefig(weak_CO2_plt_half_DUaerosol, joinpath(save_dir, "WeakCO2Band_half_DUaerosol.png"))
savefig(weakCO2_diff_half_DUaerosol, joinpath(save_dir, "WeakCO2_diff_half_DUaerosol.png"))
display(O2_plt_half_DUaerosol)
display(O2_diff_half_DUaerosol)
savefig(O2_plt_half_DUaerosol, joinpath(save_dir, "O2Band_half_DUaerosol.png"))
savefig(O2_diff_half_DUaerosol, joinpath(save_dir, "O2_diff_half_DUaerosol.png"))

mult[31:33] .= 2
double_DUAerosol_x = true_x .* mult
double_DUAerosol_Y = F_matrix * double_DUAerosol_x

strong_CO2_plt_double_DUaerosol,weak_CO2_plt_double_DUaerosol, O2_plt_double_DUaerosol = plot_modified_radiances(wavelengths, Fx, double_DUAerosol_Y, "2x DU Aerosol Params")
strongCO2_diff_double_DUaerosol, weakCO2_diff_double_DUaerosol, O2_diff_double_DUaerosol = plot_radiance_differences(wavelengths, Fx, double_DUAerosol_Y, "2x DU Aerosol Params")
display(strong_CO2_plt_double_DUaerosol)
display(strongCO2_diff_double_DUaerosol)
savefig(strong_CO2_plt_double_DUaerosol, joinpath(save_dir, "StrongCO2Band_double_DUaerosol.png"))
savefig(strongCO2_diff_double_DUaerosol, joinpath(save_dir, "StrongCO2_diff_double_DUaerosol.png"))
display(weak_CO2_plt_double_DUaerosol)
display(weakCO2_diff_double_DUaerosol)
savefig(weak_CO2_plt_double_DUaerosol, joinpath(save_dir, "WeakCO2Band_double_DUaerosol.png"))
savefig(weakCO2_diff_double_DUaerosol, joinpath(save_dir, "WeakCO2_diff_double_DUaerosol.png"))
display(O2_plt_double_DUaerosol)
display(O2_diff_double_DUaerosol)
savefig(O2_plt_double_DUaerosol, joinpath(save_dir, "O2Band_double_DUaerosol.png"))
savefig(O2_diff_double_DUaerosol, joinpath(save_dir, "O2_diff_double_DUaerosol.png"))




#Halving and doubling Ice Aerosol Params
println(true_x[28:39])
mult = ones(39)
mult[34:36] .= 0.5
half_IceAerosol_x = true_x .* mult
half_IceAerosol_Y = F_matrix * half_IceAerosol_x

strong_CO2_plt_half_IceAerosol,weak_CO2_plt_half_IceAerosol, O2_plt_half_IceAerosol = plot_modified_radiances(wavelengths, Fx, half_IceAerosol_Y, "1/2 Ice Aerosol Params")
strongCO2_diff_half_IceAerosol, weakCO2_diff_half_IceAerosol, O2_diff_half_IceAerosol = plot_radiance_differences(wavelengths, Fx, half_IceAerosol_Y, "1/2 Ice Aerosol Params")
display(strong_CO2_plt_half_IceAerosol)
display(strongCO2_diff_half_IceAerosol)
savefig(strong_CO2_plt_half_IceAerosol, joinpath(save_dir, "StrongCO2Band_half_IceAerosol.png"))
savefig(strongCO2_diff_half_IceAerosol, joinpath(save_dir, "StrongCO2_diff_half_IceAerosol.png"))
display(weak_CO2_plt_half_IceAerosol)
display(weakCO2_diff_half_IceAerosol)
savefig(weak_CO2_plt_half_IceAerosol, joinpath(save_dir, "WeakCO2Band_half_IceAerosol.png"))
savefig(weakCO2_diff_half_IceAerosol, joinpath(save_dir, "WeakCO2_diff_half_IceAerosol.png"))
display(O2_plt_half_IceAerosol)
display(O2_diff_half_IceAerosol)
savefig(O2_plt_half_IceAerosol, joinpath(save_dir, "O2Band_half_IceAerosol.png"))
savefig(O2_diff_half_IceAerosol, joinpath(save_dir, "O2_diff_half_IceAerosol.png"))

mult[34:36] .= 2
double_IceAerosol_x = true_x .* mult
double_IceAerosol_Y = F_matrix * double_IceAerosol_x

strong_CO2_plt_double_IceAerosol,weak_CO2_plt_double_IceAerosol, O2_plt_double_IceAerosol = plot_modified_radiances(wavelengths, Fx, double_IceAerosol_Y, "2x Ice Aerosol Params")
strongCO2_diff_double_IceAerosol, weakCO2_diff_double_IceAerosol, O2_diff_double_IceAerosol = plot_radiance_differences(wavelengths, Fx, double_IceAerosol_Y, "2x Ice Aerosol Params")
display(strong_CO2_plt_double_IceAerosol)
display(strongCO2_diff_double_IceAerosol)
savefig(strong_CO2_plt_double_IceAerosol, joinpath(save_dir, "StrongCO2Band_double_IceAerosol.png"))
savefig(strongCO2_diff_double_IceAerosol, joinpath(save_dir, "StrongCO2_diff_double_IceAerosol.png"))
display(weak_CO2_plt_double_IceAerosol)
display(weakCO2_diff_double_IceAerosol)
savefig(weak_CO2_plt_double_IceAerosol, joinpath(save_dir, "WeakCO2Band_double_IceAerosol.png"))
savefig(weakCO2_diff_double_IceAerosol, joinpath(save_dir, "WeakCO2_diff_double_IceAerosol.png"))
display(O2_plt_double_IceAerosol)
display(O2_diff_double_IceAerosol)
savefig(O2_plt_double_IceAerosol, joinpath(save_dir, "O2Band_double_IceAerosol.png"))
savefig(O2_diff_double_IceAerosol, joinpath(save_dir, "O2_diff_double_IceAerosol.png"))


#Halving and doubling Water Aerosol Params
println(true_x[28:39])
mult = ones(39)
mult[37:39] .= 0.5
half_WaterAerosol_x = true_x .* mult
half_WaterAerosol_Y = F_matrix * half_WaterAerosol_x

strong_CO2_plt_half_WaterAerosol,weak_CO2_plt_half_WaterAerosol, O2_plt_half_WaterAerosol = plot_modified_radiances(wavelengths, Fx, half_WaterAerosol_Y, "1/2 Water Aerosol Params")
strongCO2_diff_half_WaterAerosol, weakCO2_diff_half_WaterAerosol, O2_diff_half_WaterAerosol = plot_radiance_differences(wavelengths, Fx, half_WaterAerosol_Y, "1/2 Water Aerosol Params")
display(strong_CO2_plt_half_WaterAerosol)
display(strongCO2_diff_half_WaterAerosol)
savefig(strong_CO2_plt_half_WaterAerosol, joinpath(save_dir, "StrongCO2Band_half_WaterAerosol.png"))
savefig(strongCO2_diff_half_WaterAerosol, joinpath(save_dir, "StrongCO2_diff_half_WaterAerosol.png"))
display(weak_CO2_plt_half_WaterAerosol)
display(weakCO2_diff_half_WaterAerosol)
savefig(weak_CO2_plt_half_WaterAerosol, joinpath(save_dir, "WeakCO2Band_half_WaterAerosol.png"))
savefig(weakCO2_diff_half_WaterAerosol, joinpath(save_dir, "WeakCO2_diff_half_WaterAerosol.png"))
display(O2_plt_half_WaterAerosol)
display(O2_diff_half_WaterAerosol)
savefig(O2_plt_half_WaterAerosol, joinpath(save_dir, "O2Band_half_WaterAerosol.png"))
savefig(O2_diff_half_WaterAerosol, joinpath(save_dir, "O2_diff_half_WaterAerosol.png"))

mult[37:39] .= 2
double_WaterAerosol_x = true_x .* mult
double_WaterAerosol_Y = F_matrix * double_WaterAerosol_x

strong_CO2_plt_double_WaterAerosol,weak_CO2_plt_double_WaterAerosol, O2_plt_double_WaterAerosol = plot_modified_radiances(wavelengths, Fx, double_WaterAerosol_Y, "2x Water Aerosol Params")
strongCO2_diff_double_WaterAerosol, weakCO2_diff_double_WaterAerosol, O2_diff_double_WaterAerosol = plot_radiance_differences(wavelengths, Fx, double_WaterAerosol_Y, "2x Water Aerosol Params")
display(strong_CO2_plt_double_WaterAerosol)
display(strongCO2_diff_double_WaterAerosol)
savefig(strong_CO2_plt_double_WaterAerosol, joinpath(save_dir, "StrongCO2Band_double_WaterAerosol.png"))
savefig(strongCO2_diff_double_WaterAerosol, joinpath(save_dir, "StrongCO2_diff_double_WaterAerosol.png"))
display(weak_CO2_plt_double_WaterAerosol)
display(weakCO2_diff_double_WaterAerosol)
savefig(weak_CO2_plt_double_WaterAerosol, joinpath(save_dir, "WeakCO2Band_double_WaterAerosol.png"))
savefig(weakCO2_diff_double_WaterAerosol, joinpath(save_dir, "WeakCO2_diff_double_WaterAerosol.png"))
display(O2_plt_double_WaterAerosol)
display(O2_diff_double_WaterAerosol)
savefig(O2_plt_double_WaterAerosol, joinpath(save_dir, "O2Band_double_WaterAerosol.png"))
savefig(O2_diff_double_WaterAerosol, joinpath(save_dir, "O2_diff_double_WaterAerosol.png"))
