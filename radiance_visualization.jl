include("plot_helpers.jl")
using PyCall
np = pyimport("numpy")
using LinearAlgebra
using Distributions
using Serialization
using Accessors

save_dir = "/Users/Camila/Desktop/OCO-2_UROP/spatiotemp-oco2/Plots"

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

Fx = F_matrix*true_x


strong_CO2_plt_Fx,weak_CO2_plt_Fx,O2_plt_Fx = plot_radiances(wavelengths,Fx)
display(strong_CO2_plt_Fx)
savefig(strong_CO2_plt_Fx, joinpath(save_dir, "StrongCO2Band_Fx.png"))
display(weak_CO2_plt_Fx)
savefig(weak_CO2_plt_Fx, joinpath(save_dir, "WeakCO2Band_Fx.png"))
display(O2_plt)
savefig(O2_plt_Fx, joinpath(save_dir, "O2Band_Fx.png"))

# Generate the sample radiance
sample_Y = Fx + sample_error

strong_CO2_plt_Y,weak_CO2_plt_Y,O2_plt_Y = plot_radiances(wavelengths,sample_Y)
display(strong_CO2_plt_Y)
savefig(strong_CO2_plt_Y, joinpath(save_dir, "StrongCO2Band_Y.png"))
display(weak_CO2_plt_Y)
savefig(weak_CO2_plt_Y, joinpath(save_dir, "WeakCO2Band_Y.png"))
display(O2_plt_Y)
savefig(O2_plt_Y, joinpath(save_dir, "O2Band_Y.png"))

#decrease the conc of CO2 by 10
one = ones(20)
scaled = 100*one
diff = [scaled; zeros(19)]

low_CO2 = true_x - diff
lowCO2_Y = F_matrix * low_CO2

strong_CO2_plt_lowCO2,weak_CO2_plt_lowCO2, O2_plt_lowCO2 = plot_radiances(wavelengths, lowCO2_Y)
display(strong_CO2_plt_lowCO2)
savefig(strong_CO2_plt_lowCO2, joinpath(save_dir, "StrongCO2Band_lowCO2.png"))
display(weak_CO2_plt_lowCO2)
savefig(weak_CO2_plt_lowCO2, joinpath(save_dir, "WeakCO2Band_lowCO2.png"))
display(O2_plt_lowCO2)
savefig(O2_plt_lowCO2, joinpath(save_dir, "O2Band_lowCO2.png"))


#Change surface pressure
diff = zeros(39)
diff[21] = 100
high_surface_pressure_x = true_x + diff
println(high_surface_pressure_x[21])
high_surface_pressure_Y = F_matrix * high_surface_pressure_x

strong_CO2_plt_highpressure,weak_CO2_plt_highpressure, O2_plt_highpressure = plot_radiances(wavelengths, high_surface_pressure_Y)
display(strong_CO2_plt_highpressure)
savefig(strong_CO2_plt_highpressure, joinpath(save_dir, "StrongCO2Band_highpressure.png"))
display(weak_CO2_plt_highpressure)
savefig(weak_CO2_plt_highpressure, joinpath(save_dir, "WeakCO2Band_highpressure.png"))
display(O2_plt_highpressure)
savefig(O2_plt_highpressure, joinpath(save_dir, "O2Band_highpressure.png"))
