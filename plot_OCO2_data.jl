using HDF5
using Random
using Plots
pyplot()

plots_dir = "/Users/Camila/Desktop/OCO-2_UROP/spatiotemp-oco2/Plots/OCO2-Lamont2015"

# Open the HDF5 file in read-only mode
h5file = h5open("/Users/Camila/Desktop/OCO-2_UROP/spatiotemp-oco2/OCO2_Data-Lamont2015/oco2_L2StdND_06750a_151008_B10206r_200719003702.h5", "r")

retrieval_results = h5file["RetrievalResults"]

co2_obj = (retrieval_results["co2_profile"])
co2_profiles = 10e5 .* read(co2_obj)
cols = size(co2_profiles)[2]
co2_attrs = read(attributes(co2_obj)["Description"])

xco2_obj = retrieval_results["xco2"]
xco2s = 10e5 .* read(xco2_obj)
xco2_attr = read(attributes(xco2_obj)["Description"])


# Generate 64 unique random column indices
sample_idxs = Vector{Int}(undef, 64)
rand!(sample_idxs, 1:cols)
sample_idxs = reshape(sample_idxs,8,8)


sample_tensor = zeros(8,8,20)

for row in 1:8
    for col in 1:8
        idx = sample_idxs[row,col]
        # println("index = $idx")
        co2_profile = co2_profiles[:,idx]
        # println(size(co2_profile))
        sample_tensor[row,col,:] = co2_profile
    end
end

#Generate plots from tensor
for level in 1:20
    level_plt = heatmap(sample_tensor[:,:,level],
        colorbar=:bottom, 
        colorbar_title="CO2 Concentration (ppm)", 
        clims=(325,475),
        plot_title="Lamont2015 Level $(level) CO2 Sample", 
        size=(600,500))
    display(level_plt)
    savefig(level_plt, joinpath(plots_dir, "Lamont2015_CO2_Level$(level).png"))
end

x_plt = heatmap(sample_tensor[1,:,:],
colorbar=:bottom, 
    colorbar_title="CO2 Concentration (ppm)", 
    clims=(325,475),
    plot_title="Lamont2015 X=1 CO2 Sample", 
    size=(600,500))
display(x_plt)
savefig(x_plt, joinpath(plots_dir, "Lamont2015_CO2_X1.png"))

y_plt = heatmap(sample_tensor[:,1,:],
colorbar=:bottom, 
    colorbar_title="CO2 Concentration (ppm)", 
    clims=(325,475),
    plot_title="Lamont2015 Y=1 CO2 Sample", 
    size=(600,500))
display(y_plt)
savefig(y_plt, joinpath(plots_dir, "Lamont2015_CO2_Y1.png"))