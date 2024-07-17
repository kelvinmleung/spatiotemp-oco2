using HDF5
using Random
using Plots
default()

plots_dir = "/Users/Camila/Desktop/OCO-2_UROP/spatiotemp-oco2/Plots/OCO2-Lamont2015"

# Open the HDF5 file in read-only mode
h5file = h5open("/Users/Camila/Desktop/OCO-2_UROP/spatiotemp-oco2/OCO2_Data-Lamont2015/oco2_L2StdND_06750a_151008_B10206r_200719003702.h5", "r")

retrieval_header = h5file["RetrievalHeader"]
sounding_id_obj = retrieval_header["sounding_id"]
sounding_ids = read(sounding_id_obj)
sounding_ids[1598:1605]
sounding_ids[1606:1612]
sounding_ids[1613:1618]
sounding_ids[1619:1625]
sounding_ids[1626:1632]
sounding_ids[1633:1639]
sounding_ids[1640:1647]
sounding_ids[1648:1655]

retrieval_results = h5file["RetrievalResults"]
co2_obj = (retrieval_results["co2_profile"])
co2_profiles = 10e5 .* read(co2_obj)

#Get CO2 profiles corresponding to sounding ids
sounding1 = co2_profiles[:,1598:1605]

sounding2 = co2_profiles[:,1606:1612]
f2 = sounding2[:,1]
sounding2 = hcat(f2, sounding2)

sounding3 = co2_profiles[:,1613:1618]
f1f4_mean = (co2_profiles[:,1613] + co2_profiles[:,1614])/2
sounding3 = hcat(hcat(hcat(co2_profiles[:,1613], f1f4_mean), f1f4_mean), sounding3[:,2:end])

sounding4 = co2_profiles[:,1619:1625]
f3f5_mean = (co2_profiles[:,1621] + co2_profiles[:,1622])/2
sounding4 = hcat(hcat(co2_profiles[:, 1619:1621], f3f5_mean),co2_profiles[:,1622:1625])

sounding5 = co2_profiles[:,1626:1632]
f5f7_mean = (co2_profiles[:,1630] + co2_profiles[:,1631])/2
sounding5 = hcat(hcat(co2_profiles[:,1626:1630], f5f7_mean),co2_profiles[:,1631:1632])

sounding6 = co2_profiles[:,1633:1639]
sounding6 = hcat(sounding6,co2_profiles[:,1639])

sounding7 = co2_profiles[:, 1640:1647]

sounding8 = co2_profiles[:, 1648:1655]

soundings = [sounding1,sounding2,sounding3, sounding4, sounding5, sounding6, sounding7, sounding8]

#Building tensor
sample_tensor = zeros(8,8,20)
for row in 1:8
    sample_tensor[row,:,:] = soundings[row]'
end

sample_tensor

#Generate plots from tensor
for level in 1:20
    level_plt = heatmap(sample_tensor[:,:,level],
        colorbar_title="CO2 Concentration (ppm)", 
        clims=(325,475),
        title="Lamont2015 Level $(level) OCO2 CO2 Sample", 
        size=(700,500), 
        xlabel="X",
        ylabel="Y")
    display(level_plt)
    savefig(level_plt, joinpath(plots_dir, "Lamont2015_CO2_Level$(level).png"))
end

x_plt = heatmap(sample_tensor[1,:,:], 
    colorbar_title="CO2 Concentration (ppm)", 
    clims=(325,475),
    title="Lamont2015 X=1 CO2 Sample", 
    size=(700,500), 
    xlabel="Level",
    ylabel="Y")
display(x_plt)
savefig(x_plt, joinpath(plots_dir, "Lamont2015_CO2_X1.png"))

y_plt = heatmap(sample_tensor[:,1,:],
    colorbar_title="CO2 Concentration (ppm)", 
    clims=(325,475),
    title="Lamont2015 Y=1 CO2 Sample", 
    size=(700,500))
display(y_plt)
savefig(y_plt, joinpath(plots_dir, "Lamont2015_CO2_Y1.png"))