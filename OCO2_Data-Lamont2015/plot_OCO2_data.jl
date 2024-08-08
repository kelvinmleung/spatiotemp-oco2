using HDF5
using Random
using Plots
default()
using Statistics

plots_dir = "/Users/Camila/Desktop/OCO-2_UROP/spatiotemp-oco2/Plots/OCO2-Lamont2015"
sample_dir = "/Users/Camila/Desktop/OCO-2_UROP/spatiotemp-oco2/OCO2_Data-Lamont2015"

# Open the HDF5 file in read-only mode
OCO2file = h5open("/Users/Camila/Desktop/OCO-2_UROP/spatiotemp-oco2/OCO2_Data-Lamont2015/oco2_L2StdND_06750a_151008_B10206r_200719003702.h5", "r")

retrieval_header = OCO2file["RetrievalHeader"]
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

retrieval_results = OCO2file["RetrievalResults"]
co2_obj = (retrieval_results["co2_profile"])
co2_profiles = 10e5 .* read(co2_obj)

#Get CO2 profiles corresponding to sounding ids
sounding1 = co2_profiles[:,1598:1605]

sounding2 = co2_profiles[:,1606:1612]
f2 = sounding2[:,1]
sounding2 = hcat(f2, sounding2)

sounding3 = co2_profiles[:,1613:1618]
f1f4_mean = (co2_profiles[:,1613] + co2_profiles[:,1614])/2
sounding3 = hcat(co2_profiles[:,1613], f1f4_mean, f1f4_mean, sounding3[:,2:end])

sounding4 = co2_profiles[:,1619:1625]
f3f5_mean = (co2_profiles[:,1621] + co2_profiles[:,1622])/2
sounding4 = hcat(hcat(co2_profiles[:, 1619:1621], f3f5_mean),co2_profiles[:,1622:1625])

sounding5 = co2_profiles[:,1626:1632]
f5f7_mean = (co2_profiles[:,1630] + co2_profiles[:,1631])/2
sounding5 = hcat(hcat(co2_profiles[:,1626:1630], f5f7_mean),co2_profiles[:,1631:1632])

sounding6 = co2_profiles[:,1633:1639]
sounding6 = hcat(co2_profiles[:,1633:1639],co2_profiles[:,1639])

sounding7 = co2_profiles[:, 1640:1647]

sounding8 = co2_profiles[:, 1648:1655]

soundings = [sounding1,sounding2,sounding3, sounding4, sounding5, sounding6, sounding7, sounding8]

#Building tensor
sample_tensor = zeros(8,8,20)
for row in 1:8
    sample_tensor[row,:,:] = soundings[row]'
end

sample_tensor
# h5write(joinpath(sample_dir,"Lamont2015_CO2.h5"), "CO2_Profile", sample_tensor)

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
    size=(700,500),
    xlabel="Level",
    ylabel="X")
display(y_plt)
savefig(y_plt, joinpath(plots_dir, "Lamont2015_CO2_Y1.png"))


#Find Sample Variance
sample_mean = mean(co2_profiles', dims=1)
centered_data  = co2_profiles' .- sample_mean
sample_cov_matrix = cov(centered_data)
# h5write(joinpath(sample_dir,"SampleCov.h5"), "sample_cov_matrix", sample_cov_matrix)

#Compare XCO2s 
xco2_obj = retrieval_results["xco2"]
xco2s = 10e5 .* read(xco2_obj)

s1_xco2 = xco2s[1598:1605]
s2_xco2 = xco2s[1606:1612]
s3_xco2 = xco2s[1613:1618]
s4_xco2 = xco2s[1619:1625]
s5_xco2 = xco2s[1626:1632]
s6_xco2 = xco2s[1633:1639]
s7_xco2 = xco2s[1640:1647]
s8_xco2 = xco2s[1648:1655]

diff_cor_tensor = read(h5open("/Users/Camila/Desktop/OCO-2_UROP/spatiotemp-oco2/GRF/Lamont2015_CO2_corGRF.h5", "r")["CO2tensor-DiffCor"])
diff_cov_tensor = read(h5open("/Users/Camila/Desktop/OCO-2_UROP/spatiotemp-oco2/GRF/Lamont2015_CO2_covGRF.h5", "r")["CO2tensor-DiffCov"])

loc_cor_tensor = read(h5open("/Users/Camila/Desktop/OCO-2_UROP/spatiotemp-oco2/GRF/Lamont2015_CO2_corGRF.h5", "r")["CO2tensor-LocCor"])
loc_cov_tensor = read(h5open("/Users/Camila/Desktop/OCO-2_UROP/spatiotemp-oco2/GRF/Lamont2015_CO2_covGRF.h5", "r")["CO2tensor-LocCov"])

hs = read(retrieval_results["xco2_pressure_weighting_function"])
s1_h = hs[:,1598:1605]
s2_h = hcat(zeros(20),hs[:,1606:1612])
s3_h = hcat(hs[:,1613], zeros(20,2), hs[:,1614:1618])
s4_h = hcat(hs[:,1619:1621], zeros(20), hs[:,22:25])
s5_h = hcat(hs[:,1626:1630], zeros(20), hs[:,1631:1632])
s6_h = hcat(hs[:,1633:1639], zeros(20))
s7_h = hs[:,1640:1647]
s8_h = hs[:,1648:1655]

pressure_weighting_funcs = [s1_h, s2_h, s3_h, s4_h, s5_h, s6_h, s7_h, s8_h]

for row in 1:8
    h = pressure_weighting_funcs[row]'
    diff_cor_xco2s  = diff_cor_tensor[row,:,:]
    println(size(diff_cor_xco2s))
end 


#Surface Pressure Profile 
surf_press = read(retrieval_results["surface_pressure_fph"])

s1_sp =  surf_press[1598:1605]
s2_sp =  vcat(surf_press[1606],surf_press[1606:1612])
s3_sp =  vcat(surf_press[1613],fill((surf_press[1613] + surf_press[1614])/2,2), surf_press[1614:1618])
s4_sp =  vcat(surf_press[1619:1621],(surf_press[1621] + surf_press[1622])/2,surf_press[1622:1625])
s5_sp =  vcat(surf_press[1626:1630],(surf_press[1630] + surf_press[1631])/2, surf_press[1631:1632])
s6_sp =  vcat(surf_press[1633:1639], surf_press[1639])
s7_sp =  surf_press[1640:1647]
s8_sp =  surf_press[1648:1655]

Lamont2015_SP = 0.01 .* vcat(s1_sp', s2_sp', s3_sp', s4_sp', s5_sp', s6_sp', s7_sp', s8_sp')

sp_plt = heatmap(Lamont2015_SP, 
    colorbar_title="Surface Pressure (hPa)",
    clims=(975,990),
    title="Lamont2015 Surface Pressure", 
    size=(700,500))
display(sp_plt)
savefig(sp_plt, joinpath(plots_dir, "Lamont2015_SP.png"))


#Plot Aerosol results 
aerosol_results = OCO2file["AerosolResults"]
#DU, SS, BC, OC, SO, Ice, water, and ST
metadata = read(OCO2file["Metadata"]["AllAerosolTypes"])

read(aerosol_results["aerosol_param"])
OC_params = read(aerosol_results["aerosol_param"])[:,4,:]
SO_params = read(aerosol_results["aerosol_param"])[:,5,:]
ice_params = read(aerosol_results["aerosol_param"])[:,6,:]
water_params = read(aerosol_results["aerosol_param"])[:,7,:]


s1_oc = OC_params[:,1598:1605]
s2_oc =  hcat(OC_params[:,1606],OC_params[:,1606:1612])
footprint_mean = (OC_params[:,1613] .+ OC_params[:,1614])./2
s3_oc =  hcat(OC_params[:,1613],footprint_mean, footprint_mean, OC_params[:,1614:1618])
s4_oc =  hcat(OC_params[:,1619:1621],(OC_params[:,1621] .+ OC_params[:,1622])./2,OC_params[:,1622:1625])
s5_oc =  hcat(OC_params[:,1626:1630],(OC_params[:,1630] .+ OC_params[:,1631])./2, OC_params[:,1631:1632])
s6_oc =  hcat(OC_params[:,1633:1639], OC_params[:,1639])
s7_oc =  OC_params[:,1640:1647]
s8_oc =  OC_params[:,1648:1655]
OC_tensor = cat(s1_oc, s2_oc, s3_oc, s4_oc, s5_oc, s6_oc, s7_oc, s8_oc, dims=3)
OC = permutedims(OC_tensor, (3,2,1))
# h5write(joinpath(sample_dir,"Lamont2015_OC_Params.h5"), "ParamTensor", OC)


s1_so = SO_params[:,1598:1605]
s2_so =  hcat(SO_params[:,1606],SO_params[:,1606:1612])
footprint_mean = (SO_params[:,1613] .+ SO_params[:,1614])./2
s3_so =  hcat(SO_params[:,1613],footprint_mean, footprint_mean, SO_params[:,1614:1618])
s4_so =  hcat(SO_params[:,1619:1621],(SO_params[:,1621] .+ SO_params[:,1622])./2,SO_params[:,1622:1625])
s5_so =  hcat(SO_params[:,1626:1630],(SO_params[:,1630] .+ SO_params[:,1631])./2, SO_params[:,1631:1632])
s6_so =  hcat(SO_params[:,1633:1639], SO_params[:,1639])
s7_so =  SO_params[:,1640:1647]
s8_so =  SO_params[:,1648:1655]
SO_tensor = cat(s1_so, s2_so, s3_so, s4_so, s5_so, s6_so, s7_so, s8_so, dims=3)
SO = permutedims(SO_tensor, (3,2,1))
# h5write(joinpath(sample_dir,"Lamont2015_SO_Params.h5"), "ParamTensor", SO)


s1_ice = ice_params[:,1598:1605]
s2_ice =  hcat(ice_params[:,1606],ice_params[:,1606:1612])
footprint_mean = (ice_params[:,1613] .+ ice_params[:,1614])./2
s3_ice =  hcat(ice_params[:,1613],footprint_mean, footprint_mean, ice_params[:,1614:1618])
s4_ice =  hcat(ice_params[:,1619:1621],(ice_params[:,1621] .+ ice_params[:,1622])./2,ice_params[:,1622:1625])
s5_ice =  hcat(ice_params[:,1626:1630],(ice_params[:,1630] .+ ice_params[:,1631])./2, ice_params[:,1631:1632])
s6_ice =  hcat(ice_params[:,1633:1639], ice_params[:,1639])
s7_ice =  ice_params[:,1640:1647]
s8_ice =  ice_params[:,1648:1655]
ice_tensor = cat(s1_ice, s2_ice, s3_ice, s4_ice, s5_ice, s6_ice, s7_ice, s8_ice, dims=3)
ice = permutedims(ice_tensor, (3,2,1))
# h5write(joinpath(sample_dir,"Lamont2015_Ice_Params.h5"), "ParamTensor", ice)



s1_water = water_params[:,1598:1605]
s2_water =  hcat(water_params[:,1606],water_params[:,1606:1612])
footprint_mean = (water_params[:,1613] .+ water_params[:,1614])./2
s3_water =  hcat(water_params[:,1613],footprint_mean, footprint_mean, water_params[:,1614:1618])
s4_water =  hcat(water_params[:,1619:1621],(water_params[:,1621] .+ water_params[:,1622])./2,water_params[:,1622:1625])
s5_water =  hcat(water_params[:,1626:1630],(water_params[:,1630] .+ water_params[:,1631])./2, water_params[:,1631:1632])
s6_water =  hcat(water_params[:,1633:1639], water_params[:,1639])
s7_water =  water_params[:,1640:1647]
s8_water =  water_params[:,1648:1655]
water_tensor = cat(s1_water, s2_water, s3_water, s4_water, s5_water, s6_water, s7_water, s8_water, dims=3)
water = permutedims(water_tensor, (3,2,1))
# h5write(joinpath(sample_dir,"Lamont2015_water_Params.h5"), "ParamTensor", water)



for param in 1:3
    OC_sample = OC[:,:,param]
    SO_sample = SO[:,:,param]
    ice_sample = ice[:,:,param]
    water_sample = water[:,:,param]
    println("water sample $param \n $water_sample")

    OC_plt = heatmap(OC_sample, 
    title="Lamont2015 OC Aerosol Param $(param) Sample", 
    size=(700,500))
    display(OC_plt)
    savefig(OC_plt, joinpath(plots_dir, "Lamont2015_OC_Param$param.png"))

    SO_plt = heatmap(SO_sample, 
    title="Lamont2015 SO Aerosol Param $(param) Sample", 
    size=(700,500))
    display(SO_plt)
    savefig(SO_plt, joinpath(plots_dir, "Lamont2015_SO_Param$param.png"))

    ice_plt = heatmap(ice_sample, 
    title="Lamont2015 Ice Aerosol Param $(param) Sample", 
    size=(700,500))
    display(ice_plt)
    savefig(ice_plt, joinpath(plots_dir, "Lamont2015_Ice_Param$param.png"))

    water_plt = heatmap(water_sample, 
    title="Lamont2015 Water Aerosol Param $(param) Sample", 
    size=(700,500))
    display(water_plt)
    savefig(water_plt, joinpath(plots_dir, "Lamont2015_Water_Param$param.png"))

end