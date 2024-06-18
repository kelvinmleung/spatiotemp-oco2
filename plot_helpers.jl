function plot_CO2_by_level(true_x, MAP, posterior_cov, est_MAP, est_cov, bar_width=0.3)
    true_CO2 = true_x[1:20]
    true_stds = zeros(20)

    MAP_CO2 = MAP[1:20]
    posterior_CO2_stds = sqrt.(diag(posterior_cov)[1:20])

    est_CO2 = est_MAP[1:20]
    est_CO2_stds = sqrt.(diag(est_cov)[1:20])

    true_stds = zeros(n)
    x = 1:20
    plot = bar(x .- bar_width, true_CO2, bar_width=bar_width, label="True CO2", color=:lightblue, legend=:bottomright, yerr=true_stds, size=(1200,600))
    bar!(x, MAP_CO2, bar_width=bar_width, label="MAP CO2", color=:mediumpurple, yerr=posterior_CO2_stds)
    bar!(x .+ bar_width, est_CO2, bar_width=bar_width, label="Estimate CO2", color=:palevioletred, yerr=est_CO2_stds)
    xlabel!("Level")
    ylabel!("CO2 [ppm]")
    ylims!(375,425)
    return plot
end 

function plot_vertical_profile(estimate_CO2,prior_mean, MAP_CO2=zeros(20), true_CO2=zeros(20))
    x = 1:20
    plt = plot(x, estimate_CO2[1:20], 
     seriestype=:line,  
     label="Estimate", 
     xlabel="Vertical Level", 
     ylabel="CO2 [ppm]",  
     linecolor=:blue, 
     linewidth=2, 
     legend=:bottomright)

     plot!(x,prior_mean[1:20], 
        seriestype=:line,
        linestyle=:dot,
        label="Prior",
        linecolor=:black,
        linewidth=2)

     ylims!(plt, 390,410)

    # Check if MAP_CO2 is not all zeros and plot if it's not
    if any(MAP_CO2 .!= 0)
        plot!(plt, x, MAP_CO2[1:20], 
              seriestype=:line,  
              label="MAP", 
              linecolor=:red, 
              linewidth=2)
    end

    # Check if true_CO2 is not all zeros and plot if it's not
    if any(true_CO2 .!= 0)
        plot!(plt, x,true_CO2[1:20], 
              seriestype=:line,
              label="True", 
              linecolor=:green, 
              linewidth=2)
    end
    return plt
    
end

function plot_col_avg_co2(h_matrix, MAP_est, posterior_cov_est, MAP=zeros(20), true_post_cov=zeros(20,20), true_CO2=zeros(20))
    est_col_avg = h_matrix'*MAP_est[1:20]
    post_est_std = sqrt(h_matrix'*posterior_cov_est[1:20,1:20]*h_matrix)
    labels = ["MAP Estimate Column Avg CO2"]
    values = [est_col_avg]
    std_devs = [post_est_std]

    if any(MAP[1:20] .!= 0)
        map_col_avg = h_matrix'*MAP[1:20]
        post_std = sqrt(h_matrix'*true_post_cov[1:20,1:20]*h_matrix)
        push!(labels, "MAP Column Avg CO2")
        push!(values, map_col_avg)
        push!(std_devs, post_std)
    end 

    if any(true_CO2[1:20] .!= 0)
        true_col_avg = h_matrix'*true_CO2[1:20]
        push!(labels, "True Column Avg CO2")
        push!(values, true_col_avg)
        push!(std_devs, 0)
    end 
    ColAvgCO2_plot = bar(labels, values, yerr=std_devs, bar_width=0.5, title="Column Average CO2 Concentration Comparison", ylabel="CO2 [ppm]", legend=false)
    ylims!(370,420)
    return ColAvgCO2_plot
end


function plot_radiances(wavelengths, radiances)
    #Plot the Strong CO2 band
    strong_CO2_plt = plot(wavelengths[1:1016], radiances[1:1016],
    seriestype=:line,
    xlabel="Wavelength",
    ylabel="Radiance",
    linecolor=:red,
    linewidth=1,
    label="Strong CO2 Band Radiance",
    size=(1200,400))

    #Plot the Weak CO2 band
    weak_CO2_plt = plot(wavelengths[1017:2032], radiances[1017:2032],
    seriestype=:line,
    xlabel="Wavelength",
    ylabel="Radiance",
    linecolor=:green,
    linewidth=1,
    label="Weak CO2 Band Radiance",
    size=(1200,400))

    #Plot the O2 band
    O2_plt = plot(wavelengths[2033:3048], radiances[2033:3048],
    seriestype=:line,
    xlabel="Wavelength",
    ylabel="Radiance",
    linecolor=:blue,
    linewidth=1,
    label="O2 Band Radiance",
    size=(1200,400))

    return strong_CO2_plt,weak_CO2_plt,O2_plt
end

function plot_modified_radiances(wavelengths, true_radiances, modified_radiences, label)
    #Plot the Strong CO2 band
    strong_CO2_plt = plot(wavelengths[1:1016], true_radiances[1:1016],
    seriestype=:line,
    xlabel="Wavelength",
    ylabel="Radiance",
    title="Strong CO2 Band",
    linecolor=:black,
    linewidth=1,
    label="Fx radiance",
    legend=:topright,
    size=(1200,400))

    plot!(wavelengths[1:1016], modified_radiences[1:1016],
    linecolor=:lightcoral,
    linewidth=1,
    label="$(label) radiance",)

    #Plot the Weak CO2 band
    weak_CO2_plt = plot(wavelengths[1017:2032], true_radiances[1017:2032],
    seriestype=:line,
    xlabel="Wavelength",
    ylabel="Radiance",
    title="Weak CO2 Band",
    linecolor=:black,
    linewidth=1,
    label="Fx Radiance",
    legend=:bottomleft,
    size=(1200,400))

    plot!(wavelengths[1017:2032], modified_radiences[1017:2032],
    linecolor=:lightgreen,
    linewidth=1,
    label="$(label) radiance")

    #Plot the O2 band
    O2_plt = plot(wavelengths[2033:3048], true_radiances[2033:3048],
    seriestype=:line,
    xlabel="Wavelength",
    ylabel="Radiance",
    title="O2 Band",
    linecolor=:black,
    linewidth=1,
    label="Fx Radiance",
    legend=:topleft,
    size=(1200,400))

    plot!(wavelengths[2033:3048], modified_radiences[2033:3048],
    linecolor=:steelblue1,
    linewidth=1,
    label="$(label) radiance")

    return strong_CO2_plt,weak_CO2_plt,O2_plt
end

function plot_radiance_differences(wavelengths, true_rads, modified_rads, label)
    diff = modified_rads .- true_rads 

    strong_ymin = minimum(diff[1:1016])
    strong_ymax = maximum(diff[1:1016])
    strongCO2_diff_plt = plot(wavelengths[1:1016],
    diff[1:1016],
    seriestype=:line,
    xlabel="Wavelength",
    ylabel="Δ Radiance",
    title="Strong CO2 Band Differences ",
    linecolor=:red2,
    linewidth=1,
    label=label,
    ylims=(strong_ymin,strong_ymax),
    size=(1200,400))

    weak_ymin = minimum(diff[1017:2032])
    weak_ymax = maximum(diff[1017:2032])
    weakCO2_diff_plt = plot(wavelengths[1017:2032],
    diff[1017:2032],
    seriestype=:line,
    xlabel="Wavelength",
    ylabel="Δ Radiance",
    title="Weak CO2 Band Differences ",
    linecolor=:green4,
    linewidth=1,
    label=label,
    ylims=(weak_ymin, weak_ymax),
    size=(1200,400))

    O2_ymin = minimum(diff[2033:3048])
    O2_ymax = maximum(diff[2033:3048])
    O2_diff_plt = plot(wavelengths[2033:3048],
    diff[2033:3048],
    seriestype=:line,
    xlabel="Wavelength",
    ylabel="Δ Radiance",
    title="O2 Band Differences ",
    linecolor=:mediumblue,
    linewidth=1,
    label=label,
    ylims=(O2_ymin, O2_ymax),
    size=(1200,400))

    return strongCO2_diff_plt, weakCO2_diff_plt, O2_diff_plt
end