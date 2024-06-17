function plot_CO2_by_level(true_CO2, MAP_CO2, posterior_CO2_stds, est_CO2, est_CO2_stds, bar_width=0.3)
    number_elts = length(true_CO2)
    true_stds = zeros(n)
    x = 1:number_elts
    plot = bar(x .- bar_width, true_CO2, bar_width=bar_width, label="True CO2", color=:lightblue, legend=:bottomright, yerr=true_stds, size=(1200,600))
    bar!(x, MAP_CO2, bar_width=bar_width, label="MAP CO2", color=:mediumpurple, yerr=posterior_CO2_stds)
    bar!(x .+ bar_width, est_CO2, bar_width=bar_width, label="Estimate CO2", color=:palevioletred, yerr=est_CO2_stds)
    xlabel!("Level")
    ylabel!("CO2 [ppm]")
    ylims!(360,420)
    return plot
end 

function plot_vertical_profile(estimate_CO2, MAP_CO2=zeros(20), true_CO2=zeros(20))
    x = 1:20
    plt = plot(x, estimate_CO2, 
     seriestype=:line,  
     label="Estimate", 
     xlabel="CO2 [ppm]", 
     ylabel="Vertical Level",  
     linecolor=:blue, 
     linewidth=2, 
     legend=:bottomright)

    # Check if MAP_CO2 is not all zeros and plot if it's not
    if any(MAP_CO2 .!= 0)
        plot!(plt, x, MAP_CO2, 
              seriestype=:line,  
              label="MAP", 
              linecolor=:red, 
              linewidth=2)
    end

    # Check if true_CO2 is not all zeros and plot if it's not
    if any(true_CO2 .!= 0)
        plot!(plt, x,true_CO2, 
              seriestype=:line,
              label="True", 
              linecolor=:green, 
              linewidth=2)
    end
    return plt
    
end