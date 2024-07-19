using PyCall
np = pyimport("numpy")
using LinearAlgebra
using Distributions
using Serialization
using GaussianRandomFields
using Plots
default()
using SpecialFunctions
using HDF5
using Printf

plots_dir = "/Users/Camila/Desktop/OCO-2_UROP/spatiotemp-oco2/Plots/CO2_GRF"
sample_dir = "/Users/Camila/Desktop/OCO-2_UROP/spatiotemp-oco2/GRF"


#Import known values
#Lamont 2015
numpy_true_x = np.load("/Users/Camila/Desktop/OCO-2_UROP/spatiotemp-oco2/SampleState-Lamont2015/true_state_vector_2015-10_lamont.npy")
Lamont2015_true_xCO2 = convert(Array{Float64}, numpy_true_x)[1:20]
Lamont2015_lambdas = [18.5, 15.5, 10, 27.5, 100,
    100,100,100,100,100,
    100,100,100,100, 2.5,
    2.5, 40, 100,100,100]
Lamont_lambda = mean(Lamont2015_lambdas)

# Wollongong 2016
numpy_true_x = np.load("/Users/Camila/Desktop/OCO-2_UROP/spatiotemp-oco2/SampleState-Wollongong2016/true_state_vector_Wollongong2016.npy")
Wollongong2016_true_xCO2 = convert(Array{Float64}, numpy_true_x)[1:20]
Wollongong2016_lambdas = [68.75, 31.25, 35, 18.5, 57.5,
        27.5,27.5,31.25,36,35,
        37,41,50,75, 87,
        22, 9.5, 100,72,53]
Wollongong2016_lambda = mean([68.75, 31.25, 35, 18.5, 57.5,
                27.5,27.5,31.25,36,35,
                37,41,50,75, 87,
                22, 9.5, 100,72,53])

#Wollongong 2017
numpy_true_x = np.load("/Users/Camila/Desktop/OCO-2_UROP/spatiotemp-oco2/SampleState-Wollongong2017/true_state_vector_Wollongong2017.npy")
Wollongong2017_true_xCO2 = convert(Array{Float64}, numpy_true_x)[1:20]
Wollongong2017_lambdas = [40.5, 20, 30, 42.5, 100,
                79,81, 68.75,59.5,37.5,
                28.5,20,12.5,17.5, 38,
                7.5, 12, 100,100,100]
Wollongong2017_lambda = mean([40.5, 20, 30, 42.5, 100,
                79,81, 68.75,59.5,37.5,
                28.5,20,12.5,17.5, 38,
                7.5, 12, 100,100,100])


labels = ["Lamont2015", "Wollongong2016", "Wollongong2017"]
true_CO2s = [Lamont2015_true_xCO2, Wollongong2016_true_xCO2, Wollongong2017_true_xCO2]
lambdas = [Lamont_lambda, Wollongong2016_lambda, Wollongong2017_lambda]


#Cov and Correlation matrix
numpy_cov = np.load("/Users/Camila/Desktop/OCO-2_UROP/spatiotemp-oco2/SampleState-Lamont2015/prior_cov_matrix_2015-10_lamont.npy")
prior_cov = convert(Array{Float64}, numpy_C)[1:20, 1:20]
prior_cov = tril(prior_cov) + tril(prior_cov)'
variances = diag(prior_cov)
D = sqrt.(diagm(variances))
prior_cor = inv(D)* prior_cov * inv(D)
prior_cor= tril(prior_cor) + tril(prior_cor,-1)'

#Sample covariance and correlation matrix 
sample_cov = OCO2file = read(h5open("/Users/Camila/Desktop/OCO-2_UROP/spatiotemp-oco2/OCO2_Data-Lamont2015/SampleCov.h5", "r")["sample_cov_matrix"])
sample_variances = diag(sample_cov)
sample_D = sqrt.(diagm(sample_variances))
sample_cor = inv(sample_D) * sample_cov * inv(sample_D)
sample_cor = tril(sample_cor) + tril(sample_cor, -1)'

#Plot difference in covariance matrices
difference_Cs = prior_cov - sample_cov
cov_plt = heatmap(difference_Cs, 
            title="Lamont2015 Difference Between Prior and Sample Covariance Matrices",  
            titlefontsize=10)
savefig(cov_plt, joinpath("/Users/Camila/Desktop/OCO-2_UROP/spatiotemp-oco2/Plots", "Lamont2015_Sample-Prior_Cov_Difference.png"))



#Constants
diffusion_coef = 16.0 / (10^6)  
diffusion_lambda = sqrt(diffusion_coef)  # km
fixed_nu = 1.0
x_pts = range(0.65,step=1.3, length=8)
y_pts = range(1.15, step=2.3, length=8)
z_pts = 1:20



function matern_kernel(p1,p2,lambda, nu; sigma=1.0)
    d = abs(p1-p2)
    if iszero(d)
        return 1.0
    end 
    scale = sigma^2 * 2^(1 - nu) / gamma(nu)
    factor = (sqrt(2 * nu) * d / lambda)
    result = scale * factor^nu * besselk(nu, factor)
    # @printf("x1: %.4f, x2: %.4f, d: %.4f, factor: %.4f, besselk: %.4e, result: %.4e\n", p1, p2, d, factor, besselk(nu, factor), result)
    return result
end

function cov_matrix_entry(z1,z2,C)
    @assert C == C' "C is not symmetric"
    i = Int(z1)
    j = Int(z2)
    return C[i,j]
end



function construct_cov_matrix(x_pts, y_pts, z_pts, lambdas, nu, C)
    n = length(x_pts)
    m = length(y_pts)
    k = length(z_pts)
    
    K = zeros(n*m*k, n*m*k)

    row = 1
    col = 1
    for x1 in 1:n
        for y1 in 1:m
            for z1 in 1:k
                for x2 in 1:n
                    for y2 in 1:m 
                        for z2 in 1:k
                            if col == 1281
                                col = 1
                                row +=1
                            end 
                            
                            lambda_k = lambdas[z1]
                            lambda_l = lambdas[z2]
                            lambda_kl = (lambdas[z1] + lambdas[z2])/2
                            
                            cov_x = matern_kernel(x_pts[x1], x_pts[x2], lambda_kl, nu)
                            cov_y = matern_kernel(y_pts[y1], y_pts[y2], lambda_kl, nu)
                            cov_z = cov_matrix_entry(z_pts[z1], z_pts[z2], C)
                            factor = sqrt(lambda_k)*sqrt(lambda_l)/lambda_kl
                            prod = cov_x*cov_y*cov_z*factor
                            K[row,col] = prod
                            # if row == 21
                            #     println("X:$(x1), $(x2), Y: $(y1), $(y2), Z: $(z1), $(z2)")
                            #     println("lambda k = $lambda_k")
                            #     println("lambda l = $lambda_l")
                            #     println("factor = $factor")
                            #     println("cov_x: $(cov_x), cov_y: $(cov_y), cov_z: $(cov_z), prod: $(prod)")
                            # end

                            col +=1
                        end
                    end
                end 
            end
        end
    end
    K = tril(K) + tril(K,-1)'
    K = K + 1e-9 .* I
    return K
 
end


##Attempt to construct matrix with new formula
test_cov = construct_cov_matrix(x_pts,y_pts,z_pts, fill(diffusion_lambda, 20), fixed_nu, sample_cov)
cond(test_cov)
eig_K = eigvals(test_cov)



#For each location
for loc in 1:3
    #Make mean vector
    true_x = true_CO2s[loc]
    mean = zeros(1280)

    for i in 1:1280
        idx = i % 20
        if iszero(idx)
            mean[i] = true_x[20]
        else
            mean[i] = true_x[idx]
        end
    end

    #Make covariance matrix 
    mean_lambda_vec = fill(lambdas[loc], 20)
    diffusion_lambda_vec = fill(diffusion_lambda, 20)
    loc_cov_K = construct_cov_matrix(x_pts,y_pts,z_pts,mean_lambda_vec, fixed_nu, sample_cov)
    loc_cor_K = construct_cov_matrix(x_pts,y_pts,z_pts,mean_lambda_vec, fixed_nu, sample_cor)
    
    diffusion_cov_K = construct_cov_matrix(x_pts,y_pts,z_pts,diffusion_lambda_vec, fixed_nu, sample_cov)
    diffusion_cor_K = construct_cov_matrix(x_pts, y_pts, z_pts, diffusion_lambda_vec, fixed_nu, sample_cor)

    loc_cov_grf = MvNormal(mean, loc_cov_K)
    loc_cov_sample_vec = rand(loc_cov_grf)

    loc_cor_grf = MvNormal(mean,loc_cor_K)
    loc_cor_sample_vec = rand(loc_cor_grf)

    diff_cov_grf = MvNormal(mean,diffusion_cov_K)
    diff_cov_sample_vec = rand(diff_cov_grf)

    diff_cor_grf = MvNormal(mean,diffusion_cor_K)
    diff_cor_sample_vec = rand(diff_cor_grf)

    #Construct tensor from sample & save it
    n = length(x_pts)
    m = length(y_pts)
    k = length(z_pts)
    sample_loc_cov_tensor = zeros(n,m,k)
    sample_loc_cor_tensor = zeros(n,m,k)
    sample_diff_cov_tensor = zeros(n,m,k)
    sample_diff_cor_tensor = zeros(n,m,k)

    j = 1
    for x in 1:n
        for y in 1:m
            for z in 1:k
                sample_loc_cov_tensor[x,y,z] = loc_cov_sample_vec[j]
                sample_loc_cor_tensor[x,y,z] = loc_cor_sample_vec[j]
                sample_diff_cov_tensor[x,y,z] = diff_cov_sample_vec[j]
                sample_diff_cor_tensor[x,y,z] = diff_cor_sample_vec[j]
                j+=1
            end
        end
    end

    # h5write(joinpath(sample_dir,"$(labels[loc])_CO2_covGRF.h5"), "CO2tensor-LocCov", sample_loc_tensor)
    # h5write(joinpath(sample_dir,"$(labels[loc])_CO2_covGRF.h5"), "CO2tensor-DiffCov", sample_diff_tensor)

    # h5write(joinpath(sample_dir,"$(labels[loc])_CO2_corGRF.h5"), "CO2tensor-LocCor", sample_loc_tensor)
    # h5write(joinpath(sample_dir,"$(labels[loc])_CO2_corGRF.h5"), "CO2tensor-DiffCor", sample_diff_tensor)

    #Generate plots from tensor
    for level in 1:k
        loc_cov_level_plt = heatmap(sample_loc_cov_tensor[:,:,level],
            clims=(325,475), 
            title="$(labels[loc]) Level $(level) Location λ Sample Covariance GRF", 
            xlabel="X", 
            ylabel="Y",
            colorbar_title="CO2 Concentration (ppm)", 
            size=(700,500), 
            titlefontsize=12)
        display(loc_cov_level_plt)
        savefig(loc_cov_level_plt, joinpath(plots_dir, "$(labels[loc])_CO2_GRF_Level$(level)_LocSampleCov.png"))

        loc_cor_level_plt = heatmap(sample_loc_cor_tensor[:,:,level],
            clims=(325,475), 
            title="$(labels[loc]) Level $(level) Location λ Sample Correlation GRF", 
            xlabel="X", 
            ylabel="Y",
            colorbar_title="CO2 Concentration (ppm)", 
            size=(700,500), 
            titlefontsize=12)
        display(loc_cor_level_plt)
        savefig(loc_cor_level_plt, joinpath(plots_dir, "$(labels[loc])_CO2_GRF_Level$(level)_LocSampleCor.png"))


        diffusion_cov_level_plt = heatmap(sample_diff_cov_tensor[:,:,level],
            clims=(325,475), 
            title="$(labels[loc]) Level $(level) Diffusion λ Sample Covariance GRF", 
            xlabel="X", 
            ylabel="Y", 
            colorbar_title="CO2 Concentration (ppm)", 
            size=(700,500), 
            titlefontsize=12)
        display(diffusion_cov_level_plt)
        savefig(diffusion_cov_level_plt, joinpath(plots_dir, "$(labels[loc])_CO2_GRF_Level$(level)_DiffSampleCov.png"))

        diffusion_cor_level_plt = heatmap(sample_diff_cor_tensor[:,:,level],
            clims=(325,475), 
            title="$(labels[loc]) Level $(level) Diffusion λ Sample Correlation GRF", 
            xlabel="X", 
            ylabel="Y", 
            colorbar_title="CO2 Concentration (ppm)", 
            size=(700,500), 
            titlefontsize=12)
        display(diffusion_cor_level_plt)
        savefig(diffusion_cor_level_plt, joinpath(plots_dir, "$(labels[loc])_CO2_GRF_Level$(level)_DiffSampleCor.png"))


        # difference_level_plt = heatmap(difference_tensor[:,:,level], title="$(labels[loc]) Level $(level) Difference b/w Mean Location and Diffusion Range GRF Sample", xlabel="X", ylabel="Y", colorbar_title="CO2 Concentration (ppm)", size=(700,400),titlefontsize=10)
        # display(difference_level_plt)
        # savefig(difference_level_plt, joinpath(plots_dir, "$(labels[loc])_CO2_GRF_Level$(level)_DifferenceCor.png"))
    end

    #Plot X=1 slice for location, diffusion and difference
    loc_cov_x_plt = heatmap(sample_loc_cov_tensor[1,:,:], 
        clims=(325,475), 
        title="$(labels[loc]) Location λ Sample Covariance GRF X = 1", 
        xlabel="Level", 
        ylabel="Y", 
        colorbar_title="CO2 Concentration (ppm)", 
        size=(700,500), 
        titlefontsize=12)
    display(loc_cov_x_plt)
    savefig(loc_cov_x_plt, joinpath(plots_dir, "$(labels[loc])_CO2_GRF_X1_LocSampleCov.png"))

    loc_cor_x_plt = heatmap(sample_loc_cor_tensor[1,:,:], 
        clims=(325,475), 
        title="$(labels[loc]) Location λ Sample Correlation GRF X = 1", 
        xlabel="Level", 
        ylabel="Y", 
        colorbar_title="CO2 Concentration (ppm)", 
        size=(700,500))
    display(loc_cor_x_plt)
    savefig(loc_cor_x_plt, joinpath(plots_dir, "$(labels[loc])_CO2_GRF_X1_LocSampleCor.png"))

    diffusion_cov_x_plt = heatmap(sample_diff_cov_tensor[1,:,:], 
        clims=(325,475), 
        title="$(labels[loc]) Diffusion λ Covariance Sample GRF X = 1", 
        xlabel="Level", 
        ylabel="Y", 
        colorbar_title="CO2 Concentration (ppm)", 
        size=(700,500))
    display(diffusion_cov_x_plt)
    savefig(diffusion_cov_x_plt, joinpath(plots_dir, "$(labels[loc])_CO2_GRF_X1_DiffSampleCov.png"))

    diffusion_cor_x_plt = heatmap(sample_diff_cor_tensor[1,:,:], 
        clims=(325,475), 
        title="$(labels[loc]) Diffusion λ Correlation Sample GRF X = 1", 
        xlabel="Level", 
        ylabel="Y", 
        colorbar_title="CO2 Concentration (ppm)", 
        size=(700,500))
    display(diffusion_cor_x_plt)
    savefig(diffusion_cor_x_plt, joinpath(plots_dir, "$(labels[loc])_CO2_GRF_X1_DiffSampleCor.png"))

    
    
    # difference_x_cov_plt = heatmap(difference_cov_tensor[1,:,:], 
    # title="$(labels[loc]) Difference b/w Location and Diffusion λ X = 1", 
    # xlabel="Level", 
    # ylabel="Y", 
    # colorbar_title="CO2 Concentration (ppm)", 
    # size=(700,500))
    # display(difference_x_cov_plt)
    # savefig(difference_x_cov_plt, joinpath(plots_dir, "$(labels[loc])_CO2_GRF_X1_DifferenceCov.png"))

    # difference_x_cor_plt = heatmap(difference_cor_tensor[1,:,:], 
    # title="$(labels[loc]) Difference b/w Location and Diffusion λ X = 1", 
    # xlabel="Level", 
    # ylabel="Y", 
    # colorbar_title="CO2 Concentration (ppm)", 
    # size=(700,500))
    # display(difference_x_cor_plt)
    # savefig(difference_x_cor_plt, joinpath(plots_dir, "$(labels[loc])_CO2_GRF_X1_DifferenceCorgt.png"))

# #Plot Y=1 slice for location, diffusion and difference
#     loc_y_plt = heatmap(sample_loc_tensor[:,1,:], clims=(325,475), title="$(labels[loc]) Mean Range Normalized GRF Sample Y = 1", xlabel="Level", ylabel="X", colorbar_title="CO2 Concentration (ppm)", size=(700,400),titlefontsize=12)
#     display(loc_y_plt)
#     savefig(loc_y_plt, joinpath(plots_dir, "$(labels[loc])_CO2_GRF_Y1_LocCor.png"))

#     diffusion_y_plt = heatmap(sample_diff_tensor[:,1,:], clims=(325,475), title="$(labels[loc]) Diffusion Range GRF Sample Y = 1", xlabel="Level", ylabel="X", colorbar_title="CO2 Concentration (ppm)",size=(700,400),titlefontsize=12)
#     display(diffusion_y_plt)
#     savefig(diffusion_y_plt, joinpath(plots_dir, "$(labels[loc])_CO2_GRF_Y1_DiffCor.png"))
    
#     difference_y_plt = heatmap(difference_tensor[:,1,:], title="$(labels[loc]) Difference b/w Mean Location and Diffusion Range Y = 1", xlabel="Level", ylabel="X", colorbar_title="CO2 Concentration (ppm)",size=(700,400),titlefontsize=10)
#     display(difference_y_plt)
#     savefig(difference_y_plt, joinpath(plots_dir, "$(labels[loc])_CO2_GRF_Y1_DifferenceCor.png"))

end

