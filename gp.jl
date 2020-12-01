
using CUDA
using Distributions
using KernelFunctions
using LinearAlgebra
using ProgressMeter
using Random
using DelimitedFiles
#using Plots

include("util.jl")

function ess(prng, θ0, X, y, k, nsamples, nburn,
             prior_logℓ, prior_logσ, prior_logϵ)
    logpθ = Product([prior_logℓ..., prior_logσ, prior_logϵ])
    μ     = mean(logpθ)

    function calc_mll!(θ_in)
        logℓ = θ_in[1:length(prior_logℓ)]
        logσ = θ_in[length(prior_logℓ)+1]
        logϵ = θ_in[length(prior_logℓ)+2]
        #logℓ = θ_in[1]
        #logσ = θ_in[2]
        #logϵ = θ_in[3]

        α, mll = train_gp(X, y, k, logℓ, logσ, logϵ)
        α, mll
    end

    function sample!(f::AbstractVector, L)
        ν      = rand(prng, logpθ)
        u      = rand(prng)
        logy   = L + log(u)
        θ      = rand(prng)*2*π
        θ_min  = θ - 2*π
        θ_max  = θ
        f_prime = (f - μ)*cos(θ) + (ν - μ)*sin(θ) + μ
        props = 1
        while true
            α, L = calc_mll!(f_prime)
            if(L > logy)
                break
            end
            props += 1
            if θ < 0
                θ_min = θ
            else
                θ_max = θ
            end
            θ = rand(prng) * (θ_max - θ_min) + θ_min
            f_prime = (f - μ)*cos(θ) + (ν - μ)*sin(θ) + μ
        end
        f_prime, props, L, α
    end

    θ_cur  = θ0
    dim    = length(θ_cur)
    θ_post = Array{Float64}(undef, dim,       nsamples)
    α_post = Array{Float64}(undef, length(y), nsamples)

    α, mll_cur = calc_mll!(θ_cur)

    prog = ProgressMeter.Progress(nsamples+nburn)
    for i = 1:nsamples + nburn
        θ_cur, num_props, mll_cur, α_cur = sample!(θ_cur, mll_cur)
        if(i > nburn)
            θ_post[:, i - nburn] = θ_cur
            α_post[:, i - nburn] = α_cur
        end
        ProgressMeter.next!(prog;
                            showvalues = [(:iter, i),
                                          (:mll,  mll_cur),
                                          (:acc,  1 / num_props)])
    end
    θ_post, α_post
end

function train_gp(X, y, k, logℓ, logσ, logϵ)
    ℓinv = exp.(-logℓ) 
    σ²  = exp(logσ*2) 
    ϵ²  = exp(logϵ*2) 

    #ard  = ARDTransform(ℓinv)
    ard  = ScaleTransform(ℓinv)
    kard = σ² * TransformedKernel(k, ard)

    K  = kernelmatrix(kard, X, obsdim=2)
    K += ϵ²*I

    try
        cholesky!(K)
        U   = UpperTriangular(K)
        α   = U' \ (U \ y)
        t1  = dot(y, α) / -2
        t2  = -sum(log.(diag(U)))
        t3  = -size(K,1) / 2 * log(2*π)
        mll = t1+t2+t3
        α, mll 
    catch e
        println("chol failed!")
        zeros(size(K, 1)), -Inf
    end
end

function predict_gp(α::Matrix, θ::Matrix, X_data::Matrix, k, X_in::Matrix)
    μs = @showprogress map(1:size(α,2)) do i 
        logℓ = θ[1:size(X_data, 1), i]
        logσ = θ[size(X_data, 1)+1, i]
        logϵ = θ[size(X_data, 1)+2, i]

        ℓinv = exp.(-logℓ) 
        σ²   = exp(logσ*2) 
        ϵ²   = exp(logϵ*2) 
        ard  = ARDTransform(ℓinv)
        kard = σ² * TransformedKernel(k, ard)
        kstar = kernelmatrix(kard, X_data, X_in, obsdim=2)
        kstar' * α[:,i]
    end
    μ = mean(μs)
end

using GaussianProcesses

function train_gp(prng, X_train, y_train, X_test, y_test)
    # 114 < longitude           < 124
    # 32  < latitude            < 42 
    # 0   < housing_median_age  < 1000
    # 1   < total_rooms         < 100
    # 1   < total_bedrooms      < 100
    # 10  < population          < 40e+6
    # 0   < households          < 10e+6
    # 0   < median_income (10k) < 100
    # 1   < ocean_proximity     < 4

    # X_train = X_train[:,1:100]
    # y_train = y_train[1:100]

    # k  = Mat52Ard(10.0*ones(9), 100.0)
    # set_priors!(k, [Normal(0.0, 10000.0) for i = 1:10])
    # gp = GP(X_train,y_train, MeanZero(), k, 100.0)       #Fit the GP
    # set_priors!(gp.logNoise, [Normal(0.0,100.0)])

    # θ = optimize!(gp)
    # display(θ)
    # println(GaussianProcesses.get_params(gp))
    # #display(plot(θ'))
    # #display(Chains(reshape(θ', (:,11,1))))
    # throw()

    k          = Matern52Kernel()
    nsamples   = 200
    nburn      = 100
    dims       = size(X_train,1)
    prior_logℓ = [Normal(124,      10),
                  Normal(42,       10),
                  Normal(100,     500),
                  Normal(50,      100),
                  Normal(50,      100),
                  Normal(100,     100), #Normal(10e+6, 20e+6),
                  Normal(100,     100),#Normal(1e+6,  10e+6),
                  Normal(100,     100),
                  Normal(2,         2)
                  ]
    #prior_logℓ = [Normal(-1e+2, 1e+2)]
    prior_logσ = Normal(10, 5)
    prior_logϵ = Normal(10, 5)

    logℓ0 = log.([100, 40, 500, 50, 50, 20e+6, 5e+6, 50, 2])
    #logℓ0  = log.([0.01])
    logσ0 = log(10.0)
    logϵ0 = log(10.0)
    θ0    = vcat(logℓ0, logσ0, logϵ0)

    θ_post, α_post = ess(prng, θ0, X_train, y_train, k, nsamples, nburn,
                         prior_logℓ, prior_logσ, prior_logϵ)

    #display(plot(θ_post[:,:]'))
    #display(plot(θ_post[11,:]))
    #display(Chains(reshape(θ_post', (:,3,1))))
    #throw()
    θ_post, α_post
end

function main(mode = :train)
    prng = MersenneTwister(1)
    total_folds = 5
    for i = 1:total_folds
        X_train, y_train, X_test, y_test = prepare_date(i, 5)

        if(mode == :train)
            θ, α = train_gp(prng, X_train, y_train, X_test, y_test)
            writedlm("hyperparameter_posterior_$(i).csv", θ)
            writedlm("latent_posterior_$(i).csv",         α)
        else
            θ = readdlm("hyperparameter_posterior_$(i).csv")
            α = readdlm("latent_posterior_$(i).csv")

            k       = Matern52Kernel()
            μ_train = predict_gp(α, θ, X_train, k, X_train)
            μ_test  = predict_gp(α, θ, X_train, k, X_test)
            rmse_train = sqrt(mean((y_train - μ_train).^2))
            rmse_test  = sqrt(mean((y_test  - μ_test).^2))
            @info "Accuracy fold #$(i)" test_rmse=rmse_test train_rmse=rmse_train
        end
    end
end

