
using CUDA
using Distributions
using KernelFunctions
using Krylov
using LinearAlgebra
using LinearOperators
using ProgressMeter
using Random
using DelimitedFiles

include("util.jl")

function ess(prng, θ0, X, y, k, nsamples, nburn,
             prior_logℓ, prior_logσ, prior_logϵ)
    logpθ = Product([prior_logℓ..., prior_logσ, prior_logϵ])
    μ     = mean(logpθ)

    function calc_mll!(θ_in)
        logℓ = θ_in[1:length(prior_logℓ)]
        logσ = θ_in[length(prior_logℓ)+1]
        logϵ = θ_in[length(prior_logℓ)+2]

        # logℓ = θ_in[1]
        # logσ = θ_in[2]
        # logϵ = θ_in[3]

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
    θ_post = Array{Float32}(undef, dim,       nsamples)
    α_post = Array{Float32}(undef, length(y), nsamples)

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
    ℓinv  = exp.(-logℓ) 
    σ²   = exp(logσ*2) 
    ϵ²   = exp(logϵ*2) 

    ard  = ARDTransform(ℓinv)
    k    = σ² * TransformedKernel(k, ard)

    # scale = ScaleTransform(ℓinv)
    # k     = σ² * TransformedKernel(k, scale)

    K  = kernelmatrix(k, X, obsdim=2)
    K += ϵ²*I

    y_dev = CUDA.CuArray{Float32}(y)
    K_dev = CUDA.CuArray{Float32}(K)

    try
        cholesky!(K_dev)
        L     = K_dev 
        α_dev = y_dev
        CUBLAS.trsv!('U', 'T', 'N', L, α_dev)
        CUBLAS.trsv!('U', 'N', 'N', L, α_dev)

        α     = Array(α_dev)
        t1    = dot(y, α) / -2
        t2    = -sum(log.(Array(diag(L))))
        t3    = -size(K,1) / 2 * log(2*π)
        mll   = t1+t2+t3
        α, mll 
    catch e
        println("chol failed!")
        zeros(size(K, 1)), -Inf
    end
end

function predict_gp(α::CuArray, X_data::Matrix, k, X_in::Matrix)
    println(size(X_data))
    println(size(X_in))
    kstar = kernelmatrix(k, X_data, X_in, obsdim=2)
    kstar = CUDA.CuArray{Float32}(kstar)
    μ     = Array(kstar' * α)
    μ
end

function train_gp(prng, X_train, y_train, X_test, y_test)
    k          = Matern52Kernel()
    nsamples   = 200
    nburn      = 100
    dims       = size(X_train,1)
    prior_logℓ = [Normal(0, 3) for i = 1:dims]
    prior_logσ = Normal(4, 8)
    prior_logϵ = Normal(0, 2)

    logℓ0 = log.(ones(dims) / 2)
    logσ0 = log(10.0)
    logϵ0 = log(2.0)
    θ0    = vcat(logℓ0, logσ0, logϵ0)

    θ_post, α_post = ess(prng, θ0, X_train, y_train, k, nsamples, nburn,
                         prior_logℓ, prior_logσ, prior_logϵ)
    θ_post, α_post
    
    # @info "Accuracy" test_rmse=rmse_test train_rmse=rmse_train

    # ϵ = 0.01
    # @time  α = train_gp(X_train, y_train, k, ϵ)
    # μ = predict_gp(α, X_train, k, X_train)

    # #println(μ)
    # rmse_test = sqrt(mean((y_train - μ).^2))

    # @info "Accuracy" test_rmse=rmse_test


    # 114 < longitude           < 124
    # 32  < latitude            < 42 
    # 0   < housing_median_age  < 1000
    # 1   < total_rooms         < 100
    # 1   < total_bedrooms      < 100
    # 10  < population          < 40e+6
    # 0   < households          < 10e+6
    # 0   < median_income (10k) < 100
    # 1   < ocean_proximity     < 4
end

function main()
    prng = MersenneTwister(1)
    total_folds = 5
    for i = 1:total_folds
        X_train, y_train, X_test, y_test = prepare_date(i, 5)
        θ, α = train_gp(prng, X_train, y_train, X_test, y_test)

        writedlm("hyperparameter_posterior_$(i).csv", θ)
        writedlm("latent_posterior_$(i).csv",         α)
    end
end

