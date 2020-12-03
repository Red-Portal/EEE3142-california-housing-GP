
using CUDA
using Distributions
using KernelFunctions
using LinearAlgebra
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

    ard  = ARDTransform(ℓinv)
    kard = σ² * TransformedKernel(k, ard)

    K  = kernelmatrix(kard, X, obsdim=2)
    K += ϵ²*I

    K_dev = K |> CUDA.CuArray{Float32}
    y_dev = y |> CUDA.CuArray{Float32}
    try
        chol = cholesky!(K_dev)
        α    = chol.U \ (chol.L \ y_dev) |> Array{Float64}

        t1  = dot(y, α) / -2
        t2  = -sum(log.(Array(diag(chol.U))))
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

function train_gp(prng, X_train, y_train, X_test, y_test)
    # 114 < longitude           < 124,    Δ = 10
    # 32  < latitude            < 42,     Δ = 10
    # 0   < housing_median_age  < 1000,   Δ = 1000
    # 1   < total_rooms         < 100,    Δ = 99
    # 1   < total_bedrooms      < 100,    Δ = 99
    # 10  < population          < 40e+6,  Δ ≈ 40e+6
    # 0   < households          < 10e+6,  Δ ≈ 10e+6
    # 0   < median_income (10k) < 100,    Δ = 100
    # 1   < ocean_proximity     < 4,      Δ = 3

    # X_test  = X_train[:,300:400]
    # y_test  = y_train[300:400]
    # X_train = X_train[:,1:200]
    # y_train = y_train[1:200]

    k          = Matern52Kernel()
    nsamples   = 200
    nburn      = 100
    dims       = size(X_train,1)

    σℓ = log(2.0)
    α  = 0.5
    prior_logℓ = [Normal(log(10 * α),    σℓ),
                  Normal(log(10 * α),    σℓ),
                  Normal(log(1000 * α),  σℓ),
                  Normal(log(99 * α),    σℓ*2),
                  Normal(log(99 * α),    σℓ*2),
                  Normal(log(40e+6 * α), σℓ),
                  Normal(log(10e+6 * α), σℓ),
                  Normal(log(100 * α),   σℓ*2),
                  Normal(log(3 * α),     σℓ*2)
                  ]
    prior_logσ = Normal(10, 1)
    prior_logϵ = Normal(10, 1)

    logℓ0 = log.([10/2, 10/2, 1000/2, 99/2, 99/2, 40e+6/2, 10e+6/2, 100/2, 3/2])
    logσ0 = 10.0
    logϵ0 = 10.0
    θ0    = vcat(logℓ0, logσ0, logϵ0)

    θ_post, α_post = ess(prng, θ0, X_train, y_train, k, nsamples, nburn,
                         prior_logℓ, prior_logσ, prior_logϵ)
    # display(plot(θ_post[:,:]'))
    # display(MCMCChains.Chains(reshape(θ_post', (:,11,1))))

    # μ = predict_gp(α_post, θ_post, X_train, k, X_train)
    # println(sqrt(mean((μ - y_train).^2)))
    # μ = predict_gp(α_post, θ_post, X_train, k, X_test)
    # println(sqrt(mean((μ - y_test).^2)))
    # throw()
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

