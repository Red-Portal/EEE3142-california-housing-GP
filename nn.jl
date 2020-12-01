
using Flux
using Statistics
using Flux.Data: DataLoader
using Flux: throttle, @epochs
using Base.Iterators: repeated
using CUDA
using Random
using Plots
using DelimitedFiles
using ProgressMeter

include("util.jl")

if has_cuda()		# Check if CUDA is available
    @info "CUDA is on"
    CUDA.allowscalar(false)
end

function train_mlp(prng, X_train, y_train, X_test, y_test)
    η          = 1.0
    batch_size = 32
    epochs     = 10

    X_train = X_train |> gpu
    X_test  = X_test  |> gpu
    y_train = y_train |> gpu
    y_test  = y_test  |> gpu

    train_data = DataLoader(X_train, y_train, batchsize=batch_size, shuffle=true) |> gpu
    test_data  = DataLoader(X_test,  y_test, batchsize=batch_size) |> gpu

    train_loss_hist = Float64[]
    test_loss_hist  = Float64[]

    m = Chain(Dense(size(X_train, 1), 512, tanh),
              Dropout(0.5),
              Dense(512, 512, tanh),
              Dropout(0.5),
              Dense(512, 1)) |> gpu

    mse(ŷ, y)  = mean((ŷ .- y).^2)
    loss(x, y) = mse(m(x), y)

    ## Training
    evalcb = () -> begin
        train_rmse = sqrt(loss(X_train, y_train))
        test_rmse  = sqrt(loss(X_test,  y_test))

        Flux.testmode!(m, true)
        push!(train_loss_hist, train_rmse)
        push!(test_loss_hist,  test_rmse)
        Flux.testmode!(m, false)

        display(Plots.plot(train_loss_hist, label="Train RMSE"))
        display(Plots.plot!(test_loss_hist, label="Test  RMSE"))
        #@show(train_rmse, test_rmse)
    end

    opt = ADAM(η)
    @showprogress for epoch = 1:epochs
        Flux.train!(loss, params(m), train_data, opt, cb = evalcb)
        if epoch == 3 || epoch == 6
            opt.eta /= 5.0
        end
    end
    @show("Final RMSE",
          train_rmse = sqrt(loss(X_train, y_train)),
          test_rmse  = sqrt(loss(X_train, y_test)))

    train_loss_hist, test_loss_hist 
end

function main()
    prng = MersenneTwister(1)
    total_folds = 5
    for i = 1:total_folds
        X_train, y_train, X_test, y_test = prepare_date(i, 5)
        train_hist, test_hist = train_mlp(prng, X_train, y_train, X_test, y_test)

        writedlm("mlp_train_rmse_$(i).csv", train_hist)
        writedlm("mlp_test_rmse_$(i).csv",  test_hist)
    end
end
