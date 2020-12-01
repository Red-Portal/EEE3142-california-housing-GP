
using XLSX
using DataFrames
using DataFramesMeta

function prepare_date(num_fold, total_folds)
    xf = XLSX.readtable("train.xlsx", "Sheet1")
    df = DataFrame(xf...)

    ocean_encode(prox) = begin
        if(prox == "INLAND")
            1
        elseif(prox == "<1H OCEAN")
            2
        elseif(prox == "NEAR OCEAN")
            3
        elseif(prox == "NEAR BAY")
            4
        elseif(prox == "ISLAND")
            5
        else
        end
    end

    df = @transform(df, ocean_proximity = ocean_encode.(:ocean_proximity))
    dropmissing!(df)

    X = Array(Array{Float64}(df[:,1:end-1])')
    y = Array{Float64}(df[:,end])

    N         = size(X, 2)
    fold_size = floor(Int, N / total_folds)

    idx_test  = collect((fold_size*(num_fold-1)+1):(fold_size*num_fold))
    idx_train = setdiff(1:N, idx_test)
    X_train   = X[:,idx_train]
    X_test    = X[:,idx_test]
    y_train   = y[idx_train]
    y_test    = y[idx_test]
    X_train, y_train, X_test, y_test
end
