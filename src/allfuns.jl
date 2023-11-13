#-------------------------------------------------------------------------------
# Functions for reading in data
#-------------------------------------------------------------------------------

"""
    Igrp(X, g)

Transforms the input matrix `X` based on the groupings in `g`. 

# Arguments
- `X`: A matrix to be transformed.
- `g`: A vector indicating groupings. Should have the same number of rows as `X`.

# Returns
- A transformed version of `X` where each row is multiplied by a one-hot encoding of its corresponding group in `g`.
"""
function Igrp(X, g)
    I = [ones(size(g, 1)) g .== 2 g .== 3]
    I = reshape(I, size(g, 1), 1, 3)
    X = I .* X
    X = reshape(X, size(X, 1), :)
    return X
end

"""
        choiceX(D::NamedTuple)

Constructs a matrix `X` based on the fields in the NamedTuple `D`.

# Arguments
- `D`: A NamedTuple containing the fields `T`, `sch`, `pvsch`, `exper`, and `pvwrk`.

# Returns
- A matrix `X` where the columns are as follows:
- The first column is all ones, repeated `D.T` times.
- The second column is the `sch` field from `D`.
- The third column is the square of the `sch` field from `D`.
- The fourth column is the `pvsch` field from `D`.
- The fifth column is the `exper` field from `D`.
- The sixth column is the square of the `exper` field from `D`.
- The seventh column is the `pvwrk` field from `D`.
"""
function choiceX(D::NamedTuple)
    intcpt = repeat([1], outer=[D.T, 1])
    X = hcat(
             intcpt,
             D.sch,
             D.sch.^2,
             D.pvsch,
             D.exper,
             D.exper.^2,
             D.pvwrk
            )
    return X
end

"""
        lnwageX(D::NamedTuple)

Constructs a matrix `X` based on the fields in the NamedTuple `D`.

# Arguments
- `D`: A NamedTuple containing the fields `T`, `sch`, and `exper`.

# Returns
- A matrix `X` where the columns are as follows:
- The first column is all ones, repeated `D.T` times.
- The second column is the `sch` field from `D`.
- The third column is the square of the `sch` field from `D`.
- The fourth column is the `exper` field from `D`.
- The fifth column is the square of the `exper` field from `D`.
- The sixth column is the product of the `sch` and `exper` fields from `D`.
"""
function lnwageX(D::NamedTuple)
    intcpt = repeat([1], outer=[D.T, 1])
    X = hcat(
             intcpt,
             D.sch,
             D.sch.^2,
             D.exper,
             D.exper.^2,
             D.sch .* D.exper
            )
    return X
end

"""
    typeX(D::NamedTuple)

Constructs a matrix `X` based on the `grp` field in the NamedTuple `D`.

# Arguments
- `D`: A NamedTuple containing the field `grp`.

# Returns
- A matrix `X` where each row is a one-hot encoding of its corresponding group in `D.grp`.
"""
function typeX(D::NamedTuple)
    X = Igrp(1, D.grp)
    return X
end

"""
    read_data(fpath::String)

Reads data from CSV files and processes them.

# Arguments
- `fpath`: A string representing the file path to the CSV files.

# Returns
    return DATA, measInfo
- `DATA`: A NamedTuple with all data required for estimation
- `measInfo`: A NamedTuple containing information about the names and types of the measurements in `DATA`

# Notes
- The CSV files should be named "pop.csv", "meas.csv", and "panel.csv", and should be located at `fpath`.
- The function assumes that all CSV files have a column named "id" that uniquely identifies each person, and that the "id" columns in both files match.
- The function also assumes that the "pop.csv" file has a column named "race", which is used to create the `grp` field in `pers`.
"""
function read_data(fpath::String)
    # Person information
    pers = CSV.read(fpath * "pop.csv", DataFrame, missingstring=".")
    sort!(pers, :id)
    N = nrow(pers)
    pers.pwt = ones(N)

    pers.grp = ifelse.(ismissing.(pers.race), NaN, pers.race)
    select!(pers, Not(:race))

    # Measurement information
    meas = CSV.read(fpath * "meas.csv", DataFrame, missingstring=".")
    sort!(meas, :id)
    @assert all(meas.id .== pers.id) "id mismatch in meas and pers data"
    select!(meas, Not(:id))

    measName = names(meas)
    
    measType = zeros(length(measName))
    for j in eachindex(measType)
        I::BitVector = .!ismissing.(meas[!, measName[j]])
        uniw = unique(skipmissing(meas[I, measName[j]]))
        if length(uniw) < 12
            w = indexin(meas[!, measName[j]], uniw)
            meas[I, measName[j]] = w[I]
            measType[j] = length(uniw)
        else
            measType[j] = Inf
        end
    end

    measHas = zeros(size(meas))
    for i in axes(meas,1)
        for j in eachindex(measType)
            measHas[i,j] = !ismissing(meas[i, measName[j]])
        end
    end

    measInfo = (name = measName, type = measType)

    # Time-varying information
    panel = CSV.read(fpath * "panel.csv", DataFrame, missingstring=[".",""])
    sort!(panel, [:id, :period])
    @assert all(in.(panel.id, Ref(pers.id))) "id mismatch in panel and pers data"


    # Create DATA NamedTuple in four pieces

    # First piece: read in time-invariant data (incl. measurements)
    DATA = NamedTuple{(    :id, :pwt,    :grp, :meas,           :measHas,       :measY,     :T), 
                      Tuple{Int, Float64, Int, Array{Float64,1}, Array{Bool,1}, NamedTuple, Int}}[
                      (; zip((:id, :pwt, :grp, :meas, :measHas, :measY, :T), 
                      (pers.id[i], 
                       pers.pwt[i],
                       pers.grp[i],
                       collect(meas[i,:]),
                       collect(measHas[i,:]), 
                       NamedTuple{Tuple(Symbol.(measName))}([(isinf(measType[j]) ? meas[i, j] : [meas[i, j] == k for k in 1:measType[j]]) for j in axes(measName,1)]),
                       nrow(panel[panel.id .== pers.id[i], :])))...)
        for i in 1:nrow(pers)
    ]

    # Second piece: read in panel data
    DATA2 = []
    for i in 1:nrow(pers)
        P = panel[panel.id .== pers.id[i], :]
        push!(DATA2, (
            id        = pers.id[i],
            t         = round.(P.period .- minimum(P.period)) .+ 1,
            exper     = P.experA,
            sch       = P.schA,
            choice    = P.choice,
            pvwrk     = P.pvwrkA,
            pvsch     = P.pvschA,
            lnwage    = coalesce.(P.lnWageA, NaN),
            lnwageHas = .!ismissing.(P.lnWageA)
        ))
    end
    DATA = [merge(DATA[i], DATA2[i]) for i in eachindex(DATA)]

    # Third piece: create X and Y matrices for different model equations
    DATA3 = []
    for i in eachindex(DATA)
        wflg::BitArray = DATA[i].lnwageHas
        push!(DATA3, (
            id      = DATA[i].id,
            choiceX = choiceX(DATA[i]), 
            choiceY = hcat([Int.(DATA[i].choice .== k) for k in 1:3]...), 
            choiceN = length(DATA[i].choice),
            lnwageX = lnwageX(DATA[i])[wflg, :], 
            lnwageY = DATA[i].lnwage[wflg], 
            lnwageN = sum(wflg),
            typeX   = typeX(DATA[i])
        ))
    end
    DATA = [merge(DATA[i], DATA3[i]) for i in eachindex(DATA)]

    # Last piece: re-normalize weights
    pwtn = length(DATA) .* [DATA[i].pwt for i in eachindex(DATA)] ./ sum([DATA[i].pwt for i in eachindex(DATA)])
    DATA4 = []
    for i in eachindex(DATA)
        push!(DATA4, (
            id  = DATA[i].id,
            pwt = pwtn[i]
        ))
    end
    DATA = [merge(DATA[i], DATA4[i]) for i in eachindex(DATA)]

    # Save data as a MATLAB file
    matfile = matopen(fpath * "testdataA.mat", "w")
    write(matfile, "DATA", DATA)
    write(matfile, "measInfo", measInfo)
    close(matfile)

    return DATA, measInfo
end

#-------------------------------------------------------------------------------
# Starting values
#-------------------------------------------------------------------------------
"""
    start_values(DATA, measInfo)

This function initializes the starting values for the model estimation process. 

# Arguments
- `DATA`: An array of data structures, where each structure represents an individual in the dataset. Each structure should contain fields for different types of measurements.
- `measInfo`: A structure containing information about the measurements, including their names and types.

# Returns
- `est`: A tuple containing the starting values for the factor means (`factmean`), factor covariance (`factcov`), type coefficients (`typecoef`), measurement coefficients (`meascoef`), measurement variances (`measvar`), choice coefficients (`choicecoef`), wage coefficients (`lnwagecoef`), wage variance (`lnwagevar`), factor names (`factorNames`), and the model type (`model`).

# Notes
The function assumes that there are two factors, "Cognitive" and "Family", and that the model is a "full" model. The starting values are initialized based on the data and measurement information, with some values set to small constants or derived from the data. The function also prints a table of the measurements and their loadings on the factors.
"""
function start_values(DATA, measInfo; modeltype="full")
    nf = 2
    ntype = 2
    numGRP = 0
    factorNames = ["Cognitive", "Family"]

    numW = length(measInfo.name)

    meas = vcat([d.meas for d in DATA]'...)

    measName = measInfo.name
    measSimp = deepcopy(measName)
    measType = measInfo.type

    ##  Factor Info
    measloadMAP = zeros(nf, length(measName))

    # measurements of Cognitive factor
    usemeas = ["asvab1", "asvab2", "asvab3", "asvab4", "asvab5", "asvab6", "AP1", "AP2", "AP3", "AP4", "AP5", "AP6"]
    measloadMAP[1, [name in usemeas for name in measSimp]] .= NaN
    measloadMAP[1, measName .== "asvab1"] .= 1  # normalize loading on asvab1 to 1

    # measurements of Family factor
    usemeas = ["finc1", "finc2", "finc3", "meduc", "feduc"]
    measloadMAP[2, [name in usemeas for name in measSimp]] .= NaN
    measloadMAP[2, measName .== "finc1"] .= 1  # normalize loading on asvab1 to 1

    # measurements of both Cognitive and Family factors
    usemeas = ["GPA1", "GPA2", "GPA3"]
    measloadMAP[1:2, [name in usemeas for name in measSimp]] .= NaN

    measTab = DataFrame(Name = measName, Cognitive = measloadMAP[1, :], Family = measloadMAP[2, :], Type = measType)
    sort!(measTab, :Name)
    println(measTab)

    # normalize constant of asvab1 to 0
    measNoConst = [findfirst(occursin.(x, measName)) for x in ["asvab1", "finc1"]]

    ## Starting Values
    factmean = zeros(nf)
    factcov = eps() * Matrix{Float64}(I, nf, nf)
    for f = 1:nf
        factmean[f] = nanmean(meas[:, measNoConst[f]]) * measloadMAP[f, measNoConst[f]]
        factcov[f, f] = nanvar(meas[:, measNoConst[f]]) / 2
    end
    factmean = kron(factmean, rand(1, ntype))

    typecoef = 2e-16 * ones(size(typeX(DATA[1]), 2), ntype-1)

    meascoef = Vector{Array{Float64,2}}(undef, numW)
    measvar = fill(NaN, numW)
    for j = 1:numW
        W = meas[.!isnan.(meas[:, j]), j]
        if isinf(measType[j])
            consty = !isnothing(indexin(j, measNoConst)[1]) ? 0 : mean(W)
            meascoef[j] = [consty; reshape(measloadMAP[:, j], :, 1)]
            measvar[j] = var(W)
        else
            consty = mean([W[i]==k for i in eachindex(W), k in 1:measType[j]], dims=1)
            consty = log.(consty[2:end] ./ consty[1])'
            meascoef[j] = [consty; measloadMAP[:, j] * ones(1, Int64(measType[j])-1)]
        end
        meascoef[j][isnan.(meascoef[j])] .= 2e-16
    end

    choicecoef = 2e-16 * ones(size(choiceX(DATA[1]), 2) + nf, size(DATA[1].choiceY, 2)-1)
    lnwagecoef = 2e-16 * ones(size(lnwageX(DATA[1]), 2) + nf)
    lnwagevar = 0.5

    factorNames = ["Cognitive", "Family"]

    model = modeltype

    est = (factmean = factmean, factcov = factcov, typecoef = typecoef, meascoef = meascoef, measvar = measvar, choicecoef = choicecoef, lnwagecoef = lnwagecoef, lnwagevar = lnwagevar, factorNames = factorNames, model = model)

    return est
end    


#-------------------------------------------------------------------------------
# Functions for estimation
#-------------------------------------------------------------------------------
"""
    updateReg(XX, XY, coef, outcometype)

Update the coefficients of a regression model based on the outcome type.

# Arguments
- `XX`: The XX matrix in the regression model.
- `XY`: The XY matrix in the regression model.
- `coef`: The current coefficients of the model.
- `outcometype`: The type of the outcome variable. Can be "discrete" or "continuous".

# Returns
- The updated coefficients of the model.
"""
function updateReg(XX::Matrix{Float64}, XY::Union{Array{T, 1}, Array{T, 2}} where T <: Float64, coef::Array, outcometype::String)
    ndp = vec(indexin(coef, [0, 1, -1]) .== nothing)

    if outcometype == "discrete"
        nc = size(XY, 2) + 1
        LBXX = kron(-(1/2) * (I(nc-1) - ones(nc-1,nc-1) ./ nc), XX)
        coef[ndp] = coef[ndp] - (LBXX[ndp[:], ndp[:]] \ XY[ndp])
    elseif outcometype == "continuous"
        coef[ndp] = XX[ndp, ndp] \ (XY[ndp] - XX[ndp, .!ndp] * coef[.!ndp])
    else
        error("Unknown Outcome Type: $outcometype")
    end

    return coef
end


"""
    SuffStatsFun(D::NamedTuple, est::NamedTuple, R::Int64, samplemethod::String)

This function calculates the sufficient statistics for a given observation of data, estimation parameters, number of samples, and sampling method.

# Arguments
- `D`: The dataset *for one observation*, which includes the type of data, measurements, choices, and wages.
- `est`: The estimation parameters, which includes the factor mean, factor covariance, measurement coefficients, choice coefficients, wage coefficients, and model type.
- `R`: The number of simulation draws.
- `samplemethod`: The sampling method, which can be "independent" or "mass".

# Returns
- `suffstats`: The sufficient statistics, which includes the log-likelihood, type_y, Sfact, Nfact, Sfact2, meas_x, meas_xx, meas_xy, choice_xx, choice_xy, lnwage_xx, lnwage_xy, lnwage_yy, and lnwage_n.
- `L`: The overall likelihood for the given observation of data.
- `θi`: The sampled factor vector.
"""
function SuffStatsFun(D::NamedTuple, est::NamedTuple, R::Int64, samplemethod::String)

    numFact, numType = size(est.factmean)
    numW = length(est.meascoef)

    type_x = D.typeX
    expv = exp.([0 type_x*est.typecoef])
    typepr0 = expv ./ sum(expv, dims=2)
    E = est.factmean
    V = est.factcov

    if samplemethod == "independent"
        typeI = rand(Categorical(typepr0), R)
    elseif samplemethod == "mass"
        RperType = ceil(Int, R / numType)
        R = RperType * numType
        typeI = repeat((1:numType), inner=RperType)
    end
    Ei = E[:, typeI]
    θi = hcat([rand(MvNormal(Ei[:, k], Matrix(Hermitian(V)))) for k in axes(Ei, 2)]...)'
    
    # likelihood of measurements
    meas_x = [ones(R,1) θi]
    meas_y = Vector{Array{Float64,2}}(undef, numW)
    L_meas = ones(R,1)
    for j in findall(D.measHas)
        if !isnan(est.measvar[j])
            L = pdf.(Normal(0,1), (D.measY[j] .- meas_x*est.meascoef[j]) ./ sqrt(est.measvar[j]))
        else
            meas_y[j] = repeat(D.measY[j]', R)
            expv = exp.([zeros(size(meas_x,1),1) meas_x*est.meascoef[j]])            
            pr = expv ./ sum(expv, dims=2)
            L = prod(pr .^ meas_y[j], dims=2)
            meas_y[j] = meas_y[j][:,2:end] - pr[:,2:end]
        end
        L_meas .*= L
    end

    # likelihood choice
    L_choice = ones(R,1)
    if D.choiceN > 0
        choice_x = [repeat(D.choiceX, R, 1) θi[repeat(ones(Int, D.choiceN), R),:]]
        choice_y = repeat(D.choiceY, R, 1)
        expv = exp.([zeros(size(choice_x,1),1) choice_x*est.choicecoef])    
        pr = expv ./ sum(expv, dims=2)
        L_choice = prod(reshape(prod(pr .^ choice_y, dims=2), D.choiceN, R), dims=1)'  
        choice_y = choice_y[:,2:end] - pr[:,2:end]    
    end

    # likelihood lnwage
    L_lnwage = ones(R,1)
    lnwage_x = []
    if D.lnwageN > 0
        lnwage_x = [repeat(D.lnwageX, R, 1) θi[repeat(ones(Int, D.lnwageN), R),:]]
        lnwage_pr = pdf.(Normal(0,1), (repeat(D.lnwageY, R, 1) - lnwage_x*est.lnwagecoef) ./ sqrt(est.lnwagevar))
        L_lnwage = prod(reshape(lnwage_pr, D.lnwageN, R), dims=1)'
    end

    # Complete Likelihood
    if est.model == "meas only"
        L = L_meas
    elseif est.model == "meas and choice"
        L = L_choice .* L_meas
    elseif est.model == "full"
        L = L_lnwage .* L_choice .* L_meas
    else
        error("unrecognized model in estimation structure")
    end
    
    if samplemethod == "independent"
        ll = log(sum(L)/R)
    elseif samplemethod == "mass"
        L = reshape(typepr0[typeI], :, 1) .* L
        ll = log(sum(L)/RperType)
    end

    qi = L ./ sum(L)

    # Suff Stats for Distribution 
    θ_pdf = zeros(R, numType)
    for k in 1:numType
        #θ_pdf[:, k] = [pdf.(MvNormal(est.factmean[:, k], est.factcov), θi[i, :] for i in 1:R)]
        dist = MvNormal(est.factmean[:, k], est.factcov)
        θ_pdf[:, k] = [pdf(dist, θi[i, :]) for i in axes(θi, 1)]
    end

    typewt = typepr0 .* θ_pdf
    typewt = typewt ./ sum(typewt, dims=2)
    typewt = qi .* typewt

    Nfact = sum(typewt, dims=1)
    Sfact = θi' * typewt
    Sfact2 = θi .* reshape(θi, R, 1, numFact)
    Sfact2 = reshape(Sfact2, R, numFact^2)' * typewt

    if numType > 1
        type_y = sum(Nfact[:, 2:end], dims=1) - typepr0[2:end]
    else
        type_y = zeros(1, 0)
    end

    # Suff Stats for meas coef and meas var
    wtmeas_x = qi .* meas_x
    meas_xx = wtmeas_x' * meas_x
    meas_x = sum(wtmeas_x, dims=1)'
    meas_xy = [Array{Float64,2}(undef, 0, 0) for _ in 1:numW]
    for j in findall(D.measHas .& isnan.(est.measvar))
        meas_xy[j] = (wtmeas_x' * meas_y[j])
    end
    
    # Suff Stats for choice
    choice_xx = []
    choice_xy = []
    if D.choiceN > 0
        wtchoice_x = qi[repeat(ones(Int, D.choiceN), R), 1] .* choice_x
        choice_xx = wtchoice_x' * choice_x
        choice_xy = wtchoice_x' * choice_y
    end

    # Suff Stats for lnwage
    lnwage_xx = []
    lnwage_xy = []
    lnwage_yy = []
    lnwage_n  = []
    if D.lnwageN > 0
        wtlnwage_x = qi[repeat(ones(Int, D.lnwageN), R), 1] .* lnwage_x
        lnwage_xx  = wtlnwage_x' * lnwage_x
        lnwage_xy  = wtlnwage_x' * repeat(D.lnwageY, R, 1)
        lnwage_yy  = sum(qi[repeat(ones(Int, D.lnwageN), R), 1] .* repeat(D.lnwageY, R, 1) .* repeat(D.lnwageY, R, 1))
        lnwage_n   = sum(qi[repeat(ones(Int, D.lnwageN), R), 1])
    end

    suffstats = (
                 ll = ll,
                 type_y = type_y,
                 Sfact = Sfact, 
                 Nfact = Nfact,
                 Sfact2 = Sfact2,
                 meas_x = meas_x,
                 meas_xx = meas_xx,
                 meas_xy = meas_xy,
                 choice_xx = choice_xx,
                 choice_xy = choice_xy,
                 lnwage_xx = lnwage_xx,
                 lnwage_xy = lnwage_xy,
                 lnwage_yy = lnwage_yy,
                 lnwage_n = lnwage_n
                )

    return suffstats, L, θi
end


"""
    vecparm(sv)

Transforms the input `sv` into a different representation and returns a function `p2p` that can perform this transformation.

# Arguments
- `sv`: A structure containing various parameters and coefficients. It should have the following fields:
- `factmean`: A matrix representing factor means.
- `factorNames`: Names of the factors.
- `model`: The model being used.
- `factcov`: A matrix representing factor covariances.
- `meascoef`: A vector of measurement coefficients.
- `measvar`: A vector of measurement variances.
- `choicecoef`: A vector of choice coefficients.
- `lnwagecoef`: A vector of log wage coefficients.
- `lnwagevar`: A vector of log wage variances.

# Returns
- `p2p`: A function that takes a NamedTuple or a vector and transforms it into the representation used by `sv`.

# Notes
- The transformation involves reshaping, exponentiation, and other operations.
- The returned function `p2p` can be used to transform new data into the same representation as `sv`.
"""
function vecparm(sv)

    nf, K = size(sv.factmean)

    factorNames = sv.factorNames
    model = sv.model

    factcovFULL = !isdiag(sv.factcov)

    split_meascoef = size.(sv.meascoef, 2)
    meascoefNORM = hcat(sv.meascoef...)
    meascoefEST = (indexin(meascoefNORM, [0, -1, 1]).==nothing) .& (.!isnan.(meascoefNORM))

    measvarNORM = sv.measvar
    measvarEST = (indexin(measvarNORM, [0, -1, 1]).==nothing) .& (.!isnan.(measvarNORM))

    choicecoefNORM = sv.choicecoef
    choicecoefEST = (indexin(choicecoefNORM, [0, -1, 1]).==nothing) .& (.!isnan.(choicecoefNORM))

    lnwagecoefNORM = sv.lnwagecoef
    lnwagecoefEST = (indexin(lnwagecoefNORM, [0, -1, 1]).==nothing) .& (.!isnan.(lnwagecoefNORM))

    lnwagevarNORM = sv.lnwagevar
    lnwagevarEST = (indexin(lnwagevarNORM, [0, -1, 1]).==nothing) .& (.!isnan.(lnwagevarNORM))
    
    nnz = x -> count(!iszero, x)

    split_vec = [nf * K;
                 nf + factcovFULL * (nf^2 - nf) ÷ 2;
                 nnz(meascoefEST);
                 nnz(measvarEST);
                 nnz(choicecoefEST);
                 nnz(lnwagecoefEST);
                 nnz(lnwagevarEST)]

    function calc(X)
        if !isa(X, NamedTuple)
            println("enter !isa loop")
            X = [X[i:sum(split_vec[1:j])] for (j, i) in enumerate(cumsum([1; split_vec[1:end-1]]))]

            Y = deepcopy(sv)

            Y.factmean = reshape(X[1], nf, K)

            if factcovFULL
                covchol = zeros(nf, nf)
                covchol[tril(true(nf, nf))] = X[2]
                factcov = covchol * covchol'
                factcov = (factcov + factcov') ./ 2
            else
                factcov = Diagonal(exp.(X[2]))
            end
            Y.factcov = factcov

            meascoef = meascoefNORM
            meascoef[meascoefEST] = X[3]
            Y.meascoef = [meascoef[1:nf+1] for n in split_meascoef]

            measvar = measvarNORM
            measvar[measvarEST] = exp.(X[4])
            Y.measvar = measvar

            choicecoef = choicecoefNORM
            choicecoef[choicecoefEST] = X[5]
            Y.choicecoef = choicecoef

            lnwagecoef = lnwagecoefNORM
            lnwagecoef[lnwagecoefEST] = X[6]
            Y.lnwagecoef = lnwagecoef

            lnwagevar = lnwagevarNORM;
            lnwagevar[lnwagevarEST] = exp.(X[7])
            Y.lnwagevar = lnwagevar
    
            Y.factorNames = factorNames
            Y.model = model
        else
            if factcovFULL
                covchol = cholesky(X.factcov, Val(true)).L
                factcovtrans = covchol[tril(true(nf, nf))]
            else
                factcovtrans = log.(diag(X.factcov))
            end

            meascoef = hcat(X.meascoef...)

            Y = [X.factmean[:];
                 factcovtrans;
                 meascoef[meascoefEST];
                 log.(X.measvar[measvarEST]);
                 X.choicecoef[choicecoefEST];
                 X.lnwagecoef[lnwagecoefEST];
                 X.lnwagevar[lnwagevarEST]]
        end
        return Y
    end

    p2p = calc

    return p2p
end


"""
    em_alg(EMstep::Function, sv::NamedTuple, p2p::Function; maxIter=1e6, TolX=1e-6, TolRelX=0, TolFun=0, TolRelFun=0, printIter=1, savelocation=nothing)

This function implements the Expectation-Maximization (EM) algorithm for a given EM step function, initial parameters, and a function to update parameters.

# Arguments
- `EMstep::Function`: The function to perform the E-step and M-step of the EM algorithm.
- `sv::NamedTuple`: The initial parameters for the EM algorithm.
- `p2p::Function`: A function to update the parameters.

# Keyword Arguments
- `maxIter=1e6`: The maximum number of iterations for the EM algorithm.
- `TolX=1e-6`: The tolerance for the change in parameters.
- `TolRelX=0`: The tolerance for the relative change in parameters.
- `TolFun=0`: The tolerance for the change in log-likelihood.
- `TolRelFun=0`: The tolerance for the relative change in log-likelihood.
- `printIter=1`: The number of iterations between printing progress.
- `savelocation=nothing`: The location to save the parameters and output. If `nothing`, no data is saved.

# Returns
The function does not return any value but saves the parameters and output at the specified location if provided.
"""
function em_alg(EMstep::Function, sv::NamedTuple, p2p::Function; maxIter=1e6, TolX=1e-6, TolRelX=0, TolFun=0, TolRelFun=0, printIter=1, savelocation=nothing)

    abch(x) = x[:, 1] .- x[:, 2]
    rech(x) = (x[:, 1] .- x[:, 2]) ./ (x[:, 2] .+ 1e-6)
    pach(x) = ifelse.(abs.(x[:, 2]) .< .01, abch(x), rech(x))

    epnorm(x) = x ./ (x' * x)
    epval(x, y, z) = y + epnorm(epnorm(z - y) + epnorm(x - y))

    est = deepcopy(sv)
    oll = -Inf
    op = p2p(est) * ones(1, 3)
    ep = op[:, 1:2]

    output = nothing
    tic = time()
    for m in 1:maxIter
        est, ll = EMstep(est)

        op = hcat(p2p(est), op[:, 1:end-1])
        ep = hcat(epval(op[:, 1], op[:, 2], op[:, 3]), ep[:, 1:end-1])

        output = (Iterations = m,
                  LogLike = ll,
                  ChangeInLogLike = ll - oll,
                  ChangeInParms = norm(abch(op), Inf),
                  PercentChangeInParms = norm(pach(op), Inf),
                  ChangeInVecEps = norm(abch(ep), Inf),
                  Minutes = (time() - tic) / 60)

        HasConverged = any([norm(abch(op), Inf) < TolX,
                            norm(pach(op), Inf) < TolRelX,
                            norm(ll - oll, Inf) < TolFun,
                            norm((ll - oll) / oll, Inf) < TolRelFun])

        if m == 1 || m % printIter == 0 || HasConverged
            println("$m \t $ll \t $(ll - oll) \t $((ll - oll) / abs(oll)) \t $(norm(abch(op), Inf)) \t $(sum(abs.(abch(op)) .>= TolX)) \t $(norm(pach(op), Inf)) \t $(sum(abs.(pach(op)) .>= TolRelX)) \t $(norm(abch(ep), Inf)) \t $(time() - tic)")
        end

        if !isnothing(savelocation)
            matfile = matopen(savelocation, "w") 
            write(matfile, "est", est)
            write(matfile, "output", output)
            write(matfile, "op", op)
            close(matfile)
        end

        if HasConverged
            break
        end

        if norm(abch(ep), Inf) < TolX
            est = p2p(ep[:, 1])
        end

        oll = ll
    end
    return est, output
end


"""
    MM(DATA::Vector{<:NamedTuple}, R::Int64, sv::NamedTuple; maxIter=1e6, TolX=1e-6, TolRelX=0, TolFun=0, TolRelFun=0, printIter=1, savelocation=nothing)

This function performs the MM (Maximization-Minorization) algorithm on the given data.

# Arguments
- `DATA`: A vector of NamedTuples, where each NamedTuple represents a data point.
- `R`: An integer representing the number of simulation draws, which is passed as a parameter to the `SuffStatsFun` function.
- `sv`: A NamedTuple containing initial values for the parameters to be estimated.

# Keyword Arguments
- `maxIter`: Maximum number of iterations for the EM algorithm (default is 1e6).
- `TolX`: Tolerance for the change in parameters (default is 1e-6).
- `TolRelX`: Relative tolerance for the change in parameters (default is 0).
- `TolFun`: Tolerance for the change in the log-likelihood (default is 0).
- `TolRelFun`: Relative tolerance for the change in the log-likelihood (default is 0).
- `printIter`: Number of iterations between printing of intermediate results (default is 1).
- `savelocation`: Location to save intermediate results (default is `nothing`).

# Returns
- `est`: A NamedTuple containing the estimated parameters.

# Description
The function first prepares the data and then iteratively updates the estimates of the parameters using the EM algorithm. The updates are done based on the sufficient statistics calculated for each data point. The function returns the final estimates of the parameters once convergence has been achieved.

"""
function MM(DATA::Vector{<:NamedTuple}, R::Int64, sv::NamedTuple; maxIter=1e6, TolX=1e-6, TolRelX=0, TolFun=0, TolRelFun=0, printIter=1, savelocation=nothing)

    numFact, numType = size(sv.factmean)

    N = size(DATA, 1)
    pwt = vcat([d.pwt for d in DATA]'...)
    pwt_dist = reshape(pwt, 1, 1, :)

    measHas = vcat([d.measHas for d in DATA]'...)
    numW = size(measHas,2)
    pwt_meas = [reshape(findall(!iszero, pwt .* measHas[j]), 1, 1, :) for j in 1:numW]
    measY = vcat([collect(values(d.meas))' for d in DATA]...)

    pwt_choice = reshape(pwt[vcat([d.choiceN for d in DATA]'...) .> 0], 1, 1, :)
    pwt_lnwage = reshape(pwt[vcat([d.lnwageN for d in DATA]'...) .> 0], 1, 1, :)

    type_x = vcat([d.typeX for d in DATA]...)
    wttype_x = pwt .* type_x
    type_xx = wttype_x' * type_x

    function calc(est::NamedTuple)
        
        suffstats = NamedTuple[]
        
        for i = 1:N
            Random.seed!(i)
            suffstatsi, _, _ = SuffStatsFun(DATA[i], est, R, "mass")
            push!(suffstats, suffstatsi)
        end

        ll = sum(getfield.(suffstats, :ll) .* pwt)

        Sfact = zeros(size(getfield(suffstats[1], :Sfact)))
        Nfact = zeros(size(getfield(suffstats[1], :Nfact)))
        for i = 1:N
            Sfact .+= pwt_dist[i] .* suffstats[i].Sfact
            Nfact .+= pwt_dist[i] .* suffstats[i].Nfact
        end
        est = merge(est, (factmean = Sfact ./ Nfact,))
        Sfact2 = zeros(size(suffstats[1].Sfact2))
        for i = 1:N
            Sfact2 .+= pwt_dist[i] .* suffstats[i].Sfact2
        end
        Sfact2 = reshape(Sfact2, (numFact, numFact, numType))
        Sfactresid2 = zeros((numFact, numFact, numType))
        for k = 1:numType
            Sfactresid2[:, :, k] = Sfact2[:, :, k] - est.factmean[:, k] * Sfact[:, k]' - Sfact[:, k] * est.factmean[:, k]' + Nfact[k] * est.factmean[:, k] * est.factmean[:, k]'
        end
        factcov = sum(Sfactresid2, dims = 3) ./ sum(Nfact)
        factcov = dropdims(factcov, dims = 3)
        if isdiag(est.factcov)
            est = merge(est, (factcov = Diagonal(diag(factcov)),))
        else
            est = merge(est, (factcov = factcov,))
        end

        type_xy = wttype_x' * vcat(getfield.(suffstats, :type_y)...)
        est = merge(est, (typecoef = updateReg(type_xx, type_xy, est.typecoef, "discrete"),))

        meas_x = hcat(getfield.(suffstats, :meas_x)...)
        meas_xx = cat(dims = 3, getfield.(suffstats, :meas_xx)...)
        meas_xy = permutedims(cat([x for x in getfield.(suffstats, :meas_xy)]..., dims = 3), (3, 1, 2))
        for j = 1:length(est.meascoef)
            I = measHas[:, j]
            XX = sum(pwt_meas[j] .* meas_xx[:, :, I], dims = 3)
            XX = dropdims(XX, dims = 3)
            if !isnan(est.measvar[j])
                Y = measY[I, j]
                wtY = Y .* reshape(pwt_meas[j], :, 1)
                XY = meas_x[:, I] * wtY
                YY = wtY' * Y
                c = updateReg(XX, XY, est.meascoef[j], "continuous")
                SSR = (YY .- c' * XY .- XY' * c .+ c' * XX * c)
                est = merge(est, (meascoef = setindex!(est.meascoef, c, j), 
                                  measvar  = setindex!(est.measvar, SSR[1] / sum(pwt_meas[j]), j)))
            else
                XY = sum([pwt_meas[j][:][i] * meas_xy[i, j, 1] for i in 1:N])
                est = merge(est, (meascoef = setindex!(est.meascoef, updateReg(XX,XY,est.meascoef[j],"discrete"), j),))
            end
        end

        type_xy = wttype_x' * vcat(getfield.(suffstats, :type_y)...)
        est = merge(est, (typecoef = updateReg(type_xx, type_xy, est.typecoef, "discrete"),))

        XX = sum(pwt_choice .* cat(dims = 3, getfield.(suffstats, :choice_xx)...), dims = 3)
        XY = sum(pwt_choice .* cat(dims = 3, getfield.(suffstats, :choice_xy)...), dims = 3)
        XX = dropdims(XX, dims = 3)
        XY = dropdims(XY, dims = 3)
        est = merge(est, (choicecoef = updateReg(XX, XY, est.choicecoef, "discrete"),))

        XX = sum(pwt_lnwage .* cat(dims = 3, filter(x -> !isempty(x), getfield.(suffstats, :lnwage_xx))...), dims = 3)
        XY = sum(pwt_lnwage .* cat(dims = 3, filter(x -> !isempty(x), getfield.(suffstats, :lnwage_xy))...), dims = 3)
        XX = dropdims(XX, dims = 3)
        XY = dropdims(XY, dims = 3)
        YY = sum(pwt_lnwage .* cat(dims = 3, filter(x -> !isempty(x), getfield.(suffstats, :lnwage_yy))...), dims = 3) 
        YY = dropdims(YY, dims = 3)
        nt = sum(pwt_lnwage .* cat(dims = 3, filter(x -> !isempty(x), getfield.(suffstats, :lnwage_n))...), dims = 3)  
        c = updateReg(XX, XY, est.lnwagecoef, "continuous")
        SSR = (YY .- c' * XY .- XY' * c .+ c' * XX * c)
        est = merge(est, (lnwagecoef = c, 
                          lnwagevar = SSR[1] / nt[1]))
        return est, ll
    end

    function calc_old(est::NamedTuple)
        
        suffstats = NamedTuple[]
        
        for i = 1:N
            Random.seed!(i)
            suffstatsi, _, _ = SuffStatsFun(DATA[i], est, R, "mass")
            push!(suffstats, suffstatsi)
        end

        ll = sum(getfield.(suffstats, :ll) .* pwt)

        Sfact = sum(pwt_dist .* cat(dims = 3, getfield.(suffstats, :Sfact)...), dims = 3)
        Nfact = sum(pwt_dist .* cat(dims = 3, getfield.(suffstats, :Nfact)...), dims = 3)
        Sfact = dropdims(Sfact, dims = 3)
        Nfact = dropdims(Nfact, dims = 3)
        est = merge(est, (factmean = Sfact ./ Nfact,))

        Sfact2 = sum(pwt_dist .* cat(dims = 3, getfield.(suffstats, :Sfact2)...), dims = 3)
        Sfact2 = reshape(Sfact2, (numFact, numFact, numType))
        Sfactresid2 = zeros((numFact, numFact, numType))
        for k = 1:numType
            Sfactresid2[:, :, k] = Sfact2[:, :, k] - est.factmean[:, k] * Sfact[:, k]' - Sfact[:, k] * est.factmean[:, k]' + Nfact[k] * est.factmean[:, k] * est.factmean[:, k]'
        end
        factcov = sum(Sfactresid2, dims = 3) ./ sum(Nfact)
        factcov = dropdims(factcov, dims = 3)
        if isdiag(est.factcov)
            est = merge(est, (factcov = Diagonal(diag(factcov)),))
        else
            est = merge(est, (factcov = (factcov + factcov') ./ 2,))
        end

        type_xy = wttype_x' * vcat(getfield.(suffstats, :type_y)...)
        est = merge(est, (typecoef = updateReg(type_xx, type_xy, est.typecoef, "discrete"),))

        meas_x = hcat(getfield.(suffstats, :meas_x)...)
        meas_xx = cat(dims = 3, getfield.(suffstats, :meas_xx)...)
        meas_xy = permutedims(cat([x for x in getfield.(suffstats, :meas_xy)]..., dims = 3), (3, 1, 2))
        for j = 1:length(est.meascoef)
            I = measHas[:, j]
            XX = sum(pwt_meas[j] .* meas_xx[:, :, I], dims = 3)
            XX = dropdims(XX, dims = 3)
            if !isnan(est.measvar[j])
                Y = measY[I, j]
                wtY = Y .* reshape(pwt_meas[j], :, 1)
                XY = meas_x[:, I] * wtY
                YY = wtY' * Y
                c = updateReg(XX, XY, est.meascoef[j], "continuous")
                SSR = (YY .- c' * XY .- XY' * c .+ c' * XX * c)
                est = merge(est, (meascoef = setindex!(est.meascoef, c, j), 
                                  measvar  = setindex!(est.measvar, SSR[1] / sum(pwt_meas[j]), j)))
            else
                XY = sum([pwt_meas[j][:][i] * meas_xy[i, j, 1] for i in 1:N])
                est = merge(est, (meascoef = setindex!(est.meascoef, updateReg(XX,XY,est.meascoef[j],"discrete"), j),))
            end
        end

        type_xy = wttype_x' * vcat(getfield.(suffstats, :type_y)...)
        est = merge(est, (typecoef = updateReg(type_xx, type_xy, est.typecoef, "discrete"),))

        XX = sum(pwt_choice .* cat(dims = 3, getfield.(suffstats, :choice_xx)...), dims = 3)
        XY = sum(pwt_choice .* cat(dims = 3, getfield.(suffstats, :choice_xy)...), dims = 3)
        XX = dropdims(XX, dims = 3)
        XY = dropdims(XY, dims = 3)
        est = merge(est, (choicecoef = updateReg(XX, XY, est.choicecoef, "discrete"),))

        XX = sum(pwt_lnwage .* cat(dims = 3, filter(x -> !isempty(x), getfield.(suffstats, :lnwage_xx))...), dims = 3)
        XY = sum(pwt_lnwage .* cat(dims = 3, filter(x -> !isempty(x), getfield.(suffstats, :lnwage_xy))...), dims = 3)
        XX = dropdims(XX, dims = 3)
        XY = dropdims(XY, dims = 3)
        YY = sum(pwt_lnwage .* cat(dims = 3, filter(x -> !isempty(x), getfield.(suffstats, :lnwage_yy))...), dims = 3) 
        YY = dropdims(YY, dims = 3)
        nt = sum(pwt_lnwage .* cat(dims = 3, filter(x -> !isempty(x), getfield.(suffstats, :lnwage_n))...), dims = 3)  
        c = updateReg(XX, XY, est.lnwagecoef, "continuous")
        SSR = (YY .- c' * XY .- XY' * c .+ c' * XX * c)
        est = merge(est, (lnwagecoef = c, 
                          lnwagevar = SSR[1] / nt[1]))
        return est, ll
    end

    p2p = vecparm(sv)
    est, _ = em_alg(calc, sv, p2p; maxIter=maxIter, TolX=TolX, TolRelX=TolRelX, TolFun=TolFun, TolRelFun=TolRelFun, printIter=printIter, savelocation=savelocation)

    return est
end


"""
    bootsamp(DATA, s)

This function performs bootstrapping on the given data. It multiplies the `pwt` field of each element in `DATA` by the number of times that element's index is randomly selected.

# Arguments
- `DATA`: A vector of NamedTuples, where each NamedTuple represents a data point.
- `s`: The seed (integer) for the random number generator.

# Returns
- `DATA`: The bootstrapped data.
"""
function bootsamp(DATA::Vector{<:NamedTuple}, s::Int64)
    if s != 0
        N = length(DATA)
        pwt_total = sum([d.pwt for d in DATA])

        Random.seed!(s)

        ids = rand(1:N, 2*N)

        ids_end = findfirst(cumsum([DATA[id].pwt for id in ids]) .>= pwt_total)
        
        ids = ids[1:ids_end]
        ids = countmap(ids)
        ids_repeat = [v for v in values(ids)]
        ids = [k for k in keys(ids)]
            
        nDATA = DATA[ids]
        
        nDATA = [merge(d, (pwt = d.pwt * ids_repeat[i],)) for (i, d) in enumerate(DATA[ids])]
    else
        nDATA = DATA
    end
    return nDATA
end


#-------------------------------------------------------------------------------
# Main estimation function
#-------------------------------------------------------------------------------
"""
    estimate_model(DATA::Vector{<:NamedTuple}, sv::NamedTuple, fpath::Union{String, Nothing}, bootiter::Int, simdraws::Int)

Estimates the model using bootstrap iterations and simulation draws.

# Arguments
- `DATA::Vector{<:NamedTuple}`: The data as a vector of NamedTuples.
- `sv::NamedTuple`: The starting values for the estimation algorithm.
- `fpath::String`: A string representing the file path where the estimation results will be saved, or nothing if saving is not desired.
- `bootiter::Int`: An integer representing the number of bootstrap iterations.
- `simdraws::Int`: An integer representing the number of simulation draws.

# Procedure
- The function performs bootstrap iterations. In each iteration, it sets a save location and runs the estimation algorithm on the bootstrapped data.
- The estimation results are saved at the specified location.

# Returns
- The function does not return any value.

# Notes
- The estimation algorithm used is `MM`, which is not defined in this function.
- The bootstrap sampling is performed by the `bootsamp` function, which is also not defined in this function.
"""
function estimate_model(DATA::Vector{<:NamedTuple}, sv::NamedTuple, fpath::Union{String, Nothing}, bootiter::Int, simdraws::Int; maxIter=1e6, TolX=1e-6, TolRelX=0, TolFun=0, TolRelFun=0, printIter=1)
    # Loop over bootstrap iterations
    for bs in 0:bootiter
        # Set save location
        if isnothing(fpath)
            saveloc = nothing
        else
            saveloc = fpath * "FM_$(bs)"
        end 

        # Estimation algorithm
        _ = MM(bootsamp(DATA, bs), simdraws, sv; maxIter=maxIter, TolX=TolX, TolRelX=TolRelX, TolFun=TolFun, TolRelFun=TolRelFun, printIter=printIter, savelocation=saveloc) 
    end

    return nothing
end