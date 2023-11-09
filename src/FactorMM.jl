module FactorMM

using Distributions, Statistics

function drawnorm(N = 100_000, μ = 1., σ = 2.)
    return rand(Normal(μ, σ), N)
end

export drawnorm

end
