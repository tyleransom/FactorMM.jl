using FactorMM
using Test
using Statistics

@testset "drawnorm checks" begin
    outp = drawnorm()
    @test size(outp) == (100_000,)
    @test isapprox(mean(outp), 1., atol = 1E-2)
end
