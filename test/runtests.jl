using FactorMM
using Test
using Distributions
using Random
using Statistics
using NaNStatistics
using LinearAlgebra
using DataFrames
using CSV
using MAT

@testset "updateReg function with fixed coef" begin
    # Define test data
    XXd = [100000.0 190379  667307  
           190379.0 667307  2566277 
           667307.0 2566277 10420439]
    XXc = [40670  108078  434544  67382.0
           108078 434544  1814574 202403
           434544 1814574 7788696 830571
           67382  202403  830571  180080]
    XYd = [ 2677.67569926499 -2096.61323241936
            17433.7722193412 -8452.80637614894
            82255.6235419561 -33172.6513903503];
    XYc = [150471.005724600
           496537.749200499
           2029884.30853330
           267380.434504700]
    coefc = [.0051, 0, .000015, -.00153]
    coefd = 2e-16*rand(3,2)
    coefd[3,1] = -1
    coefd[3,2] = 1
    
    # Call the function with the test data
    new_coefd = updateReg(XXd, XYd, copy(coefd), "discrete")
    # Check that the output is a vector of the same length as the input coefficients
    @test isequal(size(new_coefd), size(coefd))
    # Check that the output coefficients are different from the input coefficients
    @test !all(new_coefd .!= coefd)
    @test new_coefd[3,1] == -1
    @test new_coefd[3,2] == 1

    # Repeat the test for the "continuous" outcome type
    new_coefc = updateReg(XXc, XYc, copy(coefc), "continuous")
    @test isequal(size(new_coefc), size(coefc))
    @test !all(new_coefc .!= coefc)
    @test new_coefc[2] == 0
    
    # Force unknown type error
    @test_throws ErrorException updateReg(XXc, XYc, copy(coefc), "discontinuous")
end
 
@testset "updateReg function" begin
    # Define test data
    XXd = [100000.0 190379  667307  
           190379.0 667307  2566277 
           667307.0 2566277 10420439]
    XXc = [40670  108078  434544  67382.0
           108078 434544  1814574 202403
           434544 1814574 7788696 830571
           67382  202403  830571  180080]
    XYd = [ 2677.67569926499 -2096.61323241936
            17433.7722193412 -8452.80637614894
            82255.6235419561 -33172.6513903503];
    XYc = [150471.005724600
           496537.749200499
           2029884.30853330
           267380.434504700]
    coefc = 2e-16*rand(4)
    coefd = 2e-16*rand(3,2)
    
    # Call the function with the test data
    new_coefd = updateReg(XXd, XYd, copy(coefd), "discrete")
    # Check that the output is a vector of the same length as the input coefficients
    @test isequal(size(new_coefd), size(coefd))
    # Check that the output coefficients are different from the input coefficients
    @test all(new_coefd .!= coefd)

    # Repeat the test for the "continuous" outcome type
    new_coefc = updateReg(XXc, XYc, copy(coefc), "continuous")
    @test isequal(size(new_coefc), size(coefc))
    @test all(new_coefc .!= coefc)
end
 
@testset "read_data, bootsamp, suffstats, vecparm, MM, estimate_model checks" begin
    DTA, mInfo = read_data("smalldata/")
    est = start_values(DTA, mInfo; modeltype="full")
    outp, Lp, θp = SuffStatsFun(DTA[10], est, 1_000, "mass")
    new_DATA = bootsamp(copy(DTA), 4)
    # read_data
    @testset "read_data" begin
        @test isa(DTA, Vector{<:NamedTuple})
        @test isa(mInfo,NamedTuple)
        @test length(DTA)==2_000
        @test length(mInfo)==2
        @test length(mInfo.name)==20
        @test isequal(length(mInfo.name), length(mInfo.type))
    end
    # bootsamp
    @testset "bootsamp" begin
        @test isa(new_DATA, Vector{<:NamedTuple})
        @test length(new_DATA) < length(DTA)
        @test sum([d.pwt for d in new_DATA]) == sum([d.pwt for d in DTA])
    end
    # starting values
    @testset "starting values" begin
        @test isa(est, NamedTuple)
    end
    # SuffStatsFun
    @testset "SuffStatsFun" begin
        @test isa(outp, NamedTuple)
        @test isequal(size(Lp), (1_000,1))
        @test isequal(size(θp), (1_000,2))
    end
    # vecparm
    @testset "vecparm" begin
        @test isa(vecparm(est), Function)
    end 
    # MM
    @testset "MM" begin
        @test isa(MM(DTA, 1_000, est; maxIter=2, printIter=5), NamedTuple)
    end
    # estimate_model
    @testset "estimate_model" begin
        @test isnothing(estimate_model(DTA, est, nothing, 0, 2_500; maxIter=2))
    end
    # permutations of sampling type
    @testset "independent sampling" begin
        # right now this won't work because there is mixing of the unobserved types (i.e. type depends on X's)
        @test_throws MethodError SuffStatsFun(DTA[10], est, 1_000, "independent")
    end
    # permutations of suffstatsfun
    @testset "model type permutations in suffstats" begin
        est = start_values(DTA, mInfo; modeltype="meas only")
        outp, Lp, θp = SuffStatsFun(DTA[10], est, 100, "mass")
        @test isequal(size(θp), (100, 2))
        est = start_values(DTA, mInfo; modeltype="meas and choice")
        outp, Lp, θp = SuffStatsFun(DTA[10], est, 100, "mass")
        @test isequal(size(θp), (100, 2))
        est = start_values(DTA, mInfo; modeltype="wage only")
        @test_throws ErrorException SuffStatsFun(DTA[10], est, 100, "mass")
    end
end

@testset "Igrp function" begin
    X = [1]
    g = [1; 2; 3]
    @test Igrp(X, g) == [1 0 0; 1 1 0; 1 0 1]
    X = [1]
    g = [1; 3; 3; 2; 2; 1]
    @test Igrp(X, g) == [1 0 0; 1 0 1; 1 0 1; 1 1 0; 1 1 0; 1 0 0] 
end

@testset "choiceX function" begin
    D = (T=2, sch=[1, 2], pvsch=[3, 4], exper=[5, 6], pvwrk=[7, 8])
    expected_output = [1 1 1 3 5 25 7; 1 2 4 4 6 36 8]
    @test choiceX(D) == expected_output
end

@testset "lnwageX function" begin
    D = (T=2, sch=[1, 2], exper=[3, 4])
    expected_output = [1 1 1 3 9 3; 1 2 4 4 16 8]
    @test lnwageX(D) == expected_output
end

@testset "typeX function" begin
    D = (grp = 1,)
    @test typeX(D) == [1 0 0]
    D = (grp = 2,)
    @test typeX(D) == [1 1 0]
    D = (grp = 3,)
    @test typeX(D) == [1 0 1]
    D = (grp = [1; 3; 3; 2; 2; 1],)
    @test typeX(D) == [1 0 0; 1 0 1; 1 0 1; 1 1 0; 1 1 0; 1 0 0] 
end