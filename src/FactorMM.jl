module FactorMM

using Distributions, Random, StatsBase, Statistics, NaNStatistics, LinearAlgebra, DataFrames, CSV, MAT

include("allfuns.jl")

export read_data,
       Igrp,
       choiceX,
       lnwageX,
       typeX,
       start_values,
       updateReg!,
       SuffStatsFun,
       em_alg,
       vecparm,
       MM,
       bootsamp,
       estimate_model

end
