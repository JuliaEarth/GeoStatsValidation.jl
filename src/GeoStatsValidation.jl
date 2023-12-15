module GeoStatsValidation

using Meshes
using GeoTables
using Transducers
using DensityRatioEstimation

using GeoStatsModels: GeoStatsModel
using StatsLearnModels: Learn, StatsLearnModel, input, output
using GeoStatsTransforms: Interpolate, InterpolateNeighbors

using ColumnSelectors: selector
using GeoStatsBase: WeightingMethod, DensityRatioWeighting, UniformWeighting
using GeoStatsBase: FoldingMethod, BallFolding, BlockFolding, OneFolding, UniformFolding
using GeoStatsBase: weight, folds, defaultloss, mean
using LossFunctions.Traits: SupervisedLoss

include("cverror.jl")

export cverror, LeaveOneOut, LeaveBallOut, KFoldValidation, BlockValidation, WeightedValidation, DensityRatioValidation

end
