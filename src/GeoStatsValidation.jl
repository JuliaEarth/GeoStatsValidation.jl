# ------------------------------------------------------------------
# Licensed under the MIT License. See LICENSE in the project root.
# ------------------------------------------------------------------

module GeoStatsValidation

using Meshes
using GeoTables
using DataScienceTraits
using DensityRatioEstimation

using GeoStatsModels: GeoStatsModel
using StatsLearnModels: StatsLearnModel
using StatsLearnModels: Learn, input, output
using GeoStatsTransforms: Interpolate, InterpolateNeighbors

using ColumnSelectors: selector
using GeoStatsBase: weight, folds, mean
using GeoStatsBase: WeightingMethod, DensityRatioWeighting, UniformWeighting
using GeoStatsBase: FoldingMethod, BallFolding, BlockFolding, OneFolding, UniformFolding
using LossFunctions: L2DistLoss, MisclassLoss
using LossFunctions.Traits: SupervisedLoss

include("utils.jl")
include("cverror.jl")

export cverror, LeaveOneOut, LeaveBallOut, KFoldValidation, BlockValidation, WeightedValidation, DensityRatioValidation

end
