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
using StatsLearnModels: Learn
using GeoStatsTransforms: Interpolate
using GeoStatsTransforms: InterpolateNeighbors

using ColumnSelectors: selector
using GeoStatsBase: weight, folds, mean
using GeoStatsBase: WeightingMethod, DensityRatioWeighting, UniformWeighting
using GeoStatsBase: FoldingMethod, BallFolding, BlockFolding, OneFolding, UniformFolding
using LossFunctions: L2DistLoss, MisclassLoss

include("utils.jl")
include("cverror.jl")

export
  # estimators
  LeaveOneOut,
  LeaveBallOut,
  KFoldValidation,
  BlockValidation,
  DensityRatioValidation,
  WeightedValidation,

  # main function
  cverror

end
