# ------------------------------------------------------------------
# Licensed under the MIT License. See LICENSE in the project root.
# ------------------------------------------------------------------

module GeoStatsValidation

using LossFunctions
using DataScienceTraits
using DensityRatioEstimation
using StatsLearnModels

using Meshes
using GeoTables
using GeoStatsBase
using GeoStatsTransforms

using GeoStatsModels: GeoStatsModel

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
