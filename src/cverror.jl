# ------------------------------------------------------------------
# Licensed under the MIT License. See LICENSE in the project root.
# ------------------------------------------------------------------

"""
    ErrorMethod

A method for estimating cross-validation error.
"""
abstract type ErrorMethod end

abstract type ErrorSetup end

struct LearnSetup{M,P,T} <: ErrorSetup
  model::M
  preds::P
  targs::T
end

struct InterpSetup{I,M,K} <: ErrorSetup
  model::M
  kwargs::K
end

"""
    cverror(model::GeoStatsModel, geotable, method; kwargs...)

Estimate cross-validation error of geostatistical `model`
on given `geotable` with error estimation `method` using
`Interpolate` or `InterpolateNeighbors` depending on `kwargs`.

    cverror(model, geotable, method)

Estimate cross-validation error of statistical learning `model`
on given `geotable` with error estimation `method`.
"""
function cverror end

function cverror(model, geotable::AbstractGeoTable, method::ErrorMethod)
  table = values(geotable)
  preds = predictors(table)
  targs = targets(table)
  setup = LearnSetup(model, preds, targs)
  cverror(setup, geotable, method)
end

const INTERPNEIGHBORS = (:minneighbors, :maxneighbors, :neighborhood, :distance)

function cverror(model::M, geotable::AbstractGeoTable, method::ErrorMethod; kwargs...) where {M<:GeoStatsModel}
  I = any(∈(INTERPNEIGHBORS), keys(kwargs)) ? InterpolateNeighbors : Interpolate
  setup = InterpSetup{I,M,typeof(kwargs)}(model, kwargs)
  cverror(setup, geotable, method)
end

# ----------------
# IMPLEMENTATIONS
# ----------------

include("cverrors/loo.jl")
include("cverrors/lbo.jl")
include("cverrors/kfv.jl")
include("cverrors/bcv.jl")
include("cverrors/drv.jl")
include("cverrors/wcv.jl")
