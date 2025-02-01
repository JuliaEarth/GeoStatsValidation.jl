# ------------------------------------------------------------------
# Licensed under the MIT License. See LICENSE in the project root.
# ------------------------------------------------------------------

"""
    ErrorMethod

A method for estimating cross-validation error.
"""
abstract type ErrorMethod end

abstract type ErrorSetup end

struct LearnSetup{M,F,T} <: ErrorSetup
  model::M
  feats::F
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

    cverror(model::StatsLearnModel, geotable, method)
    cverror((model, feats => targs), geotable, method)

Estimate cross-validation error of statistical learning `model`
on given `geotable` with error estimation `method`.
"""
function cverror end

cverror((model, cols)::Tuple{Any,Pair}, geotable::AbstractGeoTable, method::ErrorMethod) =
  cverror(StatsLearnModel(model, first(cols), last(cols)), geotable, method)

function cverror(model::StatsLearnModel, geotable::AbstractGeoTable, method::ErrorMethod)
  names = setdiff(propertynames(geotable), [:geometry])
  feats = model.feats(names)
  targs = model.targs(names)
  setup = LearnSetup(model, feats, targs)
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
