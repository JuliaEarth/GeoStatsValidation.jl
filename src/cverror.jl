# ------------------------------------------------------------------
# Licensed under the MIT License. See LICENSE in the project root.
# ------------------------------------------------------------------

"""
    ErrorMethod

A method for estimating cross-validatory error.
"""
abstract type ErrorMethod end

abstract type ErrorSetup end

struct LearnSetup{M} <: ErrorSetup
  model::M
  input::Vector{Symbol}
  output::Vector{Symbol}
end

struct InterpSetup{I,M,K} <: ErrorSetup
  model::M
  kwargs::K
end

"""
    cverror(model::GeoStatsModel, geotable, method; kwargs...)

Estimate error of `model` in a given `geotable` with
error estimation `method` using `Interpolate` or
`InterpolateNeighbors` depending on the passed
`kwargs`.

    cverror(model::StatsLearnModel, geotable, method)
    cverror((model, invars => outvars), geotable, method)

Estimate error of `model` in a given `geotable` with
error estimation `method` using the `Learn` transform.
"""
function cverror end

cverror((model, cols)::Tuple{Any,Pair}, geotable::AbstractGeoTable, method::ErrorMethod) =
  cverror(StatsLearnModel(model, first(cols), last(cols)), geotable, method)

function cverror(model::StatsLearnModel, geotable::AbstractGeoTable, method::ErrorMethod)
  names = setdiff(propertynames(geotable), [:geometry])
  invars = input(model)(names)
  outvars = output(model)(names)
  setup = LearnSetup(model, invars, outvars)
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
include("cverrors/wcv.jl")
include("cverrors/drv.jl")
