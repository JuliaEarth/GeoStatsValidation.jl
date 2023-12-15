# ------------------------------------------------------------------
# Licensed under the MIT License. See LICENSE in the project root.
# ------------------------------------------------------------------

"""
    ErrorMethod

A method for estimating cross-validatory error.
"""
abstract type ErrorMethod end

struct LearnSetup{M}
  model::M
  input::Vector{Symbol}
  output::Vector{Symbol}
end

struct InterpSetup{I,M}
  model::M
end

"""
    cverror(Lean, model, incols => outcols, geotable, method)
    cverror(Interpolate, model, geotable, method)
    cverror(InterpolateNeighbors, model, geotable, method)

Estimate error of `model` in a given `geotable` with
error estimation `method`.
"""
function cverror end

function cverror(::Type{Learn}, model, (incols, outcols)::Pair, geotable::AbstractGeoTable, method::ErrorMethod)
  names = setdiff(propertynames(geotable), [:geometry])
  input = selector(incols)(names)
  output = selector(outcols)(names)
  cverror(LearnSetup(model, input, output), geotable, method)
end

const Interp = Union{Interpolate,InterpolateNeighbors}

cverror(::Type{I}, model::M, geotable::AbstractGeoTable, method::ErrorMethod) where {I<:Interp,M} =
  cverror(InterpSetup{I,M}(model), geotable, method)

# ----------------
# IMPLEMENTATIONS
# ----------------

include("cverrors/loo.jl")
include("cverrors/lbo.jl")
include("cverrors/kfv.jl")
include("cverrors/bcv.jl")
include("cverrors/wcv.jl")
include("cverrors/drv.jl")
