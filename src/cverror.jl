# ------------------------------------------------------------------
# Licensed under the MIT License. See LICENSE in the project root.
# ------------------------------------------------------------------

"""
    ErrorMethod

A method for estimating cross-validation error.
"""
abstract type ErrorMethod end

"""
    cverror(model, geotable, method)

Estimate cross-validation error of (geo)statistical `model`
on given `geotable` with error estimation `method`.
"""
function cverror end

# ----------------
# IMPLEMENTATIONS
# ----------------

include("cverror/loo.jl")
include("cverror/lbo.jl")
include("cverror/kfv.jl")
include("cverror/bcv.jl")
include("cverror/drv.jl")
include("cverror/wcv.jl")
