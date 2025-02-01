# ------------------------------------------------------------------
# Licensed under the MIT License. See LICENSE in the project root.
# ------------------------------------------------------------------

"""
    LeaveOneOut(; loss=Dict())

Leave-one-out validation. Optionally, specify a dictionary of
`loss` functions from `LossFunctions.jl` for some of the variables.

## References

* Stone. 1974. [Cross-Validatory Choice and Assessment of Statistical Predictions]
  (https://rss.onlinelibrary.wiley.com/doi/abs/10.1111/j.2517-6161.1974.tb00994.x)
"""
struct LeaveOneOut{L} <: ErrorMethod
  loss::L
end

LeaveOneOut(; loss=Dict()) = LeaveOneOut(assymbol(loss))

function cverror(setup::ErrorSetup, geotable::AbstractGeoTable, method::LeaveOneOut)
  # uniform weights
  weighting = UniformWeighting()

  # point folds
  folding = OneFolding()

  wcv = WeightedValidation(weighting, folding, lambda=1, loss=method.loss)

  cverror(setup, geotable, wcv)
end
