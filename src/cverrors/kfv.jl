# ------------------------------------------------------------------
# Licensed under the MIT License. See LICENSE in the project root.
# ------------------------------------------------------------------

"""
    KFoldValidation(k; shuffle=true, loss=Dict())

`k`-fold cross-validation. Optionally, `shuffle` the
data, and specify a dictionary with `loss` functions
from `LossFunctions.jl` for some of the variables.

## References

* Geisser, S. 1975. [The predictive sample reuse method with applications]
  (https://www.jstor.org/stable/2285815)
* Burman, P. 1989. [A comparative study of ordinary cross-validation, v-fold
  cross-validation and the repeated learning-testing methods]
  (https://www.jstor.org/stable/2336116)
"""
struct KFoldValidation{L} <: ErrorMethod
  k::Int
  shuffle::Bool
  loss::L
end

KFoldValidation(k::Int; shuffle=true, loss=Dict()) = KFoldValidation(k, shuffle, assymbol(loss))

function cverror(setup::ErrorSetup, geotable::AbstractGeoTable, method::KFoldValidation)
  # uniform weights
  weighting = UniformWeighting()

  # random folds
  folding = UniformFolding(method.k, method.shuffle)

  wcv = WeightedValidation(weighting, folding, lambda=1, loss=method.loss)

  cverror(setup, geotable, wcv)
end
