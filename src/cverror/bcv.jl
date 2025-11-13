# ------------------------------------------------------------------
# Licensed under the MIT License. See LICENSE in the project root.
# ------------------------------------------------------------------

"""
    BlockValidation(sides; loss=Dict())

Cross-validation with blocks of given `sides`. Optionally,
specify a dictionary with `loss` functions from `LossFunctions.jl`
for some of the variables.

## References

* Roberts et al. 2017. [Cross-validation strategies for data with
  temporal, spatial, hierarchical, or phylogenetic structure]
  (https://onlinelibrary.wiley.com/doi/10.1111/ecog.02881)
* Pohjankukka et al. 2017. [Estimating the prediction performance
  of spatial models via spatial k-fold cross-validation]
  (https://www.tandfonline.com/doi/full/10.1080/13658816.2017.1346255)
"""
struct BlockValidation{S,L} <: ErrorMethod
  sides::S
  loss::L
end

BlockValidation(sides::Tuple; loss=Dict()) = BlockValidation(sides, assymbol(loss))

BlockValidation(sides::Number...; kwargs...) = BlockValidation(sides; kwargs...)

function cverror(model, geotable::AbstractGeoTable, method::BlockValidation)
  wmethod = UniformWeighting()
  fmethod = BlockFolding(method.sides)
  emethod = WeightedValidation(wmethod, fmethod, lambda=1, loss=method.loss)
  cverror(model, geotable, emethod)
end
