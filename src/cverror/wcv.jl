# ------------------------------------------------------------------
# Licensed under the MIT License. See LICENSE in the project root.
# ------------------------------------------------------------------

"""
    WeightedValidation(weighting, folding; lambda=1.0, loss=Dict())

An error estimation method which samples are weighted with
`weighting` method and split into folds with `folding` method.
Weights are raised to `lambda` power in `[0,1]`. Optionally,
specify a dictionary with `loss` functions from `LossFunctions.jl`
for some of the variables.

## References

* Sugiyama et al. 2006. [Importance-weighted cross-validation for
  covariate shift](https://link.springer.com/chapter/10.1007/11861898_36)
* Sugiyama et al. 2007. [Covariate shift adaptation by importance weighted
  cross validation](http://www.jmlr.org/papers/volume8/sugiyama07a/sugiyama07a.pdf)
"""
struct WeightedValidation{W<:WeightingMethod,F<:FoldingMethod,T,L} <: ErrorMethod
  weighting::W
  folding::F
  lambda::T
  loss::L
end

function WeightedValidation(weighting, folding; lambda=1.0, loss=Dict())
  @assert 0 ≤ lambda ≤ 1 "lambda must lie in [0,1]"
  WeightedValidation(weighting, folding, lambda, loss)
end

function cverror(model, geotable::AbstractGeoTable, method::WeightedValidation)
  vars = _outputs(model, geotable)
  loss = method.loss
  for var in vars
    if var ∉ keys(loss)
      v = getproperty(geotable, var)
      loss[var] = defaultloss(v[1])
    end
  end

  # weight all samples
  ws = weight(geotable, method.weighting) .^ method.lambda

  # folds for cross-validation
  fs = folds(geotable, method.folding)

  # error for a fold
  function ε(f)
    # fold prediction
    pred = _prediction(model, geotable, f)

    # holdout set
    holdout = view(geotable, f[2])

    # holdout weights
    𝓌 = view(ws, f[2])

    # loss for each variable
    losses = map(vars) do var
      ℒ = loss[var]
      ŷ = getproperty(pred, var)
      y = getproperty(holdout, var)
      var => mean(ℒ, ŷ, y, 𝓌, normalize=false)
    end

    Dict(losses)
  end

  # compute error for each fold
  εs = mapreduce(ε, vcat, fs)

  # combine error from different folds
  Dict(var => mean(get.(εs, var, 0)) for var in vars)
end

# output variables
_outputs(::Any, geotable) = targets(values(geotable))
_outputs(::GeoStatsModel, geotable) = setdiff(propertynames(geotable), [:geometry])

# prediction for a given fold
function _prediction(model, geotable, f)
  targs = targets(values(geotable))
  sdata = view(geotable, f[1])
  tdata = view(geotable, f[2])
  tdata |> Learn(label(sdata, targs); model)
end
function _prediction(model::GeoStatsModel, geotable, f)
  sdata = view(geotable, f[1])
  tvdom = view(domain(geotable), f[2])
  sdata |> Interpolate(tvdom; model)
end
