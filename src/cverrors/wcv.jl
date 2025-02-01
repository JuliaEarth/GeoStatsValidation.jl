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

function cverror(setup::ErrorSetup, geotable::AbstractGeoTable, method::WeightedValidation)
  ovars = _outputs(setup, geotable)
  loss = method.loss
  for var in ovars
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
    pred = _prediction(setup, geotable, f)

    # holdout set
    holdout = view(geotable, f[2])

    # holdout weights
    𝓌 = view(ws, f[2])

    # loss for each variable
    losses = map(ovars) do var
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
  Dict(var => mean(get.(εs, var, 0)) for var in ovars)
end

# output variables
_outputs(::InterpSetup, gtb) = setdiff(propertynames(gtb), [:geometry])
_outputs(s::LearnSetup, gtb) = s.targs

# prediction for a given fold
function _prediction(s::InterpSetup{I}, geotable, f) where {I}
  sdat = view(geotable, f[1])
  sdom = view(domain(geotable), f[2])
  sdat |> I(sdom; model=s.model, s.kwargs...)
end

function _prediction(s::LearnSetup, geotable, f)
  source = view(geotable, f[1])
  target = view(geotable, f[2])
  target |> Learn(source, s.model)
end
