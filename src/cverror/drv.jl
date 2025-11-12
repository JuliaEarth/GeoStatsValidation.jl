# ------------------------------------------------------------------
# Licensed under the MIT License. See LICENSE in the project root.
# ------------------------------------------------------------------

"""
    DensityRatioValidation(k; [options])

Density ratio validation where weights are first obtained with density
ratio estimation, and then used in `k`-fold weighted cross-validation.

## Options

* `shuffle`   - Shuffle the data before folding (default to `true`)
* `estimator` - Density ratio estimator (default to `LSIF()`)
* `optlib`    - Optimization library (default to `default_optlib(estimator)`)
* `lambda`    - Power of density ratios (default to `1.0`)
* `loss`      - Dictionary with loss functions (default to `Dict()`)

Please see [DensityRatioEstimation.jl]
(https://github.com/JuliaEarth/DensityRatioEstimation.jl)
for a list of supported estimators.

## References

* Hoffimann et al. 2020. [Geostatistical Learning: Challenges and Opportunities]
  (https://arxiv.org/abs/2102.08791)
"""
struct DensityRatioValidation{T,E,O,L} <: ErrorMethod
  k::Int
  shuffle::Bool
  lambda::T
  dre::E
  optlib::O
  loss::L
end

function DensityRatioValidation(
  k::Int;
  shuffle=true,
  lambda=1.0,
  estimator=LSIF(),
  optlib=default_optlib(estimator),
  loss=Dict()
)
  @assert k > 0 "number of folds must be positive"
  @assert 0 ≤ lambda ≤ 1 "lambda must lie in [0,1]"
  DensityRatioValidation(k, shuffle, lambda, estimator, optlib, assymbol(loss))
end

function cverror(setup::LearnSetup, geotable::AbstractGeoTable, method::DensityRatioValidation)
  vars = setup.preds

  # density-ratio weights
  weighting = DensityRatioWeighting(geotable, vars, estimator=method.dre, optlib=method.optlib)

  # random folds
  folding = UniformFolding(method.k, method.shuffle)

  wcv = WeightedValidation(weighting, folding, lambda=method.lambda, loss=method.loss)

  cverror(setup, geotable, wcv)
end
