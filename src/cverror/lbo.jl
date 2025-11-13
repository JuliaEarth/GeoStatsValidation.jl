# ------------------------------------------------------------------
# Licensed under the MIT License. See LICENSE in the project root.
# ------------------------------------------------------------------

"""
    LeaveBallOut(ball; loss=Dict())

Leave-`ball`-out (a.k.a. spatial leave-one-out) validation.
Optionally, specify a dictionary with `loss` functions from
`LossFunctions.jl` for some of the variables.

    LeaveBallOut(radius; loss=Dict())

By default, use Euclidean ball of given `radius` in space.

## References

* Le Rest et al. 2014. [Spatial leave-one-out cross-validation
  for variable selection in the presence of spatial autocorrelation]
  (https://onlinelibrary.wiley.com/doi/full/10.1111/geb.12161)
"""
struct LeaveBallOut{B,L} <: ErrorMethod
  ball::B
  loss::L
end

LeaveBallOut(ball; loss=Dict()) = LeaveBallOut(ball, assymbol(loss))

LeaveBallOut(radius::Number; loss=Dict()) = LeaveBallOut(MetricBall(radius); loss=loss)

function cverror(model, geotable::AbstractGeoTable, method::LeaveBallOut)
  wmethod = UniformWeighting()
  fmethod = BallFolding(method.ball)
  emethod = WeightedValidation(wmethod, fmethod, lambda=1, loss=method.loss)
  cverror(model, geotable, emethod)
end
