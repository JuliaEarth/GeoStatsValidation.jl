# ------------------------------------------------------------------
# Licensed under the MIT License. See LICENSE in the project root.
# ------------------------------------------------------------------

defaultloss(val) = defaultloss(scitype(val))
defaultloss(::Type{Continuous}) = L2DistLoss()
defaultloss(::Type{Categorical}) = MisclassLoss()

assymbol(obj) = Dict(Symbol.(keys(obj)) .=> values(obj))
