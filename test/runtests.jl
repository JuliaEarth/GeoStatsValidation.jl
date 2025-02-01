using GeoStatsValidation
using StatsLearnModels
using GeoStatsModels
using LossFunctions
using GeoTables
using Meshes
using Random
using Test

@testset "GeoStatsValidation.jl" begin
  Random.seed!(123)

  @testset "Learning" begin
    x = rand(1:2, 1000)
    y = rand(1:2, 1000)
    gtb = georef((; x, y), rand(Point, 1000))
    model = DecisionTreeClassifier()

    # dummy classifier → 0.5 misclassification rate
    for method in [
      # methods
      LeaveOneOut(),
      LeaveBallOut(0.1),
      KFoldValidation(10),
      BlockValidation(0.1),
      DensityRatioValidation(10)
    ]
      e = cverror((model, :x => :y), gtb, method)
      @test isapprox(e[:y], 0.5, atol=0.06)
    end

    # test with custom loss
    loss = Dict("y" => MisclassLoss())
    for method in [
      # methods
      LeaveOneOut(loss=loss),
      LeaveBallOut(0.1, loss=loss),
      KFoldValidation(10, loss=loss),
      BlockValidation(0.1, loss=loss),
      DensityRatioValidation(10, loss=loss)
    ]
      e = cverror((model, :x => :y), gtb, method)
      @test isapprox(e[:y], 0.5, atol=0.06)
    end
  end

  @testset "Interpolation" begin
    gtb₁ = georef((z=rand(50, 50),))
    gtb₂ = georef((z=100rand(50, 50),))
    sgtb₁ = sample(gtb₁, UniformSampling(100, replace=false))
    sgtb₂ = sample(gtb₂, UniformSampling(100, replace=false))
    model = NN()

    # low variance + dummy (mean) estimator → low error
    # high variance + dummy (mean) estimator → high error
    for method in [LeaveOneOut(), LeaveBallOut(0.1), KFoldValidation(10), BlockValidation(0.1)]
      e₁ = cverror(model, sgtb₁, method)
      e₂ = cverror(model, sgtb₂, method)
      @test e₁[:z] < 1
      @test e₂[:z] > 1
    end
  end
end
