# test/plots_test.jl
ENV["GKSwstype"] = get(ENV, "GKSwstype", "100")   # headless GR backend (must precede `using Plots`)
using Plots

@testset "Plots extension: plot_mesh_normals" begin
  @test Base.get_extension(RadiativeViewFactor,
                           :RadiativeViewFactorPlotsExt) !== nothing

  MeshData = RadiativeViewFactor.MeshIO.MeshData
  no_soups = Dict{Int,Array{Float64,3}}()

  # One-element mesh per family, in the xy-plane (unit square / triangle / segment)
  square4 = [0 0 0; 1 0 0; 1 1 0; 0 1 0]'
  square8 = [0 0 0; 1 0 0; 1 1 0; 0 1 0; 0.5 0 0; 1 0.5 0; 0.5 1 0; 0 0.5 0]'
  tri3    = [0 0 0; 1 0 0; 0 1 0]'
  tri6    = [0 0 0; 1 0 0; 0 1 0; 0.5 0 0; 0.5 0.5 0; 0 0.5 0]'
  seg2    = [0 0 0; 1 0 0]'
  seg3    = [0 0 0; 1 0 0; 0.5 0.1 0]'
  cases = [
    (:quad4, square4, 2), (:quad, square8, 2),
    (:tri3, tri3, 2),     (:tri, tri6, 2),
    (:line2, seg2, 1),    (:line3, seg3, 1),
  ]

  for (fam, coords, dim) in cases
    nn   = size(coords, 2)
    mesh = MeshData(Matrix{Float64}(coords),
                    [SurfaceElement(collect(1:nn), 1, fam)],
                    Dict(1 => "wall"), Dict(1 => [1]), no_soups, dim)
    for kw in (NamedTuple(), (show_nodes=true, show_indices=true))
      fig = plot_mesh_normals(mesh; kw...)
      @test fig isa Plots.Plot
      @test length(fig.series_list) >= 2   # element outline + normal arrow
    end
  end

  # Multi-group mesh: legend entries per group, kwargs honoured
  coords = Float64[0 0 0; 1 0 0; 2 0 0]'
  mesh = MeshData(coords,
                  [SurfaceElement([1, 2], 1, :line2), SurfaceElement([2, 3], 2, :line2)],
                  Dict(1 => "left", 2 => "right"),
                  Dict(1 => [1], 2 => [2]), no_soups, 1)
  fig = plot_mesh_normals(mesh; normal_scale=0.1, group_colors=Dict(1 => :red))
  labels = [string(s[:label]) for s in fig.series_list]
  @test any(occursin("left (tag 1)", l) for l in labels)
  @test any(occursin("right (tag 2)", l) for l in labels)
end
