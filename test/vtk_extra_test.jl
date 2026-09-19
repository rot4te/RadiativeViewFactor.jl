# test/vtk_extra_test.jl
#
# Branches of the ReadVTK extension that vtk_test.jl's single unit-cube fixture
# does not reach: verbose reporting, named/missing group arrays, offsets that
# carry a leading zero, cells of unsupported types, curve meshes, and malformed
# cells. Uses its own general-purpose .vtu writer (inline-binary arrays, which
# is what ReadVTK reads).

using ReadVTK
using Base64

function write_vtu(path::String; points, conn, offs, types, celldata=Pair{String,Any}[])
  darray(name, v; ncomp=1) = string(
    "<DataArray type=\"", eltype(v), "\" Name=\"", name, "\"",
    ncomp > 1 ? " NumberOfComponents=\"$ncomp\"" : "",
    " format=\"binary\">",
    let io = IOBuffer()
      write(io, UInt64(sizeof(v)))
      write(io, v)
      base64encode(take!(io))
    end,
    "</DataArray>")
  npts = size(points, 2)
  cd = isempty(celldata) ? "" :
    "<CellData>" * join(darray(k, v) for (k, v) in celldata) * "</CellData>"
  write(path, """<?xml version="1.0"?>
    <VTKFile type="UnstructuredGrid" version="1.0" byte_order="LittleEndian" header_type="UInt64">
    <UnstructuredGrid>
    <Piece NumberOfPoints="$npts" NumberOfCells="$(length(types))">
    <Points>$(darray("Points", Float64.(points); ncomp=3))</Points>
    <Cells>$(darray("connectivity", Int64.(conn)))$(darray("offsets", Int64.(offs)))$(darray("types", UInt8.(types)))</Cells>
    $cd
    </Piece>
    </UnstructuredGrid>
    </VTKFile>
    """)
  return path
end

const CUBE_PTS  = Float64[0 1 1 0 0 1 1 0;
                          0 0 1 1 0 0 1 1;
                          0 0 0 0 1 1 1 1]
const CUBE_CONN = Int64[0,1,2,3, 4,7,6,5, 0,4,5,1, 1,5,6,2, 2,6,7,3, 3,7,4,0]   # inward
const CUBE_OFFS = Int64[4, 8, 12, 16, 20, 24]

@testset "ReadVTK extension: further branches" begin

  @testset "verbose report, named group array, skipped cells, leading-zero offsets" begin
    # six quads plus one VTK_VERTEX (type 1), which has no supported family
    conn  = vcat(CUBE_CONN, Int64[0])
    types = vcat(fill(9, 6), 1)
    ids   = Int32[1, 1, 2, 2, 2, 2, 3]
    f = write_vtu(tempname() * ".vtu"; points=CUBE_PTS, conn=conn,
                  offs=vcat(CUBE_OFFS, 25), types=types,
                  celldata=["Region" => ids])

    # explicit group_field naming the array; offsets do not start with 0
    m, out = capture_stdout() do
      load_vtu(f; group_field="Region", verbose=true)
    end
    @test length(m.surface_elems) == 6
    @test sort(collect(keys(m.group_elems))) == [1, 2]
    @test occursin("Using per-cell group array \"Region\"", out)
    @test occursin("skipped 1 cell(s) of other types", out)
    rm(f)

    # offsets with a leading zero (length ncells + 1) address the same cells
    f0 = write_vtu(tempname() * ".vtu"; points=CUBE_PTS, conn=CUBE_CONN,
                   offs=vcat(0, CUBE_OFFS), types=fill(9, 6))
    m0 = load_vtu(f0; verbose=false)
    @test [e.nodes for e in m0.surface_elems] ==
          [CUBE_CONN[(4k-3):(4k)] .+ 1 for k in 1:6]
    rm(f0)
  end

  @testset "group_field that is not a per-cell array falls back with a warning" begin
    f = write_vtu(tempname() * ".vtu"; points=CUBE_PTS, conn=CUBE_CONN,
                  offs=CUBE_OFFS, types=fill(9, 6), celldata=["Region" => Int32[1, 1, 2, 2, 2, 2]])
    m = @test_logs (:warn, r"Requested group_field \"nope\" not found") load_vtu(f; group_field="nope", verbose=false)
    @test m.group_tags == Dict(1 => "default")
    rm(f)
  end

  @testset "curve meshes: Line and QuadraticEdge, winding warning, reverse_normals" begin
    # the unit square's boundary as 4 VTK_LINE cells, then as 2 VTK_QUADRATIC_EDGE
    pts   = Float64[0 1 1 0 0.5 1 0.5 0;
                    0 0 1 1 0   0.5 1 0.5;
                    0 0 0 0 0   0   0 0]
    lines = write_vtu(tempname() * ".vtu"; points=pts,
                      conn=[0,1, 1,2, 2,3, 3,0], offs=[2, 4, 6, 8], types=fill(3, 4))
    m = @test_logs (:warn, r"VTK curve meshes are loaded without Gmsh-based normal") load_vtu(lines; surface_dim=1, verbose=false)
    @test all(e -> e.family === :line2, m.surface_elems)
    @test size(m.group_tri_soup[1]) == (3, 2, 4)
    mr = @test_logs (:warn, r"VTK curve meshes") load_vtu(lines; surface_dim=1, reverse_normals=true, verbose=false)
    @test [e.nodes for e in mr.surface_elems] == [[2, 1], [3, 2], [4, 3], [1, 4]]
    rm(lines)

    quad_edges = write_vtu(tempname() * ".vtu"; points=pts,
                           conn=[0,1,4, 1,2,5], offs=[3, 6], types=fill(21, 2))
    mq = @test_logs (:warn, r"VTK curve meshes") load_vtu(quad_edges; surface_dim=1, verbose=false)
    @test all(e -> e.family === :line3 && length(e.nodes) == 3, mq.surface_elems)
    rm(quad_edges)
  end

  @testset "a cell with the wrong number of nodes is an error" begin
    f = write_vtu(tempname() * ".vtu"; points=CUBE_PTS, conn=Int64[0, 1, 2],
                  offs=Int64[3], types=[9])            # a VTK_QUAD needs 4 nodes
    @test_throws ErrorException load_vtu(f; verbose=false)
    rm(f)
  end

  @testset "load_vtu without ReadVTK loaded explains what to do" begin
    # ReadVTK is loaded in this process, so ask a fresh one (same environment,
    # same coverage flags) that never loads it.
    code = "using RadiativeViewFactor; load_vtu(ARGS[1])"
    errfile = tempname()
    cmd = `$(Base.julia_cmd()) --project=$(Base.active_project()) -e $code nothing.vtu`
    proc = run(pipeline(ignorestatus(cmd); stderr=errfile))
    msg = read(errfile, String)
    rm(errfile)
    @test !success(proc)
    @test occursin("requires ReadVTK.jl", msg)
  end
end
