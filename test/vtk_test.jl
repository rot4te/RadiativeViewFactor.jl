# test/vtk_test.jl
using ReadVTK
using Base64

@testset "VTK routing (sniffer)" begin
  _is_xml_vtk = RadiativeViewFactor.MeshIO._is_xml_vtk

  # XML .vtu → detected as XML VTK
  f_vtu = tempname() * ".vtu"
  write(f_vtu, "<?xml version=\"1.0\"?>\n<VTKFile type=\"UnstructuredGrid\">")
  @test _is_xml_vtk(f_vtu)

  # XML-form .vtk → detected via header sniff
  f_xvtk = tempname() * ".vtk"
  write(f_xvtk, "<?xml version=\"1.0\"?>\n<VTKFile type=\"UnstructuredGrid\">")
  @test _is_xml_vtk(f_xvtk)

  # Legacy .vtk → NOT XML (routed to Gmsh instead)
  f_legacy = tempname() * ".vtk"
  write(f_legacy, "# vtk DataFile Version 3.0\nmesh\nASCII\n")
  @test !_is_xml_vtk(f_legacy)

  # Non-VTK extension → false
  f_msh = tempname() * ".msh"
  write(f_msh, "\$MeshFormat\n")
  @test !_is_xml_vtk(f_msh)

  foreach(rm, (f_vtu, f_xvtk, f_legacy, f_msh))
end

# ReadVTK reads only "binary"/"appended" data arrays, not "ascii". This writes
# an uncompressed inline-binary .vtu: each array is base64(UInt64 byte count
# ++ raw data).
function write_cube_vtu(path::String; cell_ids=nothing)
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

  points = Float64[0 0 0; 1 0 0; 1 1 0; 0 1 0; 0 0 1; 1 0 1; 1 1 1; 0 1 1]'
  conn = Int64[0, 1, 2, 3,  4, 7, 6, 5,  0, 4, 5, 1,
               1, 5, 6, 2,  2, 6, 7, 3,  3, 7, 4, 0]   # inward normals (enclosure)
  offs = Int64[4, 8, 12, 16, 20, 24]
  typs = fill(UInt8(9), 6)                                # VTK_QUAD
  celldata = cell_ids === nothing ? "" :
    "<CellData>" * darray("CellEntityIds", Int32.(cell_ids)) * "</CellData>"

  write(path, """<?xml version="1.0"?>
    <VTKFile type="UnstructuredGrid" version="1.0" byte_order="LittleEndian" header_type="UInt64">
    <UnstructuredGrid>
    <Piece NumberOfPoints="8" NumberOfCells="6">
    <Points>$(darray("Points", points; ncomp=3))</Points>
    <Cells>$(darray("connectivity", conn))$(darray("offsets", offs))$(darray("types", typs))</Cells>
    $celldata
    </Piece>
    </UnstructuredGrid>
    </VTKFile>
    """)
  return path
end

@testset "ReadVTK extension" begin
  @test Base.get_extension(RadiativeViewFactor,
                           :RadiativeViewFactorReadVTKExt) !== nothing

  # Cube with a per-cell CellEntityIds array: bottom+top = 1, four sides = 2
  f = write_cube_vtu(tempname() * ".vtu"; cell_ids=[1, 1, 2, 2, 2, 2])
  m = load_vtu(f; verbose=false)
  @test m.mesh_dim == 2
  @test size(m.coords) == (3, 8)
  @test length(m.surface_elems) == 6
  @test all(e -> e.family === :quad4, m.surface_elems)
  @test m.surface_elems[1].nodes == [1, 2, 3, 4]   # 0-based VTK → 1-based
  @test sort(collect(keys(m.group_elems))) == [1, 2]
  @test m.group_elems[1] == [1, 2]
  @test m.group_elems[2] == [3, 4, 5, 6]
  @test m.group_tags == Dict(1 => "group_1", 2 => "group_2")

  # load_mesh routes .vtu through the same extension
  m_route = load_mesh(f; verbose=false)
  @test length(m_route.surface_elems) == 6
  @test m_route.group_elems == m.group_elems

  # Loaded geometry is usable: bottom→top of a unit cube, F ≈ 0.19982
  r = compute_view_factors(m; nquad=6, use_duffy=true, verbose=false)
  @test isapprox(r.A_elem[1], 1.0; atol=1e-9)
  @test isapprox(r.F_elem[1, 2], 0.19982; atol=2e-4)
  @test all(isapprox.(vec(sum(r.F_elem; dims=2)), 1.0; atol=1e-2))

  # No per-cell group array → single synthesized "default" group
  f_nog = write_cube_vtu(tempname() * ".vtu")
  m_nog = load_vtu(f_nog; verbose=false)
  @test length(m_nog.surface_elems) == 6
  @test m_nog.group_tags == Dict(1 => "default")
  @test m_nog.group_elems[1] == collect(1:6)

  # surface_dim=1 on a mesh with only 2D cells → helpful error
  @test_throws ErrorException load_vtu(f; surface_dim=1, verbose=false)

  foreach(rm, (f, f_nog))
end
