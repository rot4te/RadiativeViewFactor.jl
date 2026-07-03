# test/vtk_test.jl

@testset "VTK routing (sniffer + ReadVTK guard)" begin
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

  # Without `using ReadVTK`, load_vtu must raise a helpful error (the
  # extension method is not installed in this include-based test context).
  @test_throws ErrorException RadiativeViewFactor.MeshIO.load_vtu(f_vtu; verbose=false)

  foreach(rm, (f_vtu, f_xvtk, f_legacy, f_msh))
end
