using Test
using HexMeshes

@testset "HexMeshes" begin
    include("test_mesh.jl")
    include("test_mesh_1d.jl")
    include("test_mesh_2d.jl")
end
