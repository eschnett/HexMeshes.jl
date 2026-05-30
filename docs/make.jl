# Generate documentation with this command:
# (cd docs && julia make.jl)

push!(LOAD_PATH, "..")

using Documenter
using HexMeshes

makedocs(; sitename="HexMeshes", format=Documenter.HTML(), modules=[HexMeshes])

deploydocs(; repo="github.com/eschnett/HexMeshes.jl.git", devbranch="main", push_preview=true)
