# ----------------------------------------------------------------------
# 1D line mesh builder
#
# Single-patch uniform mesh on `[x0, x1]`, no skeleton infrastructure
# required (multi-patch vertex dedup isn't needed for a single-patch
# 1D line). Direct construction of the `(2, Ne)` connectivity tables
# and the `(1, Nv)` / `(2, Ne)` vertex tables.

"""
    make_line_mesh(::Type{T}, M::Int, x0, x1) → LineMesh{T}

Uniform 1D mesh of `M` line elements on the interval `[x0, x1]`.
Element `e ∈ 1..M` runs from `x = x0 + (e-1)·h` to `x = x0 + e·h`,
where `h = (x1 − x0) / M`. Vertices are stored in Gmsh-canonical
order — element `e`'s vertex 1 is its left endpoint, vertex 2 its
right endpoint.

# Boundary tags

* Face 1 (−x) of element 1 — tagged `Int8(1)`.
* Face 2 (+x) of element `M` — tagged `Int8(2)`.

These are the natural defaults for Dirichlet enforcement at the two
endpoints; downstream code can overwrite `mesh.bdry` for other BCs
(e.g. Sommerfeld) just as in the 3D cubical mesh.
"""
function make_line_mesh(::Type{T}, M::Int, x0, x1) where {T}
    @assert M ≥ 1
    Ne = M
    Nv = M + 1
    h  = (T(x1) - T(x0)) / T(M)

    vertex_coords = Matrix{T}(undef, 1, Nv)
    for v in 1:Nv
        vertex_coords[1, v] = T(x0) + (v - 1) * h
    end

    vertex_idx = Matrix{Int}(undef, 2, Ne)
    for e in 1:Ne
        vertex_idx[1, e] = e        # left endpoint (ξ = 0)
        vertex_idx[2, e] = e + 1    # right endpoint (ξ = 1)
    end

    # Connectivity: each element has 2 faces (1 = −x, 2 = +x).
    neighbour      = Matrix{Int32}(undef, 2, Ne)
    neighbour_face = Matrix{Int8}(undef, 2, Ne)
    orientation    = Matrix{Int8}(undef, 2, Ne)
    bdry           = Matrix{Int8}(undef, 2, Ne)
    fill!(orientation, Int8(0))     # 1D has only the trivial orientation
    fill!(bdry,         Int8(0))    # interior by default

    for e in 1:Ne
        # Face 1 (−x): neighbour is the element on the left.
        if e == 1
            neighbour[1, e]      = Int32(0)
            neighbour_face[1, e] = Int8(0)
            bdry[1, e]           = Int8(1)
        else
            neighbour[1, e]      = Int32(e - 1)
            neighbour_face[1, e] = Int8(2)
        end
        # Face 2 (+x): neighbour is the element on the right.
        if e == Ne
            neighbour[2, e]      = Int32(0)
            neighbour_face[2, e] = Int8(0)
            bdry[2, e]           = Int8(2)
        else
            neighbour[2, e]      = Int32(e + 1)
            neighbour_face[2, e] = Int8(1)
        end
    end

    return LineMesh{T}(Ne, neighbour, neighbour_face, orientation, bdry,
                       vertex_coords, vertex_idx)
end
