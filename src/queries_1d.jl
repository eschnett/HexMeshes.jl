# ----------------------------------------------------------------------
# Host-side queries — 1D
#
# 1D element-corner extraction and field interpolation. Much simpler
# than the 3D case: locating an element on a sorted line mesh is O(log Ne)
# via search, and inverting the linear map is one division.

"""
    element_vertices(mesh::LineMesh, e) → NTuple{2, SVector{1, T}}

The two endpoint vertices of element `e` in Gmsh order (vertex 1 = left
end, vertex 2 = right end).
"""
@inline function element_vertices(mesh::LineMesh{T}, e::Integer) where {T}
    @inbounds ntuple(v -> begin
        vi = mesh.vertex_idx[v, e]
        SVector{1, T}(mesh.vertex_coords[1, vi])
    end, Val(2))
end

"""
    invert_element_map(verts::NTuple{2, SVector{1, T}}, p) → (ξ::SVector{1, T}, ok::Bool)

Solve `linear_map(verts, ξ) = p` for the reference coordinate `ξ`. In
1D this is just `(p − v₁) / (v₂ − v₁)`; the second return value is
`true` iff `p` falls inside `[v₁, v₂]` (within `tol` of the closer
endpoint).
"""
function invert_element_map(verts::NTuple{2, SVector{1, T}}, p::SVector{1, T};
                            tol = T(1e-12), maxiter::Int = 1) where {T}
    v1 = verts[1][1]
    v2 = verts[2][1]
    h  = v2 - v1
    ξ  = (p[1] - v1) / h
    ok = -tol ≤ ξ ≤ one(T) + tol
    return SVector{1, T}(ξ), ok
end

"""
    locate_point(mesh::LineMesh, p::SVector{1, T}) → (e, ξ)

Find the element index `e` that contains the physical point `p`, and
the reference-coordinate `ξ` such that `linear_map(verts(e), ξ) = p`.
Returns `(0, …)` if `p` lies outside every element (with `tol`
slack). The line-mesh search is `O(Ne)` linear scan — sufficient for
the visualisation / diagnostic uses this is meant for.
"""
function locate_point(mesh::LineMesh{T}, p::SVector{1, T};
                      tol = T(1e-12)) where {T}
    for e in 1:mesh.Ne
        verts = element_vertices(mesh, e)
        ξ, ok = invert_element_map(verts, p; tol = tol)
        if ok
            return e, ξ
        end
    end
    return 0, SVector{1, T}(zero(T))
end

"""
    interpolate_field(mesh::LineMesh{T}, xs::AbstractVector,
                       u::AbstractArray{T, 2}, p::SVector{1, T};
                       default = zero(T)) → T

Evaluate the per-element nodal field `u` of shape `(N, Ne)` at the
physical point `p`. `xs ∈ [0, 1]^N` is the 1D reference-coord node grid
(usually the GLL nodes from `HexSBPSAT.make_element`).

Returns `default` if `p` falls outside every element.
"""
function interpolate_field(mesh::LineMesh{T},
                           xs::AbstractVector,
                           u::AbstractArray{T, 2},
                           p::SVector{1, T};
                           default = zero(T)) where {T}
    e, ξ = locate_point(mesh, p)
    e == 0 && return default
    N = length(xs)
    @assert size(u, 1) == N && size(u, 2) == mesh.Ne
    # Lagrange interpolation along the reference axis.
    L = lagrange_basis(T.(xs), ξ[1])
    s = zero(T)
    @inbounds for i in 1:N
        s += L[i] * u[i, e]
    end
    return s
end

"""
    interpolate_field(mesh::LineMesh{T}, xs::AbstractVector,
                       u::AbstractArray{T, 2},
                       pts::AbstractArray{<:SVector{1, T}};
                       default = zero(T)) → Array{T}

Vectorised form: evaluate the field at every point in `pts`. The result
has the same shape as `pts`.
"""
function interpolate_field(mesh::LineMesh{T},
                           xs::AbstractVector,
                           u::AbstractArray{T, 2},
                           pts::AbstractArray{<:SVector{1, T}};
                           default = zero(T)) where {T}
    out = similar(pts, T)
    for i in eachindex(pts)
        out[i] = interpolate_field(mesh, xs, u, pts[i]; default)
    end
    return out
end
