# ----------------------------------------------------------------------
# Host-side queries — 2D
#
# 2D analogs of `element_vertices`, `invert_element_map`, `locate_point`,
# `interpolate_field`. Newton iteration inverts the bilinear map; a
# brute-force `O(Ne)` element scan locates points. Plenty fast for the
# visualisation / diagnostic uses these are meant for.

"""
    element_vertices(mesh::QuadMesh, e) → NTuple{4, SVector{2, T}}

The four corner vertices of element `e` in Gmsh-canonical order.
"""
@inline function element_vertices(mesh::QuadMesh{T}, e::Integer) where {T}
    @inbounds ntuple(v -> begin
        vi = mesh.vertex_idx[v, e]
        SVector{2, T}(mesh.vertex_coords[1, vi],
                      mesh.vertex_coords[2, vi])
    end, Val(4))
end

"""
    invert_element_map(verts::NTuple{4, SVector{2, T}}, p; tol, maxiter) → (ξ, ok)

Solve `bilinear_map(verts, ξ, η) = p` for the reference coordinate
`(ξ, η)` via Newton iteration starting from the element centroid.
Returns `(ξ, ok)` where `ok = true` iff the residual fell below
`tol` and `(ξ, η)` is inside `[−tol, 1+tol]²`.
"""
function invert_element_map(verts::NTuple{4, SVector{2, T}}, p::SVector{2, T};
                            tol = T(1e-12), maxiter::Int = 20) where {T}
    ξ = SVector{2, T}(one(T) / 2, one(T) / 2)
    res_last = T(Inf)
    for _ in 1:maxiter
        x = bilinear_map(verts, ξ[1], ξ[2])
        r = x - p
        res = sqrt(r[1]^2 + r[2]^2)
        res < tol && break
        J = bilinear_jacobian(verts, ξ[1], ξ[2])
        Δξ = J \ r
        ξ = ξ - Δξ
        res_last = res
    end
    x_final = bilinear_map(verts, ξ[1], ξ[2])
    r_final = x_final - p
    res = sqrt(r_final[1]^2 + r_final[2]^2)
    ok = res < tol &&
         -tol ≤ ξ[1] ≤ one(T) + tol &&
         -tol ≤ ξ[2] ≤ one(T) + tol
    return ξ, ok
end

"""
    locate_point(mesh::QuadMesh, p::SVector{2, T}; tol) → (e, ξ)

Find the element `e` that contains the physical point `p`, and the
reference-coordinate `ξ` such that `bilinear_map(verts(e), ξ) = p`.
Returns `(0, …)` if `p` lies outside every element. `O(Ne)` brute-
force search — adequate for plotting / diagnostics, not for hot loops.
"""
function locate_point(mesh::QuadMesh{T}, p::SVector{2, T};
                      tol = T(1e-12)) where {T}
    for e in 1:mesh.Ne
        verts = element_vertices(mesh, e)
        ξ, ok = invert_element_map(verts, p; tol = tol)
        if ok
            return e, ξ
        end
    end
    return 0, SVector{2, T}(zero(T), zero(T))
end

"""
    interpolate_field(mesh::QuadMesh{T}, xs::AbstractVector,
                       u::AbstractArray{T, 3}, p::SVector{2, T};
                       default = zero(T)) → T

Evaluate the per-element nodal field `u` of shape `(N, N, Ne)` at the
physical point `p`. `xs ∈ [0, 1]^N` is the reference-coord node grid
(usually GLL nodes); the same `xs` is used along both axes.

Returns `default` if `p` falls outside every element.
"""
function interpolate_field(mesh::QuadMesh{T},
                           xs::AbstractVector,
                           u::AbstractArray{T, 3},
                           p::SVector{2, T};
                           default = zero(T)) where {T}
    e, ξ = locate_point(mesh, p)
    e == 0 && return default
    N = length(xs)
    @assert size(u) == (N, N, mesh.Ne)
    L1 = lagrange_basis(T.(xs), ξ[1])
    L2 = lagrange_basis(T.(xs), ξ[2])
    s = zero(T)
    @inbounds for j in 1:N, i in 1:N
        s += L1[i] * L2[j] * u[i, j, e]
    end
    return s
end

"""
    interpolate_field(mesh::QuadMesh{T}, xs::AbstractVector,
                       u::AbstractArray{T, 3},
                       pts::AbstractArray{<:SVector{2, T}};
                       default = zero(T)) → Array{T}

Vectorised form: evaluate the field at every point in `pts`. The result
has the same shape as `pts`.
"""
function interpolate_field(mesh::QuadMesh{T},
                           xs::AbstractVector,
                           u::AbstractArray{T, 3},
                           pts::AbstractArray{<:SVector{2, T}};
                           default = zero(T)) where {T}
    out = similar(pts, T)
    for i in eachindex(pts)
        out[i] = interpolate_field(mesh, xs, u, pts[i]; default)
    end
    return out
end

"""
    locate_point(mesh::InflatedSquareMesh{T}, p::SVector{2, T}; tol)
        → (element_index, ξ)

`O(1)` analytic point location with no Newton iteration. Composes
[`locate_patch`](@ref), [`global_to_patch`](@ref), and
[`locate_element_in_patch`](@ref) — patch finder, closed-form
inverse of the patch parametric map, and patch-local cell selector.
Returns `(0, …)` if `p` is outside the disk `|x| ≤ R2 + tol`.

The returned `ξ ∈ [0, 1]²` is the **analytic patch-parameter** ξ
inside the element. This matches the placement of GLL nodes by
`_patch_point_and_jac_2d`, so downstream Lagrange interpolation
against the patch GLL grid is exact at the node values and converges
at the element's polynomial order in the interior.
"""
function locate_point(mesh::InflatedSquareMesh{T}, p::SVector{2, T};
                       tol = sqrt(eps(T))) where {T}
    pi = locate_patch(mesh, p; tol)
    pi == 0 && return 0, SVector{2, T}(zero(T), zero(T))
    ξ_patch = global_to_patch(mesh, pi, p; tol)
    isnan(ξ_patch[1]) && return 0, SVector{2, T}(zero(T), zero(T))
    return locate_element_in_patch(mesh, pi, ξ_patch)
end
function interpolate_field(mesh::InflatedSquareMesh{T},
                            xs::AbstractVector,
                            u::AbstractArray{T, 3},
                            p::SVector{2, T};
                            default = zero(T)) where {T}
    return interpolate_field(mesh.base, xs, u, p; default)
end
function interpolate_field(mesh::InflatedSquareMesh{T},
                            xs::AbstractVector,
                            u::AbstractArray{T, 3},
                            pts::AbstractArray{<:SVector{2, T}};
                            default = zero(T)) where {T}
    return interpolate_field(mesh.base, xs, u, pts; default)
end
