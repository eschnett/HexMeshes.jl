# `vertex_coords` table as an 8-tuple of `SVector{3, T}`.
@inline function element_vertices(mesh::HexMesh{T}, e::Integer) where {T}
    @inbounds ntuple(v -> begin
        vi = mesh.vertex_idx[v, e]
        SVector{3, T}(mesh.vertex_coords[1, vi],
                      mesh.vertex_coords[2, vi],
                      mesh.vertex_coords[3, vi])
    end, Val(8))
end

# Forward element-corner queries through the `InflatedCubeMesh` wrapper.
# The trilinear corners are only an approximation of the curved-patch
# geometry, but match exactly on the inner cube and are good enough for
# bounding-box reject (`locate_point`) and for plotting.
@inline function element_vertices(mesh::InflatedCubeMesh{T}, e::Integer) where {T}
    return element_vertices(mesh.base, e)
end


# ----------------------------------------------------------------------
# Interpolation from per-element data to arbitrary physical points
#
# Used for visualisation and diagnostics. Not optimised: the point-to-
# element search is `O(Ne)` per query and the inversion of the trilinear
# map runs a small Newton iteration. Plenty fast for plotting grids of a
# few thousand points; never call this from a hot inner loop.

"""
    invert_element_map(verts, p; tol, maxiter) → (ξ::SVector{3, T}, ok::Bool)

Solve `trilinear_map(verts, ξ, η, ζ) = p` for the reference coordinate
`(ξ, η, ζ)`. Returns the converged `ξ` and a flag indicating whether the
residual fell below `tol` within `maxiter` Newton steps.
"""
function invert_element_map(verts::NTuple{8, SVector{3, T}}, p::SVector{3, T};
                             tol = T(1e-12), maxiter::Int = 20) where {T}
    ξ = SVector{3, T}(one(T)/2, one(T)/2, one(T)/2)
    res_last = T(Inf)
    for _ in 1:maxiter
        x = trilinear_map(verts, ξ[1], ξ[2], ξ[3])
        r = x - p
        res_last = sqrt(r[1]^2 + r[2]^2 + r[3]^2)
        res_last < tol && return ξ, true
        J = trilinear_jacobian(verts, ξ[1], ξ[2], ξ[3])
        ξ = ξ - (J \ r)
    end
    return ξ, res_last < sqrt(tol)
end

"""
    locate_point(mesh::InflatedCubeMesh{T}, p::SVector{3, T}; tol)
        → (element_index, ξ)

Fast analytic point location: combines `locate_patch` and
`locate_element_in_patch` for `O(1)` patch+element identification,
followed by one Newton iteration on the trilinear element map to
recover the precise reference coordinate `ξ` such that
`trilinear_map(verts, ξ) ≈ p` to within `tol`.

Falls back to the brute-force `locate_point(mesh.base, p)` if Newton
fails to converge on the analytically-identified element.
"""
function locate_point(mesh::InflatedCubeMesh{T}, p::SVector{3, T};
                       tol = sqrt(eps(T))) where {T}
    pi = locate_patch(mesh, p; tol)
    pi == 0 && return 0, SVector{3, T}(zero(T), zero(T), zero(T))
    ξ_patch = global_to_patch(mesh, pi, p; tol)
    isnan(ξ_patch[1]) && return 0, SVector{3, T}(zero(T), zero(T), zero(T))
    return locate_element_in_patch(mesh, pi, ξ_patch)
end

function locate_point(mesh::HexMesh{T}, p::SVector{3, T};
                       tol = T(1e-8)) where {T}
    @inbounds for e in 1:mesh.Ne
        verts = element_vertices(mesh, e)
        # Cheap reject: skip elements whose bounding box does not contain p.
        xmin = min(verts[1][1], verts[2][1], verts[3][1], verts[4][1],
                   verts[5][1], verts[6][1], verts[7][1], verts[8][1])
        xmax = max(verts[1][1], verts[2][1], verts[3][1], verts[4][1],
                   verts[5][1], verts[6][1], verts[7][1], verts[8][1])
        (p[1] < xmin - tol || p[1] > xmax + tol) && continue
        ymin = min(verts[1][2], verts[2][2], verts[3][2], verts[4][2],
                   verts[5][2], verts[6][2], verts[7][2], verts[8][2])
        ymax = max(verts[1][2], verts[2][2], verts[3][2], verts[4][2],
                   verts[5][2], verts[6][2], verts[7][2], verts[8][2])
        (p[2] < ymin - tol || p[2] > ymax + tol) && continue
        zmin = min(verts[1][3], verts[2][3], verts[3][3], verts[4][3],
                   verts[5][3], verts[6][3], verts[7][3], verts[8][3])
        zmax = max(verts[1][3], verts[2][3], verts[3][3], verts[4][3],
                   verts[5][3], verts[6][3], verts[7][3], verts[8][3])
        (p[3] < zmin - tol || p[3] > zmax + tol) && continue
        ξ, ok = invert_element_map(verts, p)
        ok && all(-tol ≤ ξ[i] ≤ 1 + tol for i in 1:3) && return e, ξ
    end
    return 0, SVector{3, T}(zero(T), zero(T), zero(T))
end

# 1D Lagrange basis values at `ξ` for nodes `xs`. Length(xs) = N → returns
# `NTuple{N, T}`. Generic and unoptimised — `O(N²)` per call.
function lagrange_basis(xs, ξ::T) where {T}
    N = length(xs)
    out = ntuple(N) do i
        v = one(T)
        @inbounds for j in 1:N
            i == j && continue
            v *= (ξ - xs[j]) / (xs[i] - xs[j])
        end
        v
    end
    return out
end

# Tensor-product Lagrange interpolation of an `(N, N, N)` block at the
# reference point `(ξ, η, ζ)`. `xs` are the 1D GLL nodes on `[0, 1]`.
function tensor_interp(ue::AbstractArray{T, 3},
                        ξ::T, η::T, ζ::T, xs) where {T}
    N = length(xs)
    ℓξ = lagrange_basis(xs, ξ)
    ℓη = lagrange_basis(xs, η)
    ℓζ = lagrange_basis(xs, ζ)
    s = zero(T)
    @inbounds for k in 1:N, j in 1:N, i in 1:N
        s += ue[i, j, k] * ℓξ[i] * ℓη[j] * ℓζ[k]
    end
    return s
end

"""
    interpolate_field(mesh, xs, u, p; default) → T

Evaluate the per-element field `u` (shape `(N, N, N, Ne)`) at the
physical point `p` by locating the element of `mesh::HexMesh` that
contains `p`, inverting the trilinear element map, and applying
tensor-product Lagrange interpolation on the 1-D reference-element
nodes `xs::AbstractVector{T}` (typically `elem.xs ∈ [0, 1]` from the
caller's SBP element). Returns `default` if `p` lies outside the
mesh. Brute-force, intended for visualisation.

Note: prior to the `HexMeshes` / `WaveToySecondOrder` split this
signature took an `elem` object. Callers now pass `elem.xs`
explicitly so `HexMeshes` does not need to know about the downstream
element / operator type.
"""
function interpolate_field(mesh::HexMesh{T},
                            xs::AbstractVector,
                            u::AbstractArray{T, 4},
                            p::SVector{3, T};
                            default = T(NaN)) where {T}
    e, ξ = locate_point(mesh, p)
    e == 0 && return default
    return tensor_interp(view(u, :, :, :, e), ξ[1], ξ[2], ξ[3], xs)
end

interpolate_field(mesh::InflatedCubeMesh, xs, u, p; kwargs...) =
    interpolate_field(mesh.base, xs, u, p; kwargs...)

# Vectorised convenience: take any iterable of points and return an
# array of values with the same shape.
function interpolate_field(mesh::HexMesh{T},
                            xs::AbstractVector,
                            u::AbstractArray{T, 4},
                            points::AbstractArray{<:SVector{3, T}};
                            default = T(NaN)) where {T}
    out = similar(points, T)
    for I in eachindex(points)
        out[I] = interpolate_field(mesh, xs, u, points[I]; default)
    end
    return out
end
