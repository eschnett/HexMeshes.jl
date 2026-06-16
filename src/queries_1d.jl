# ----------------------------------------------------------------------
# Host-side queries — 1D
#
# 1D versions of element-corner extraction, analytic patch ↔ global
# coordinate maps, and field interpolation. Current 1D mesh builders
# (`make_line_mesh`) only produce `Cubic`-kind patches; curvilinear
# variants (Wedge / Inflation / Shell) are type-legal but not
# constructed, so the patch maps below only handle `Cubic` and error
# otherwise.

"""
    element_vertices(mesh::Mesh{1, T}, e) → NTuple{2, SVector{1, T}}

The two endpoint vertices of element `e` in Gmsh order (vertex 1 = left
end, vertex 2 = right end).
"""
@inline function element_vertices(mesh::Mesh{1, T}, e::Integer) where {T}
    @inbounds ntuple(v -> begin
        vi = mesh.vertex_idx[v, e]
        SVector{1, T}(mesh.vertex_coords[1, vi])
    end, Val(2))
end

"""
    element_point_and_jac(mesh::Mesh{1, T}, e, ξ::SVector{1, T}) → (P, J)

Physical position `P` and element Jacobian `J[1, 1] = ∂x/∂ξ` of the reference
point `ξ ∈ [0, 1]` of element `e`. The 1D analog of the `Mesh{3}` method. Current
1D meshes are `Cubic` (affine), so this is the linear corner map.
"""
function element_point_and_jac(mesh::Mesh{1, T}, e::Integer, ξ::SVector{1, T}) where {T}
    verts = element_vertices(mesh, e)
    return linear_map(verts, ξ[1]), linear_jacobian(verts, ξ[1])
end

"""
    invert_element_map(verts::NTuple{2, SVector{1, T}}, p) → (ξ::SVector{1, T}, ok::Bool)

Solve `linear_map(verts, ξ) = p` for the reference coordinate `ξ`. In
1D this is just `(p − v₁) / (v₂ − v₁)`; `ok` is `true` iff `p` falls
inside `[v₁, v₂]` (within `tol`).

**Deprecated.** No direct replacement: `locate_point(mesh, p)` returns
the analytic patch-parameter ξ (which for 1D `Cubic` patches happens to
equal the trilinear ξ, so the answers coincide).
"""
function invert_element_map(verts::NTuple{2, SVector{1, T}}, p::SVector{1, T};
                            tol = default_tol(T), maxiter::Int = 1) where {T}
    Base.depwarn(
        "`invert_element_map(verts, p)` (1D) is deprecated; for point→(element, ξ) " *
        "use `locate_point(mesh, p)`.",
        :invert_element_map_1d)
    v1 = verts[1][1]
    v2 = verts[2][1]
    h  = v2 - v1
    ξ  = (p[1] - v1) / h
    ok = -tol ≤ ξ ≤ one(T) + tol
    return SVector{1, T}(ξ), ok
end

# ----- Analytic patch ↔ global maps for 1D ---------------------------

"""
    patch_to_global(pd::PatchDesc{1, T}, ξ::SVector{1, T}) → SVector{1, T}

Forward parametric map for a 1D patch. Only `Cubic` patches are
supported (the only kind produced by current 1D builders); other kinds
error.
"""
function patch_to_global(pd::PatchDesc{1, T}, ξ::SVector{1, T}) where {T}
    if pd.kind === Cubic
        c = pd.cubic
        return SVector{1, T}(c.x_lo[1] + (c.x_hi[1] - c.x_lo[1]) * ξ[1])
    else
        error("patch_to_global: 1D meshes only support PatchDesc.kind === Cubic; got $(pd.kind).")
    end
end

"""
    global_to_patch(pd::PatchDesc{1, T}, p::SVector{1, T}; tol) → SVector{1, T}

Inverse of [`patch_to_global`](@ref) for 1D. Returns `SVector(NaN)` if
`p` lies outside this patch by more than `tol`.
"""
function global_to_patch(pd::PatchDesc{1, T}, p::SVector{1, T};
                          tol = default_tol(T)) where {T}
    if pd.kind === Cubic
        c = pd.cubic
        ξ = (p[1] - c.x_lo[1]) / (c.x_hi[1] - c.x_lo[1])
        if -tol ≤ ξ ≤ one(T) + tol
            return SVector{1, T}(clamp(ξ, zero(T), one(T)))
        end
        return _outside_xi(Val(1), T)
    else
        error("global_to_patch: 1D meshes only support PatchDesc.kind === Cubic; got $(pd.kind).")
    end
end

"""
    locate_patch(mesh::Mesh{1, T}, p::SVector{1, T}; tol) → Int

Find the 1-indexed patch containing `p`. Walks `mesh.patch_desc` and
returns the first patch that accepts `p`. Returns `0` if no patch
contains it.
"""
function locate_patch(mesh::Mesh{1, T}, p::SVector{1, T};
                       tol = default_tol(T)) where {T}
    for (p_idx, pd) in enumerate(mesh.patch_desc)
        ξ = global_to_patch(pd, p; tol)
        _is_outside(ξ) && continue
        return p_idx
    end
    return 0
end

"""
    locate_element_in_patch(mesh::Mesh{1, T}, patch_index, ξ_patch)
        → (element_index, ξ_in_element)
"""
function locate_element_in_patch(mesh::Mesh{1, T},
                                  patch_index::Integer,
                                  ξ_patch::SVector{1, T}) where {T}
    pd = mesh.patch_desc[patch_index]
    d = dims(pd)
    s_a = clamp(ξ_patch[1], zero(T), one(T)) * d[1]
    a_cell = min(d[1], max(1, floor(Int, s_a) + 1))
    ξ_elem_a = s_a - (a_cell - 1)
    elem_off = mesh.patch_element_offset[patch_index]
    e = elem_off + a_cell
    return e, SVector{1, T}(ξ_elem_a)
end

"""
    locate_point(mesh::Mesh{1, T}, p::SVector{1, T}; tol) → (e, ξ)

Analytic point location for 1D meshes: identify patch via
`locate_patch`, invert the patch parametric map with `global_to_patch`,
convert to `(element_index, ξ_in_element)` via
`locate_element_in_patch`. Returns `(0, zero ξ)` if `p` lies outside.
"""
function locate_point(mesh::Mesh{1, T}, p::SVector{1, T};
                      tol = default_tol(T)) where {T}
    pidx = locate_patch(mesh, p; tol)
    pidx == 0 && return 0, SVector{1, T}(zero(T))
    ξ_patch = global_to_patch(mesh.patch_desc[pidx], p; tol)
    _is_outside(ξ_patch) && return 0, SVector{1, T}(zero(T))
    return locate_element_in_patch(mesh, pidx, ξ_patch)
end

"""
    interpolate_field(mesh::Mesh{1, T}, xs::AbstractVector,
                       u::AbstractArray{T, 2}, p::SVector{1, T};
                       default = zero(T)) → T

Evaluate the per-element nodal field `u` of shape `(N, Ne)` at the
physical point `p`. `xs ∈ [0, 1]^N` is the 1D reference-coord node grid
(usually the GLL nodes from `HexSBPSAT.make_element`).

Returns `default` if `p` falls outside every element.
"""
function interpolate_field(mesh::Mesh{1, T},
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
    interpolate_field(mesh::Mesh{1, T}, xs::AbstractVector,
                       u::AbstractArray{T, 2},
                       pts::AbstractArray{<:SVector{1, T}};
                       default = zero(T)) → Array{T}

Vectorised form: evaluate the field at every point in `pts`. The result
has the same shape as `pts`.
"""
function interpolate_field(mesh::Mesh{1, T},
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
