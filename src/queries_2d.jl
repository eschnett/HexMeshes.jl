# ----------------------------------------------------------------------
# Host-side queries — 2D
#
# 2D analogs of `element_vertices`, `invert_element_map`, `locate_point`,
# `interpolate_field`. `locate_point` uses the analytic patch path
# (`O(npatches)`, no Newton); `invert_element_map` is kept for legacy
# callers that explicitly want the bilinear-map inverse.

"""
    element_vertices(mesh::Mesh{2, T}, e) → NTuple{4, SVector{2, T}}

Four corner vertices of element `e` in Gmsh-canonical order.
"""
@inline function element_vertices(mesh::Mesh{2, T}, e::Integer) where {T}
    @inbounds ntuple(v -> begin
        vi = mesh.vertex_idx[v, e]
        SVector{2, T}(mesh.vertex_coords[1, vi],
                      mesh.vertex_coords[2, vi])
    end, Val(4))
end

"""
    invert_element_map(verts::NTuple{4, SVector{2, T}}, p; tol, maxiter) → (ξ, ok)

Newton inverse of the bilinear corner map.

**Deprecated.** No direct replacement: `locate_point` returns the
analytic patch-parameter `ξ` (different semantics from the bilinear ξ
this function recovers).
"""
function invert_element_map(verts::NTuple{4, SVector{2, T}}, p::SVector{2, T};
                            tol = default_tol(T), maxiter::Int = 20) where {T}
    Base.depwarn(
        "`invert_element_map(verts, p)` (2D) is deprecated; for point→(element, ξ) " *
        "use `locate_point(mesh, p)` which returns the analytic patch-parameter ξ.",
        :invert_element_map_2d)
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
    locate_point(mesh::Mesh{2, T}, p::SVector{2, T}; tol) → (e, ξ)

Analytic point location: identify patch via `locate_patch`, invert the
patch parametric map analytically with `global_to_patch`, then convert
to `(element_index, ξ_in_element)` via `locate_element_in_patch`.
Returns `(0, zero ξ)` if `p` lies outside every patch.

The returned `ξ_in_element` is the analytic patch-parameter ξ — for
curvilinear elements this differs from the bilinear-map ξ but matches
the placement of GLL nodes inside the element.
"""
function locate_point(mesh::Mesh{2, T}, p::SVector{2, T};
                      tol = default_tol(T)) where {T}
    pidx = locate_patch(mesh, p; tol)
    pidx == 0 && return 0, SVector{2, T}(zero(T), zero(T))
    ξ_patch = global_to_patch(mesh.patch_desc[pidx], p; tol)
    _is_outside(ξ_patch) && return 0, SVector{2, T}(zero(T), zero(T))
    return locate_element_in_patch(mesh, pidx, ξ_patch)
end

"""
    interpolate_field(mesh::Mesh{2, T}, xs, u, p; default) → T

Evaluate the per-element nodal field `u` of shape `(N, N, Ne)` at the
physical point `p`.
"""
function interpolate_field(mesh::Mesh{2, T},
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

# Vectorised convenience.
function interpolate_field(mesh::Mesh{2, T},
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

# ----- 2D analytic patch ↔ global maps ------------------------------
#
# Per-direction inverse of `_patch_direction_vec_2d`.
#   +x:  v = ( 1,  b)  →  f = x,   b = y/x
#   -x:  v = (-1, -b)  →  f = -x,  b = y/x
#   +y:  v = (-b,  1)  →  f = y,   b = -x/y
#   -y:  v = ( b, -1)  →  f = -y,  b = -x/y
@inline function _inverse_dir_vec_2d(dir::Integer, x::T, y::T) where {T}
    if dir == 1                     # +x
        return  x,  y / x
    elseif dir == 2                 # -x
        return -x,  y / x
    elseif dir == 3                 # +y
        return  y, -x / y
    else                            # -y
        return -y, -x / y
    end
end

"""
    patch_to_global(pd::PatchDesc{2, T}, ξ::SVector{2, T}) → SVector{2, T}

Forward parametric map for a 2D patch. Exact inverse of
[`global_to_patch`](@ref) on the patch's parameter domain.
"""
function patch_to_global(pd::PatchDesc{2, T}, ξ::SVector{2, T}) where {T}
    k = pd.kind
    if k === Cubic
        c = pd.cubic
        return SVector{2, T}(c.x_lo[1] + (c.x_hi[1] - c.x_lo[1]) * ξ[1],
                              c.x_lo[2] + (c.x_hi[2] - c.x_lo[2]) * ξ[2])
    elseif k === Inflation
        pi = pd.inflation
        a = pi.a_lo + (pi.a_hi - pi.a_lo) * ξ[1]
        b = pi.b_lo + (pi.b_hi - pi.b_lo) * ξ[2]
        Q = sqrt(one(T) + b * b)
        f = (one(T) - a) * pi.L + a * pi.R1 / Q
        vx, vy = _patch_direction_vec_2d(pi.dir, b)
        return SVector{2, T}(f * vx, f * vy)
    elseif k === Shell
        ps = pd.shell
        a = ps.a_lo + (ps.a_hi - ps.a_lo) * ξ[1]
        b = ps.b_lo + (ps.b_hi - ps.b_lo) * ξ[2]
        Q = sqrt(one(T) + b * b)
        r = (one(T) - a) * ps.R1 + a * ps.R2
        f = r / Q
        vx, vy = _patch_direction_vec_2d(ps.dir, b)
        return SVector{2, T}(f * vx, f * vy)
    else  # Wedge
        w = pd.wedge
        a = w.a_lo + (w.a_hi - w.a_lo) * ξ[1]
        b = w.b_lo + (w.b_hi - w.b_lo) * ξ[2]
        r = w.R1 * (w.R2 / w.R1)^a
        dir = w.dir
        if     dir == Int8(1);  return SVector{2, T}( r,  b * r)
        elseif dir == Int8(2);  return SVector{2, T}(-r,  b * r)
        elseif dir == Int8(3);  return SVector{2, T}(b * r,  r)
        else                    return SVector{2, T}(b * r, -r)
        end
    end
end

# Mesh-level convenience overload. **Deprecated** — pass the PatchDesc
# directly via `mesh.patch_desc[patch_index]`.
Base.@deprecate patch_to_global(mesh::Mesh{2, T}, patch_index::Integer,
                                  ξ::SVector{2, T}) where {T} (
    patch_to_global(mesh.patch_desc[patch_index], ξ))

"""
    global_to_patch(pd::PatchDesc{2, T}, p::SVector{2, T}; tol)
        → SVector{2, T}

Inverse of [`patch_to_global`](@ref). Returns `_outside_xi(Val(2), T)`
if `p` lies outside this patch by more than `tol`. Closed-form for all
four patch kinds.
"""
function global_to_patch(pd::PatchDesc{2, T}, p::SVector{2, T};
                          tol = default_tol(T)) where {T}
    NaN_ξ = _outside_xi(Val(2), T)
    k = pd.kind
    if k === Cubic
        c = pd.cubic
        ξ_a = (p[1] - c.x_lo[1]) / (c.x_hi[1] - c.x_lo[1])
        ξ_b = (p[2] - c.x_lo[2]) / (c.x_hi[2] - c.x_lo[2])
        if -tol ≤ ξ_a ≤ one(T) + tol && -tol ≤ ξ_b ≤ one(T) + tol
            return SVector{2, T}(clamp(ξ_a, zero(T), one(T)),
                                  clamp(ξ_b, zero(T), one(T)))
        end
        return NaN_ξ
    elseif k === Inflation || k === Shell
        if k === Inflation
            pi = pd.inflation
            dir, a_lo_p, a_hi_p, b_lo_p, b_hi_p, L_or_R1, top =
                (pi.dir, pi.a_lo, pi.a_hi, pi.b_lo, pi.b_hi, pi.L, pi.R1)
        else
            ps = pd.shell
            dir, a_lo_p, a_hi_p, b_lo_p, b_hi_p, L_or_R1, top =
                (ps.dir, ps.a_lo, ps.a_hi, ps.b_lo, ps.b_hi, ps.R1, ps.R2)
        end
        f_val, b = _inverse_dir_vec_2d(dir, p[1], p[2])
        if !(isfinite(b) && isfinite(f_val) && f_val > -tol)
            return NaN_ξ
        end
        Q = sqrt(one(T) + b * b)
        a = if k === Shell
            r = sqrt(p[1]^2 + p[2]^2)
            (r - L_or_R1) / (top - L_or_R1)
        else
            (f_val - L_or_R1) / (top / Q - L_or_R1)
        end
        ξ_a = (a - a_lo_p) / (a_hi_p - a_lo_p)
        ξ_b = (b - b_lo_p) / (b_hi_p - b_lo_p)
        if -tol ≤ ξ_a ≤ one(T) + tol && -tol ≤ ξ_b ≤ one(T) + tol
            return SVector{2, T}(clamp(ξ_a, zero(T), one(T)),
                                  clamp(ξ_b, zero(T), one(T)))
        end
        return NaN_ξ
    else  # Wedge
        w = pd.wedge
        f_val, b = _inverse_dir_vec_2d(w.dir, p[1], p[2])
        if !(isfinite(b) && isfinite(f_val) && f_val > -tol)
            return NaN_ξ
        end
        a = log(f_val / w.R1) / log(w.R2 / w.R1)
        ξ_a = (a - w.a_lo) / (w.a_hi - w.a_lo)
        ξ_b = (b - w.b_lo) / (w.b_hi - w.b_lo)
        if -tol ≤ ξ_a ≤ one(T) + tol && -tol ≤ ξ_b ≤ one(T) + tol
            return SVector{2, T}(clamp(ξ_a, zero(T), one(T)),
                                  clamp(ξ_b, zero(T), one(T)))
        end
        return NaN_ξ
    end
end

# Mesh-level convenience overload. **Deprecated** — pass the PatchDesc
# directly via `mesh.patch_desc[patch_index]`.
Base.@deprecate global_to_patch(mesh::Mesh{2, T}, patch_index::Integer,
                                  p::SVector{2, T};
                                  tol = default_tol(T)) where {T} (
    global_to_patch(mesh.patch_desc[patch_index], p; tol))

"""
    locate_patch(mesh::Mesh{2, T}, p::SVector{2, T}; tol) → Int

Find the 1-indexed patch containing `p` by trying `global_to_patch` on
each patch in `mesh.patch_desc` in order. Returns `0` if no patch
contains `p`.
"""
function locate_patch(mesh::Mesh{2, T}, p::SVector{2, T};
                       tol = default_tol(T)) where {T}
    for (p_idx, pd) in enumerate(mesh.patch_desc)
        ξ = global_to_patch(pd, p; tol)
        _is_outside(ξ) && continue
        return p_idx
    end
    return 0
end

"""
    locate_element_in_patch(mesh::Mesh{2, T}, patch_index, ξ_patch)
        → (element_index, ξ_in_element)
"""
function locate_element_in_patch(mesh::Mesh{2, T},
                                  patch_index::Integer,
                                  ξ_patch::SVector{2, T}) where {T}
    pd = mesh.patch_desc[patch_index]
    d = dims(pd)
    s_a = clamp(ξ_patch[1], zero(T), one(T)) * d[1]
    s_b = clamp(ξ_patch[2], zero(T), one(T)) * d[2]
    a_cell = min(d[1], max(1, floor(Int, s_a) + 1))
    b_cell = min(d[2], max(1, floor(Int, s_b) + 1))
    ξ_elem_a = s_a - (a_cell - 1)
    ξ_elem_b = s_b - (b_cell - 1)
    elem_off = mesh.patch_element_offset[patch_index]
    e = elem_off + a_cell + d[1] * (b_cell - 1)
    return e, SVector{2, T}(ξ_elem_a, ξ_elem_b)
end
