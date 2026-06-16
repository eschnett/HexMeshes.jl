"""
    default_tol(::Type{T}) → T

Default floating-point slack tolerance used by host-side queries
(`locate_point`, `locate_patch`, `global_to_patch`, `invert_element_map`)
to absorb roundoff at element / patch boundaries and to set Newton
convergence targets. Returns `sqrt(eps(T))`:

* `Float32`  ≈ `3.5e-4`
* `Float64`  ≈ `1.5e-8`
* `BigFloat` ≈ `7.5e-39` (with the 256-bit default precision)

`sqrt(eps(T))` is the standard "loose" tolerance: it's reliably
achievable by quadratically-convergent iterations and conservative
enough for "is this point inside the patch?" containment checks.
Override with the `tol` keyword on any individual call.

For `Rational` precisions, arithmetic on axis-aligned (`Cubic`) patches
is exact and no slack is needed — `default_tol(Rational{...}) = 0`.
Curvilinear (`Wedge`/`Inflation`/`Shell`/`WarpedCubic`) patches use
`sqrt`, `^`, and trigonometry, which type-promote out of `Rational`;
only `Cubic` patches are expected to work with rational precision.
"""
@inline default_tol(::Type{T}) where {T<:Real} = sqrt(eps(T))
@inline default_tol(::Type{R}) where {R<:Rational} = zero(R)

# ---------------------------------------------------------------------
# Out-of-patch sentinel used by `global_to_patch` to flag points that
# lie outside the patch's parameter domain. For floating-point `T` we
# use `NaN` (the natural choice). For `Rational` `T` there is no `NaN`
# representation, so we use a negative value outside the valid `[0, 1]`
# range; `_is_outside` does the matching predicate test. Both helpers
# are dispatched on the element type to keep the patch-inverse path
# allocation-free and type-stable.

@inline _outside_xi(::Val{D}, ::Type{T}) where {D, T<:AbstractFloat} =
    SVector{D, T}(ntuple(_ -> T(NaN), Val(D)))
@inline _outside_xi(::Val{D}, ::Type{R}) where {D, R<:Rational} =
    SVector{D, R}(ntuple(_ -> -one(R), Val(D)))

@inline _is_outside(ξ::SVector{D, T}) where {D, T<:AbstractFloat} = isnan(ξ[1])
@inline _is_outside(ξ::SVector{D, R}) where {D, R<:Rational} = ξ[1] < zero(R)

# Host-side queries for 3D meshes:
#
#   * `element_vertices(mesh, e)`   — 8 corner coords as SVectors.
#   * `locate_point(mesh, p)`        — analytic patch+element location
#                                      (O(npatches), no Newton, no
#                                      brute-force scan).
#   * `invert_element_map(verts, p)` — Newton inversion of the
#                                      trilinear corner map. Kept for
#                                      legacy callers; no longer used
#                                      by `locate_point`.
#   * `interpolate_field(mesh, xs, u, p)` — driver combining the two.

"""
    element_vertices(mesh::Mesh{3, T}, e::Integer) → NTuple{8, SVector{3, T}}

Read the 8 corner coordinates of element `e` from `mesh.vertex_coords`
via `mesh.vertex_idx[:, e]`. Returns an `NTuple{8}` in Gmsh-canonical
tensor-product order.
"""
@inline function element_vertices(mesh::Mesh{3, T}, e::Integer) where {T}
    @inbounds ntuple(v -> begin
        vi = mesh.vertex_idx[v, e]
        SVector{3, T}(mesh.vertex_coords[1, vi],
                      mesh.vertex_coords[2, vi],
                      mesh.vertex_coords[3, vi])
    end, Val(8))
end

"""
    invert_element_map(verts, p; tol, maxiter) → (ξ::SVector{3, T}, ok::Bool)

Newton-iteration inverse of the trilinear corner map. Solves
`trilinear_map(verts, ξ, η, ζ) = p`.

**Deprecated.** No direct replacement: `locate_point` returns the
analytic patch-parameter `ξ` (different semantics from the trilinear
ξ this function recovers). If you specifically need the trilinear
inverse, copy this function into your own code.
"""
function invert_element_map(verts::NTuple{8, SVector{3, T}}, p::SVector{3, T};
                             tol = default_tol(T), maxiter::Int = 20) where {T}
    Base.depwarn(
        "`invert_element_map(verts, p)` (3D) is deprecated; for point→(element, ξ) " *
        "use `locate_point(mesh, p)` which returns the analytic patch-parameter ξ. " *
        "If you specifically need the trilinear inverse there is no direct replacement.",
        :invert_element_map_3d)
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
    locate_point(mesh::Mesh{3, T}, p::SVector{3, T}; tol) → (element_index, ξ)

Analytic point location: identify the patch containing `p` via
`locate_patch`, invert the patch parametric map analytically with
`global_to_patch` to recover patch-local coords, and convert those
to a `(element_index, ξ_in_element)` pair via
`locate_element_in_patch`.

Returns `(0, zero ξ)` if `p` lies outside every patch. The returned
`ξ_in_element` is in the *analytic patch parameter space* — for
curvilinear elements this differs from the trilinear-map ξ, but is
the correct coordinate to feed to Lagrange interpolation against
GLL nodes laid out by `_patch_point_and_jac`.

Cost: `O(npatches)` per call; no Newton iteration.
"""
function locate_point(mesh::Mesh{3, T}, p::SVector{3, T};
                       tol = default_tol(T)) where {T}
    pidx = locate_patch(mesh, p; tol)
    pidx == 0 && return 0, SVector{3, T}(zero(T), zero(T), zero(T))
    ξ_patch = global_to_patch(mesh.patch_desc[pidx], p; tol)
    _is_outside(ξ_patch) && return 0, SVector{3, T}(zero(T), zero(T), zero(T))
    return locate_element_in_patch(mesh, pidx, ξ_patch)
end

# ----- 3D analytic patch ↔ global maps ------------------------------
#
# Per-direction inverse of `_patch_direction_vec(dir, b, c)`. The b/c
# swaps on `-x, +y, -z` (right-handed local frames) carry through to
# the inverse:
#
#   +x: v = ( 1, b, c)   →  f = x,   b = y/x,   c = z/x
#   -x: v = (-1, c, b)   →  f = -x,  c = -y/x,  b = -z/x
#   +y: v = ( c, 1, b)   →  f = y,   c = x/y,   b = z/y
#   -y: v = ( b,-1, c)   →  f = -y,  b = -x/y,  c = -z/y
#   +z: v = ( b, c, 1)   →  f = z,   b = x/z,   c = y/z
#   -z: v = ( c, b,-1)   →  f = -z,  c = -x/z,  b = -y/z
@inline function _inverse_dir_vec_3d(dir::Integer,
                                      x::T, y::T, z::T) where {T}
    if dir == 1                     # +x
        return  x,  y / x,  z / x
    elseif dir == 2                 # -x
        return -x, -z / x, -y / x   # returns (f, b, c) with the b/c swap baked in
    elseif dir == 3                 # +y
        return  y,  z / y,  x / y
    elseif dir == 4                 # -y
        return -y, -x / y, -z / y
    elseif dir == 5                 # +z
        return  z,  x / z,  y / z
    else                            # -z
        return -z, -y / z, -x / z
    end
end

# Per-direction inverse of the *Wedge* forward layout (`_vert_wedge` /
# `_ppj_wedge_3d` / `patch_to_global`'s Wedge branch). Unlike
# `_patch_direction_vec`, the Wedge map keeps its tangential
# coordinates in fixed (x, y, z) order with no b/c swap on the
# −x, +y, −z directions, so it needs its own inverse:
#
#   dir 1 (+x): P = ( r,   b·r, c·r)  →  r =  x,  b =  y/x,  c =  z/x
#   dir 2 (−x): P = (−r,   b·r, c·r)  →  r = −x,  b = −y/x,  c = −z/x
#   dir 3 (+y): P = (b·r,   r,  c·r)  →  r =  y,  b =  x/y,  c =  z/y
#   dir 4 (−y): P = (b·r,  −r,  c·r)  →  r = −y,  b = −x/y,  c = −z/y
#   dir 5 (+z): P = (b·r,  c·r,  r )  →  r =  z,  b =  x/z,  c =  y/z
#   dir 6 (−z): P = (b·r,  c·r, −r )  →  r = −z,  b = −x/z,  c = −y/z
@inline function _inverse_wedge_vec_3d(dir::Integer,
                                        x::T, y::T, z::T) where {T}
    if dir == 1                     # +x
        return  x,  y / x,  z / x
    elseif dir == 2                 # -x
        return -x, -y / x, -z / x
    elseif dir == 3                 # +y
        return  y,  x / y,  z / y
    elseif dir == 4                 # -y
        return -y, -x / y, -z / y
    elseif dir == 5                 # +z
        return  z,  x / z,  y / z
    else                            # -z
        return -z, -x / z, -y / z
    end
end

"""
    patch_to_global(pd::PatchDesc{3, T}, ξ::SVector{3, T}) → SVector{3, T}

Forward parametric map: given a patch descriptor and patch-local
coordinates `ξ ∈ [0, 1]³`, return the physical point `P`. Exact
inverse of [`global_to_patch`](@ref) for that same patch.

The mesh-level overload `patch_to_global(mesh, patch_index, ξ)` is a
thin convenience that forwards to `mesh.patch_desc[patch_index]`.
"""
function patch_to_global(pd::PatchDesc{3, T}, ξ::SVector{3, T}) where {T}
    k = pd.kind
    if k === Cubic
        c = pd.cubic
        return SVector{3, T}(
            c.x_lo[1] + (c.x_hi[1] - c.x_lo[1]) * ξ[1],
            c.x_lo[2] + (c.x_hi[2] - c.x_lo[2]) * ξ[2],
            c.x_lo[3] + (c.x_hi[3] - c.x_lo[3]) * ξ[3])
    elseif k === Inflation
        pi = pd.inflation
        a = pi.a_lo + (pi.a_hi - pi.a_lo) * ξ[1]
        b = pi.b_lo + (pi.b_hi - pi.b_lo) * ξ[2]
        c = pi.c_lo + (pi.c_hi - pi.c_lo) * ξ[3]
        Q = sqrt(one(T) + b * b + c * c)
        f = (one(T) - a) * pi.L + a * pi.R1 / Q
        vx, vy, vz = _patch_direction_vec(pi.dir, b, c)
        return SVector{3, T}(pi.center[1] + f * vx,
                             pi.center[2] + f * vy,
                             pi.center[3] + f * vz)
    elseif k === Shell
        ps = pd.shell
        a = ps.a_lo + (ps.a_hi - ps.a_lo) * ξ[1]
        b = ps.b_lo + (ps.b_hi - ps.b_lo) * ξ[2]
        c = ps.c_lo + (ps.c_hi - ps.c_lo) * ξ[3]
        Q = sqrt(one(T) + b * b + c * c)
        r = (one(T) - a) * ps.R1 + a * ps.R2
        f = r / Q
        vx, vy, vz = _patch_direction_vec(ps.dir, b, c)
        return SVector{3, T}(f * vx, f * vy, f * vz)
    elseif k === Wedge
        w = pd.wedge
        a = w.a_lo + (w.a_hi - w.a_lo) * ξ[1]
        b = w.b_lo + (w.b_hi - w.b_lo) * ξ[2]
        c = w.c_lo + (w.c_hi - w.c_lo) * ξ[3]
        r = w.R1 * (w.R2 / w.R1)^a
        dir = w.dir
        if     dir == Int8(1);  return SVector{3, T}( r,    b * r, c * r)
        elseif dir == Int8(2);  return SVector{3, T}(-r,    b * r, c * r)
        elseif dir == Int8(3);  return SVector{3, T}(b * r,  r,    c * r)
        elseif dir == Int8(4);  return SVector{3, T}(b * r, -r,    c * r)
        elseif dir == Int8(5);  return SVector{3, T}(b * r, c * r,  r)
        else                    return SVector{3, T}(b * r, c * r, -r)
        end
    elseif k === WarpedCubic
        wc = pd.warped_cubic
        q = SVector{3, T}(
            wc.x_lo[1] + (wc.x_hi[1] - wc.x_lo[1]) * ξ[1],
            wc.x_lo[2] + (wc.x_hi[2] - wc.x_lo[2]) * ξ[2],
            wc.x_lo[3] + (wc.x_hi[3] - wc.x_lo[3]) * ξ[3])
        P, _ = _warp_point_and_jac(wc, q)
        return P
    else
        error("patch_to_global: unsupported patch kind $k")
    end
end

# Mesh-level convenience overload. **Deprecated** — pass the PatchDesc
# directly via `mesh.patch_desc[patch_index]`.
Base.@deprecate patch_to_global(mesh::Mesh{3, T}, patch_index::Integer,
                                  ξ::SVector{3, T}) where {T} (
    patch_to_global(mesh.patch_desc[patch_index], ξ))

"""
    global_to_patch(pd::PatchDesc{3, T}, p::SVector{3, T}; tol)
        → SVector{3, T}

Inverse of [`patch_to_global`](@ref). Returns `_outside_xi(Val(3), T)`
if `p` lies outside this patch by more than `tol`. Closed-form for the
Cubic / Inflation / Shell / Wedge kinds; WarpedCubic has no closed-form
inverse and uses a Newton iteration on the forward warp (driven to the
roundoff floor, so still round-trip exact to ≈ machine precision for
points strictly inside the patch).
"""
function global_to_patch(pd::PatchDesc{3, T}, p::SVector{3, T};
                          tol = default_tol(T)) where {T}
    NaN_ξ = _outside_xi(Val(3), T)
    k = pd.kind
    if k === Cubic
        c = pd.cubic
        ξ_a = (p[1] - c.x_lo[1]) / (c.x_hi[1] - c.x_lo[1])
        ξ_b = (p[2] - c.x_lo[2]) / (c.x_hi[2] - c.x_lo[2])
        ξ_c = (p[3] - c.x_lo[3]) / (c.x_hi[3] - c.x_lo[3])
        if -tol ≤ ξ_a ≤ one(T) + tol &&
           -tol ≤ ξ_b ≤ one(T) + tol &&
           -tol ≤ ξ_c ≤ one(T) + tol
            return SVector{3, T}(clamp(ξ_a, zero(T), one(T)),
                                  clamp(ξ_b, zero(T), one(T)),
                                  clamp(ξ_c, zero(T), one(T)))
        end
        return NaN_ξ
    elseif k === Inflation || k === Shell
        # Common closed-form inversion.
        if k === Inflation
            pi = pd.inflation
            dir, a_lo_p, a_hi_p, b_lo_p, b_hi_p, c_lo_p, c_hi_p, L_or_R1, top =
                (pi.dir, pi.a_lo, pi.a_hi, pi.b_lo, pi.b_hi, pi.c_lo, pi.c_hi,
                 pi.L, pi.R1)
            cx, cy, cz = pi.center[1], pi.center[2], pi.center[3]
        else
            ps = pd.shell
            dir, a_lo_p, a_hi_p, b_lo_p, b_hi_p, c_lo_p, c_hi_p, L_or_R1, top =
                (ps.dir, ps.a_lo, ps.a_hi, ps.b_lo, ps.b_hi, ps.c_lo, ps.c_hi,
                 ps.R1, ps.R2)
            cx, cy, cz = zero(T), zero(T), zero(T)
        end
        # Work in the patch's own (centred) frame.
        px = p[1] - cx;  py = p[2] - cy;  pz = p[3] - cz
        f_val, b, c = _inverse_dir_vec_3d(dir, px, py, pz)
        if !(isfinite(b) && isfinite(c) && isfinite(f_val) && f_val > -tol)
            return NaN_ξ
        end
        Q = sqrt(one(T) + b * b + c * c)
        a = if k === Shell
            r = sqrt(px^2 + py^2 + pz^2)
            (r - L_or_R1) / (top - L_or_R1)
        else
            (f_val - L_or_R1) / (top / Q - L_or_R1)
        end
        ξ_a = (a - a_lo_p) / (a_hi_p - a_lo_p)
        ξ_b = (b - b_lo_p) / (b_hi_p - b_lo_p)
        ξ_c = (c - c_lo_p) / (c_hi_p - c_lo_p)
        if -tol ≤ ξ_a ≤ one(T) + tol &&
           -tol ≤ ξ_b ≤ one(T) + tol &&
           -tol ≤ ξ_c ≤ one(T) + tol
            return SVector{3, T}(clamp(ξ_a, zero(T), one(T)),
                                  clamp(ξ_b, zero(T), one(T)),
                                  clamp(ξ_c, zero(T), one(T)))
        end
        return NaN_ξ
    elseif k === Wedge
        w = pd.wedge
        f_val, b, c = _inverse_wedge_vec_3d(w.dir, p[1], p[2], p[3])
        if !(isfinite(b) && isfinite(c) && isfinite(f_val) && f_val > -tol)
            return NaN_ξ
        end
        # f = r = R1·(R2/R1)^a  ⇒  a = log(f / R1) / log(R2 / R1)
        a = log(f_val / w.R1) / log(w.R2 / w.R1)
        ξ_a = (a - w.a_lo) / (w.a_hi - w.a_lo)
        ξ_b = (b - w.b_lo) / (w.b_hi - w.b_lo)
        ξ_c = (c - w.c_lo) / (w.c_hi - w.c_lo)
        if -tol ≤ ξ_a ≤ one(T) + tol &&
           -tol ≤ ξ_b ≤ one(T) + tol &&
           -tol ≤ ξ_c ≤ one(T) + tol
            return SVector{3, T}(clamp(ξ_a, zero(T), one(T)),
                                  clamp(ξ_b, zero(T), one(T)),
                                  clamp(ξ_c, zero(T), one(T)))
        end
        return NaN_ξ
    elseif k === WarpedCubic
        wc = pd.warped_cubic
        # No closed form: the warp `x = q + A·(sinusoidal in q)` is a
        # smooth perturbation of the identity (globally invertible for
        # |A| < min(L_a)/(2π)), so Newton on the forward map converges
        # quadratically from the unwarped Cubic seed `q = p`. Iterate to
        # the roundoff floor (≈ machine precision, matching the
        # closed-form branches); the cap only guards non-invertible
        # (|A| too large) descriptors, for which the residual check
        # below rejects the point.
        q = p
        res = T(Inf)
        res_floor = 8 * eps(T) * max(one(T), abs(p[1]), abs(p[2]), abs(p[3]))
        for _ in 1:40
            x, J = _warp_point_and_jac(wc, q)
            r = x - p
            res = sqrt(r[1]^2 + r[2]^2 + r[3]^2)
            res ≤ res_floor && break
            q = q - (J \ r)
        end
        res ≤ tol || return NaN_ξ
        ξ_a = (q[1] - wc.x_lo[1]) / (wc.x_hi[1] - wc.x_lo[1])
        ξ_b = (q[2] - wc.x_lo[2]) / (wc.x_hi[2] - wc.x_lo[2])
        ξ_c = (q[3] - wc.x_lo[3]) / (wc.x_hi[3] - wc.x_lo[3])
        if -tol ≤ ξ_a ≤ one(T) + tol &&
           -tol ≤ ξ_b ≤ one(T) + tol &&
           -tol ≤ ξ_c ≤ one(T) + tol
            return SVector{3, T}(clamp(ξ_a, zero(T), one(T)),
                                  clamp(ξ_b, zero(T), one(T)),
                                  clamp(ξ_c, zero(T), one(T)))
        end
        return NaN_ξ
    else
        error("global_to_patch: unsupported patch kind $k")
    end
end

# Mesh-level convenience overload. **Deprecated** — pass the PatchDesc
# directly via `mesh.patch_desc[patch_index]`.
Base.@deprecate global_to_patch(mesh::Mesh{3, T}, patch_index::Integer,
                                  p::SVector{3, T};
                                  tol = default_tol(T)) where {T} (
    global_to_patch(mesh.patch_desc[patch_index], p; tol))

"""
    locate_patch(mesh::Mesh{3, T}, p::SVector{3, T}; tol)
        → patch_index :: Int

Find the 1-indexed patch containing `p` by trying `global_to_patch`
on each patch in `mesh.patch_desc` and returning the first that
accepts it. Returns `0` if no patch contains `p`. Walks patches in
storage order, so when patches overlap at their boundaries the
lower-numbered patch wins.
"""
function locate_patch(mesh::Mesh{3, T}, p::SVector{3, T};
                       tol = default_tol(T)) where {T}
    for (p_idx, pd) in enumerate(mesh.patch_desc)
        ξ = global_to_patch(pd, p; tol)
        _is_outside(ξ) && continue
        return p_idx
    end
    return 0
end

"""
    locate_element_in_patch(mesh::Mesh{3, T}, patch_index, ξ_patch)
        → (element_index, ξ_in_element)

Given a patch index and patch-local coordinates `ξ_patch ∈ [0, 1]³`,
return the global element index containing it and the within-element
reference coordinate (also `[0, 1]³`).
"""
function locate_element_in_patch(mesh::Mesh{3, T},
                                  patch_index::Integer,
                                  ξ_patch::SVector{3, T}) where {T}
    pd = mesh.patch_desc[patch_index]
    d = dims(pd)
    s_a = clamp(ξ_patch[1], zero(T), one(T)) * d[1]
    s_b = clamp(ξ_patch[2], zero(T), one(T)) * d[2]
    s_c = clamp(ξ_patch[3], zero(T), one(T)) * d[3]
    a_cell = min(d[1], max(1, floor(Int, s_a) + 1))
    b_cell = min(d[2], max(1, floor(Int, s_b) + 1))
    c_cell = min(d[3], max(1, floor(Int, s_c) + 1))
    ξ_elem_a = s_a - (a_cell - 1)
    ξ_elem_b = s_b - (b_cell - 1)
    ξ_elem_c = s_c - (c_cell - 1)
    elem_off = mesh.patch_element_offset[patch_index]
    e = elem_off + a_cell + d[1] * ((b_cell - 1) + d[2] * (c_cell - 1))
    return e, SVector{3, T}(ξ_elem_a, ξ_elem_b, ξ_elem_c)
end

# 1D Lagrange basis values at `ξ` for nodes `xs`. Length(xs) = N →
# returns NTuple{N, T}. Generic and unoptimised — `O(N²)` per call.
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

# 1D Lagrange basis derivative values at `ξ` for nodes `xs` (product
# rule over the basis factors). O(N²) per call, mirrors lagrange_basis.
function lagrange_basis_deriv(xs, ξ::T) where {T}
    N = length(xs)
    out = ntuple(N) do i
        s = zero(T)
        @inbounds for j in 1:N
            j == i && continue
            p = one(T) / (xs[i] - xs[j])
            for m in 1:N
                (m == i || m == j) && continue
                p *= (ξ - xs[m]) / (xs[i] - xs[m])
            end
            s += p
        end
        s
    end
    return out
end

"""
    tensor_interp_grad(ue, ξ, η, ζ, xs) → (val, ∇ref)

Tensor-product Lagrange interpolation of an `(N, N, N)` per-element
field block at the reference point `(ξ, η, ζ)`, returning both the value
and the reference-coordinate gradient `∇ref :: SVector{3}` (chain-rule
with the element Jacobian — e.g. from [`element_point_and_jac`](@ref) —
turns it into the physical gradient: `∇x f = J⁻ᵀ ∇ref f`).
"""
function tensor_interp_grad(ue::AbstractArray{T, 3},
                            ξ::T, η::T, ζ::T, xs) where {T}
    N = length(xs)
    ℓξ = lagrange_basis(xs, ξ);  dξ = lagrange_basis_deriv(xs, ξ)
    ℓη = lagrange_basis(xs, η);  dη = lagrange_basis_deriv(xs, η)
    ℓζ = lagrange_basis(xs, ζ);  dζ = lagrange_basis_deriv(xs, ζ)
    s = zero(T); g1 = zero(T); g2 = zero(T); g3 = zero(T)
    @inbounds for k in 1:N, j in 1:N, i in 1:N
        u = ue[i, j, k]
        s  += u * ℓξ[i] * ℓη[j] * ℓζ[k]
        g1 += u * dξ[i] * ℓη[j] * ℓζ[k]
        g2 += u * ℓξ[i] * dη[j] * ℓζ[k]
        g3 += u * ℓξ[i] * ℓη[j] * dζ[k]
    end
    return s, SVector{3, T}(g1, g2, g3)
end

"""
    element_point_and_jac(mesh::Mesh{3, T}, e, ξ::SVector{3, T}) → (P, J)

Physical position `P` and element Jacobian `J[i, a] = ∂x_i/∂ξ_a` of the
reference point `ξ ∈ [0, 1]³` of element `e` — analytic for the
curvilinear patch kinds (Inflation/Shell/Wedge/WarpedCubic), trilinear
for Cubic patches. The inverse transpose of `J` maps reference gradients
from [`tensor_interp_grad`](@ref) to physical gradients.
"""
function element_point_and_jac(mesh::Mesh{3, T}, e::Integer,
                               ξ::SVector{3, T}) where {T}
    pd = mesh.patch_desc[mesh.patch_id[e]]
    if pd.kind === Cubic
        verts = element_vertices(mesh, e)
        return trilinear_map(verts, ξ[1], ξ[2], ξ[3]),
               trilinear_jacobian(verts, ξ[1], ξ[2], ξ[3])
    end
    idx = (Int(mesh.patch_idx[1, e]), Int(mesh.patch_idx[2, e]),
           Int(mesh.patch_idx[3, e]))
    return _patch_point_and_jac(pd, idx, ξ[1], ξ[2], ξ[3])
end

"""
    tensor_interp(ue, ξ, η, ζ, xs) → T

Tensor-product Lagrange interpolation of an `(N, N, N)` per-element
field block at the reference point `(ξ, η, ζ)`. `xs` are the 1D GLL
nodes on `[0, 1]`.
"""
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
    interpolate_field(mesh::Mesh{3, T}, xs, u, p; default) → T

Evaluate the per-element field `u` of shape `(N, N, N, Ne)` at the
physical point `p`. Locates `p` analytically via `locate_point`, then
applies tensor-product Lagrange interpolation on the reference-element
GLL nodes `xs`. Returns `default` if `p` lies outside the mesh.
"""
function interpolate_field(mesh::Mesh{3, T},
                            xs::AbstractVector,
                            u::AbstractArray{T, 4},
                            p::SVector{3, T};
                            default = T(NaN)) where {T}
    e, ξ = locate_point(mesh, p)
    e == 0 && return default
    return tensor_interp(view(u, :, :, :, e), ξ[1], ξ[2], ξ[3], xs)
end

# Vectorised convenience: take any iterable of points and return an
# array of values with the same shape.
function interpolate_field(mesh::Mesh{3, T},
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
