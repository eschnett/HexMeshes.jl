# ----------------------------------------------------------------------
# Analytic per-element position + Jacobian for curvilinear patches.
#
# Used by downstream `make_geometry` to populate per-node coordinates,
# Jacobians, and `det J` on `Inflation` / `Shell` patches without
# resorting to the trilinear/bilinear corner interpolation that's used
# for `Cubic` and `Wedge` patches.
#
# This file contains both the 3D and 2D variants. They share the
# `_elem_sub_range` helper that maps the per-element reference
# coordinate `[0, 1]` to the patch's parameter sub-range, plus the
# per-variant `_ppj_*` workers that compose the patch parametric map
# with its chain-rule Jacobian.

"""
    _elem_sub_range(p_lo, p_hi, idx, dims) → (lo, hi)

Compute the per-element parameter sub-range along one axis. The patch
spans `[p_lo, p_hi]` divided into `dims` cells; element `idx ∈ 1..dims`
occupies `[lo, hi]` of that range.
"""
@inline function _elem_sub_range(p_lo::T, p_hi::T, idx::Integer, dims::Integer) where {T}
    Δ = p_hi - p_lo
    lo = p_lo + Δ * T(idx - 1) / T(dims)
    hi = p_lo + Δ * T(idx)     / T(dims)
    return lo, hi
end

# ----- 3D ------------------------------------------------------------

"""
    _patch_point_and_jac(pd::PatchDesc{3, T},
                          idx::NTuple{3, <:Integer},
                          ξ, η, ζ) → (P::SVector{3, T}, J::SMatrix{3, 3, T})

Analytic position + reference-cube Jacobian for one node of an
Inflation- or Shell-kind patch. `idx` is the 1-indexed position of the
element within `pd`'s structured grid; `(ξ, η, ζ) ∈ [0, 1]³` is the
reference-element coordinate. The per-element parameter sub-range is
computed inline from the patch's parameter range and `pd.dims`.

Errors if `pd.kind === Cubic` or `Wedge`; the caller (`make_geometry`)
should use the trilinear path for those.
"""
@inline function _patch_point_and_jac(pd::PatchDesc{3, T},
                                       idx::NTuple{3, <:Integer},
                                       ξ::T, η::T, ζ::T) where {T}
    k = pd.kind
    if k === Inflation
        return _ppj_inflation_3d(pd.inflation, idx, ξ, η, ζ)
    elseif k === Shell
        return _ppj_shell_3d(pd.shell, idx, ξ, η, ζ)
    elseif k === WarpedCubic
        return _ppj_warped_cubic_3d(pd.warped_cubic, idx, ξ, η, ζ)
    elseif k === Wedge
        return _ppj_wedge_3d(pd.wedge, idx, ξ, η, ζ)
    else
        error("_patch_point_and_jac: PatchDesc.kind must be Inflation, " *
              "Shell, WarpedCubic, or Wedge; got $(k). Use the trilinear " *
              "path for Cubic.")
    end
end

# Analytic per-node position + Jacobian for a `PatchWedge{3, T}`.
# Parametric map (matches `_vert_wedge` 3D in skeleton.jl):
#   r(a) = R1 · (R2/R1)^a
#   dir = 1:  P = ( r,    b·r,  c·r)
#   dir = 2:  P = (−r,    b·r,  c·r)
#   dir = 3:  P = ( b·r,   r,   c·r)
#   dir = 4:  P = ( b·r, −r,    c·r)
#   dir = 5:  P = ( b·r,  c·r,   r)
#   dir = 6:  P = ( b·r,  c·r, −r)
# Element-reference Jacobian: differentiate w.r.t. (a, b, c) and scale
# each column by (da, db, dc) = element parameter widths.
@inline function _ppj_wedge_3d(w::PatchWedge{3, T},
                                 idx::NTuple{3, <:Integer},
                                 ξ::T, η::T, ζ::T) where {T}
    a_lo, a_hi = _elem_sub_range(w.a_lo, w.a_hi, idx[1], w.dims[1])
    b_lo, b_hi = _elem_sub_range(w.b_lo, w.b_hi, idx[2], w.dims[2])
    c_lo, c_hi = _elem_sub_range(w.c_lo, w.c_hi, idx[3], w.dims[3])

    da = a_hi - a_lo;  db = b_hi - b_lo;  dc = c_hi - c_lo
    a  = a_lo + da * ξ
    b  = b_lo + db * η
    c  = c_lo + dc * ζ

    α     = w.R2 / w.R1
    r     = w.R1 * α^a
    log_α = log(α)
    dr_da = r * log_α

    dir = w.dir
    # (Px, Py, Pz) and the three columns of the Jacobian w.r.t. (a, b, c).
    if dir == Int8(1)
        # P = (r, b·r, c·r)
        Px = r;          Py = b * r;      Pz = c * r
        dPa_x = dr_da;   dPa_y = b * dr_da; dPa_z = c * dr_da
        dPb_x = zero(T); dPb_y = r;       dPb_z = zero(T)
        dPc_x = zero(T); dPc_y = zero(T); dPc_z = r
    elseif dir == Int8(2)
        # P = (−r, b·r, c·r)
        Px = -r;         Py = b * r;      Pz = c * r
        dPa_x = -dr_da;  dPa_y = b * dr_da; dPa_z = c * dr_da
        dPb_x = zero(T); dPb_y = r;       dPb_z = zero(T)
        dPc_x = zero(T); dPc_y = zero(T); dPc_z = r
    elseif dir == Int8(3)
        # P = (b·r, r, c·r)
        Px = b * r;      Py = r;          Pz = c * r
        dPa_x = b * dr_da; dPa_y = dr_da; dPa_z = c * dr_da
        dPb_x = r;       dPb_y = zero(T); dPb_z = zero(T)
        dPc_x = zero(T); dPc_y = zero(T); dPc_z = r
    elseif dir == Int8(4)
        # P = (b·r, −r, c·r)
        Px = b * r;      Py = -r;         Pz = c * r
        dPa_x = b * dr_da; dPa_y = -dr_da; dPa_z = c * dr_da
        dPb_x = r;       dPb_y = zero(T); dPb_z = zero(T)
        dPc_x = zero(T); dPc_y = zero(T); dPc_z = r
    elseif dir == Int8(5)
        # P = (b·r, c·r, r)
        Px = b * r;      Py = c * r;      Pz = r
        dPa_x = b * dr_da; dPa_y = c * dr_da; dPa_z = dr_da
        dPb_x = r;       dPb_y = zero(T); dPb_z = zero(T)
        dPc_x = zero(T); dPc_y = r;       dPc_z = zero(T)
    else  # dir == 6
        # P = (b·r, c·r, −r)
        Px = b * r;      Py = c * r;      Pz = -r
        dPa_x = b * dr_da; dPa_y = c * dr_da; dPa_z = -dr_da
        dPb_x = r;       dPb_y = zero(T); dPb_z = zero(T)
        dPc_x = zero(T); dPc_y = r;       dPc_z = zero(T)
    end

    J = SMatrix{3, 3, T}(
        dPa_x * da, dPa_y * da, dPa_z * da,    # column 1: ∂/∂ξ_ref
        dPb_x * db, dPb_y * db, dPb_z * db,    # column 2: ∂/∂η_ref
        dPc_x * dc, dPc_y * dc, dPc_z * dc)    # column 3: ∂/∂ζ_ref
    P = SVector{3, T}(Px, Py, Pz)
    return P, J
end

# Warp value + patch-coordinate Jacobian `∂x/∂q` of a 3D WarpedCubic
# patch at the *patch* coordinate `q` (not element-reference coords).
# Differentiated by hand so we don't drag ForwardDiff in here. Shared
# by `_ppj_warped_cubic_3d` (which composes it with the constant affine
# element sub-range) and by the Newton inversion in `global_to_patch`'s
# WarpedCubic branch (queries.jl).
@inline function _warp_point_and_jac(wc::PatchWarpedCubic{3, T},
                                       q::SVector{3, T}) where {T}
    # Warp: x = q + A · sin(2π (q − x_lo) / L) · [cos(…) for coupled].
    L1 = wc.x_hi[1] - wc.x_lo[1]
    L2 = wc.x_hi[2] - wc.x_lo[2]
    L3 = wc.x_hi[3] - wc.x_lo[3]
    ϕ1 = 2 * pi * (q[1] - wc.x_lo[1]) / L1
    ϕ2 = 2 * pi * (q[2] - wc.x_lo[2]) / L2
    ϕ3 = 2 * pi * (q[3] - wc.x_lo[3]) / L3
    A  = wc.amplitude

    if wc.warp_kind === :diagonal
        Px = q[1] + A * sin(ϕ1)
        Py = q[2] + A * sin(ϕ2)
        Pz = q[3] + A * sin(ϕ3)
        # ∂x_a/∂q_a — only diagonal nonzero.
        dPx_da = one(T) + A * cos(ϕ1) * (2 * pi / L1)
        dPy_db = one(T) + A * cos(ϕ2) * (2 * pi / L2)
        dPz_dc = one(T) + A * cos(ϕ3) * (2 * pi / L3)
        J = SMatrix{3, 3, T}(
            dPx_da,  zero(T), zero(T),     # column 1: ∂/∂q₁
            zero(T), dPy_db,  zero(T),     # column 2: ∂/∂q₂
            zero(T), zero(T), dPz_dc)      # column 3: ∂/∂q₃
        return SVector{3, T}(Px, Py, Pz), J
    else
        # :coupled — x_a = q_a + A sin(ϕ_a) cos(ϕ_b), b = (a mod 3) + 1.
        s1, s2, s3 = sin(ϕ1), sin(ϕ2), sin(ϕ3)
        c1, c2, c3 = cos(ϕ1), cos(ϕ2), cos(ϕ3)
        # x = q₁ + A s1 c2;  y = q₂ + A s2 c3;  z = q₃ + A s3 c1.
        Px = q[1] + A * s1 * c2
        Py = q[2] + A * s2 * c3
        Pz = q[3] + A * s3 * c1
        k1 = 2 * pi / L1; k2 = 2 * pi / L2; k3 = 2 * pi / L3
        dPxda = one(T) + A * c1 * c2 * k1
        dPxdb =          -A * s1 * s2 * k2
        # dPxdc = 0
        dPydb = one(T) + A * c2 * c3 * k2
        dPydc =          -A * s2 * s3 * k3
        # dPyda = 0
        dPzdc = one(T) + A * c3 * c1 * k3
        dPzda =          -A * s3 * s1 * k1
        # dPzdb = 0
        J = SMatrix{3, 3, T}(
            dPxda,   zero(T), dPzda,       # column 1: ∂/∂q₁
            dPxdb,   dPydb,   zero(T),     # column 2: ∂/∂q₂
            zero(T), dPydc,   dPzdc)       # column 3: ∂/∂q₃
        return SVector{3, T}(Px, Py, Pz), J
    end
end

# WarpedCubic — composition of (a) affine element-reference → patch
# parameter and (b) the warp (`_warp_point_and_jac`). The affine part
# has constant Jacobian (d1, d2, d3), so the chain rule is a per-column
# scaling.
@inline function _ppj_warped_cubic_3d(wc::PatchWarpedCubic{3, T},
                                        idx::NTuple{3, <:Integer},
                                        ξ::T, η::T, ζ::T) where {T}
    # Element-local sub-range in patch coords.
    x_lo_e_1, x_hi_e_1 = _elem_sub_range(wc.x_lo[1], wc.x_hi[1], idx[1], wc.dims[1])
    x_lo_e_2, x_hi_e_2 = _elem_sub_range(wc.x_lo[2], wc.x_hi[2], idx[2], wc.dims[2])
    x_lo_e_3, x_hi_e_3 = _elem_sub_range(wc.x_lo[3], wc.x_hi[3], idx[3], wc.dims[3])
    d1 = x_hi_e_1 - x_lo_e_1
    d2 = x_hi_e_2 - x_lo_e_2
    d3 = x_hi_e_3 - x_lo_e_3
    # Patch-space coord at this reference point.
    q = SVector{3, T}(x_lo_e_1 + d1 * ξ,
                      x_lo_e_2 + d2 * η,
                      x_lo_e_3 + d3 * ζ)
    P, Jq = _warp_point_and_jac(wc, q)
    # Multiply each patch-coord Jacobian column by the element width
    # (d1, d2, d3) to get ∂x/∂ξ_ref.
    J = SMatrix{3, 3, T}(
        Jq[1, 1] * d1, Jq[2, 1] * d1, Jq[3, 1] * d1,   # column 1: ∂/∂ξ
        Jq[1, 2] * d2, Jq[2, 2] * d2, Jq[3, 2] * d2,   # column 2: ∂/∂η
        Jq[1, 3] * d3, Jq[2, 3] * d3, Jq[3, 3] * d3)   # column 3: ∂/∂ζ
    return P, J
end

@inline function _ppj_inflation_3d(pi::PatchInflation{3, T},
                                     idx::NTuple{3, <:Integer},
                                     ξ::T, η::T, ζ::T) where {T}
    a_lo, a_hi = _elem_sub_range(pi.a_lo, pi.a_hi, idx[1], pi.dims[1])
    b_lo, b_hi = _elem_sub_range(pi.b_lo, pi.b_hi, idx[2], pi.dims[2])
    c_lo, c_hi = _elem_sub_range(pi.c_lo, pi.c_hi, idx[3], pi.dims[3])

    da = a_hi - a_lo;  db = b_hi - b_lo;  dc = c_hi - c_lo
    a  = a_lo + da * ξ
    b  = b_lo + db * η
    c  = c_lo + dc * ζ

    Q  = sqrt(one(T) + (b * b + c * c))
    Q3 = Q * Q * Q
    L  = pi.L;  R1 = pi.R1
    f     = (one(T) - a) * L + a * R1 / Q
    df_da = -L + R1 / Q
    df_db = -a * R1 * b / Q3
    df_dc = -a * R1 * c / Q3

    vx, vy, vz, dvxb, dvyb, dvzb, dvxc, dvyc, dvzc =
        _patch_direction_vec_and_derivs(pi.dir, b, c)

    Px = f * vx;  Py = f * vy;  Pz = f * vz
    dPa_x = df_da * vx;            dPa_y = df_da * vy;            dPa_z = df_da * vz
    dPb_x = df_db * vx + f * dvxb; dPb_y = df_db * vy + f * dvyb; dPb_z = df_db * vz + f * dvzb
    dPc_x = df_dc * vx + f * dvxc; dPc_y = df_dc * vy + f * dvyc; dPc_z = df_dc * vz + f * dvzc

    J = SMatrix{3, 3, T}(
        dPa_x * da, dPa_y * da, dPa_z * da,
        dPb_x * db, dPb_y * db, dPb_z * db,
        dPc_x * dc, dPc_y * dc, dPc_z * dc)
    P = SVector{3, T}(Px, Py, Pz)
    return P, J
end

@inline function _ppj_shell_3d(ps::PatchShell{3, T},
                                 idx::NTuple{3, <:Integer},
                                 ξ::T, η::T, ζ::T) where {T}
    a_lo, a_hi = _elem_sub_range(ps.a_lo, ps.a_hi, idx[1], ps.dims[1])
    b_lo, b_hi = _elem_sub_range(ps.b_lo, ps.b_hi, idx[2], ps.dims[2])
    c_lo, c_hi = _elem_sub_range(ps.c_lo, ps.c_hi, idx[3], ps.dims[3])

    da = a_hi - a_lo;  db = b_hi - b_lo;  dc = c_hi - c_lo
    a  = a_lo + da * ξ
    b  = b_lo + db * η
    c  = c_lo + dc * ζ

    Q  = sqrt(one(T) + (b * b + c * c))
    Q3 = Q * Q * Q
    R1 = ps.R1;  R2 = ps.R2
    rval  = (one(T) - a) * R1 + a * R2
    f     = rval / Q
    df_da = (R2 - R1) / Q
    df_db = -rval * b / Q3
    df_dc = -rval * c / Q3

    vx, vy, vz, dvxb, dvyb, dvzb, dvxc, dvyc, dvzc =
        _patch_direction_vec_and_derivs(ps.dir, b, c)

    Px = f * vx;  Py = f * vy;  Pz = f * vz
    dPa_x = df_da * vx;            dPa_y = df_da * vy;            dPa_z = df_da * vz
    dPb_x = df_db * vx + f * dvxb; dPb_y = df_db * vy + f * dvyb; dPb_z = df_db * vz + f * dvzb
    dPc_x = df_dc * vx + f * dvxc; dPc_y = df_dc * vy + f * dvyc; dPc_z = df_dc * vz + f * dvzc

    J = SMatrix{3, 3, T}(
        dPa_x * da, dPa_y * da, dPa_z * da,
        dPb_x * db, dPb_y * db, dPb_z * db,
        dPc_x * dc, dPc_y * dc, dPc_z * dc)
    P = SVector{3, T}(Px, Py, Pz)
    return P, J
end

# ----- 2D ------------------------------------------------------------

"""
    _patch_point_and_jac_2d(pd::PatchDesc{2, T},
                              idx::NTuple{2, <:Integer},
                              ξ, η) → (P::SVector{2, T}, J::SMatrix{2, 2, T})

Analytic position + reference-square Jacobian for one node of an
Inflation- or Shell-kind 2D patch. Errors if `pd.kind` is `Cubic` or
`Wedge` (use the bilinear path).
"""
@inline function _patch_point_and_jac_2d(pd::PatchDesc{2, T},
                                          idx::NTuple{2, <:Integer},
                                          ξ::T, η::T) where {T}
    k = pd.kind
    if k === Inflation
        return _ppj_inflation_2d(pd.inflation, idx, ξ, η)
    elseif k === Shell
        return _ppj_shell_2d(pd.shell, idx, ξ, η)
    elseif k === Wedge
        return _ppj_wedge_2d(pd.wedge, idx, ξ, η)
    else
        error("_patch_point_and_jac_2d: PatchDesc.kind must be Inflation, " *
              "Shell, or Wedge; got $(k). Use the bilinear path for Cubic.")
    end
end

# Analytic per-node position + Jacobian for a `PatchWedge{2, T}`.
# Parametric map (matches `_vert_wedge` 2D in skeleton.jl):
#   r(a) = R1 · (R2/R1)^a
#   dir = 1 (+x):  P = ( r,    b·r)
#   dir = 2 (−x):  P = (−r,    b·r)
#   dir = 3 (+y):  P = ( b·r,   r)
#   dir = 4 (−y):  P = ( b·r, −r)
@inline function _ppj_wedge_2d(w::PatchWedge{2, T},
                                 idx::NTuple{2, <:Integer},
                                 ξ::T, η::T) where {T}
    a_lo, a_hi = _elem_sub_range(w.a_lo, w.a_hi, idx[1], w.dims[1])
    b_lo, b_hi = _elem_sub_range(w.b_lo, w.b_hi, idx[2], w.dims[2])

    da = a_hi - a_lo;  db = b_hi - b_lo
    a  = a_lo + da * ξ
    b  = b_lo + db * η

    α     = w.R2 / w.R1
    r     = w.R1 * α^a
    log_α = log(α)
    dr_da = r * log_α

    dir = w.dir
    if dir == Int8(1)
        # P = (r, b·r)
        Px = r;          Py = b * r
        dPa_x = dr_da;   dPa_y = b * dr_da
        dPb_x = zero(T); dPb_y = r
    elseif dir == Int8(2)
        # P = (−r, b·r)
        Px = -r;         Py = b * r
        dPa_x = -dr_da;  dPa_y = b * dr_da
        dPb_x = zero(T); dPb_y = r
    elseif dir == Int8(3)
        # P = (b·r, r)
        Px = b * r;      Py = r
        dPa_x = b * dr_da; dPa_y = dr_da
        dPb_x = r;       dPb_y = zero(T)
    else  # dir == 4
        # P = (b·r, −r)
        Px = b * r;      Py = -r
        dPa_x = b * dr_da; dPa_y = -dr_da
        dPb_x = r;       dPb_y = zero(T)
    end

    J = SMatrix{2, 2, T}(dPa_x * da, dPa_y * da,
                          dPb_x * db, dPb_y * db)
    P = SVector{2, T}(Px, Py)
    return P, J
end

@inline function _ppj_inflation_2d(pi::PatchInflation{2, T},
                                     idx::NTuple{2, <:Integer},
                                     ξ::T, η::T) where {T}
    a_lo, a_hi = _elem_sub_range(pi.a_lo, pi.a_hi, idx[1], pi.dims[1])
    b_lo, b_hi = _elem_sub_range(pi.b_lo, pi.b_hi, idx[2], pi.dims[2])

    da = a_hi - a_lo;  db = b_hi - b_lo
    a  = a_lo + da * ξ
    b  = b_lo + db * η

    Q  = sqrt(one(T) + b * b)
    Q3 = Q * Q * Q
    L  = pi.L;  R1 = pi.R1
    f     = (one(T) - a) * L + a * R1 / Q
    df_da = -L + R1 / Q
    df_db = -a * R1 * b / Q3

    vx, vy, dvxb, dvyb = _patch_direction_vec_2d_and_derivs(pi.dir, b)

    Px = f * vx;  Py = f * vy
    dPa_x = df_da * vx;            dPa_y = df_da * vy
    dPb_x = df_db * vx + f * dvxb; dPb_y = df_db * vy + f * dvyb

    J = SMatrix{2, 2, T}(dPa_x * da, dPa_y * da,
                         dPb_x * db, dPb_y * db)
    P = SVector{2, T}(Px, Py)
    return P, J
end

@inline function _ppj_shell_2d(ps::PatchShell{2, T},
                                 idx::NTuple{2, <:Integer},
                                 ξ::T, η::T) where {T}
    a_lo, a_hi = _elem_sub_range(ps.a_lo, ps.a_hi, idx[1], ps.dims[1])
    b_lo, b_hi = _elem_sub_range(ps.b_lo, ps.b_hi, idx[2], ps.dims[2])

    da = a_hi - a_lo;  db = b_hi - b_lo
    a  = a_lo + da * ξ
    b  = b_lo + db * η

    Q  = sqrt(one(T) + b * b)
    Q3 = Q * Q * Q
    R1 = ps.R1;  R2 = ps.R2
    rval  = (one(T) - a) * R1 + a * R2
    f     = rval / Q
    df_da = (R2 - R1) / Q
    df_db = -rval * b / Q3

    vx, vy, dvxb, dvyb = _patch_direction_vec_2d_and_derivs(ps.dir, b)

    Px = f * vx;  Py = f * vy
    dPa_x = df_da * vx;            dPa_y = df_da * vy
    dPb_x = df_db * vx + f * dvxb; dPb_y = df_db * vy + f * dvyb

    J = SMatrix{2, 2, T}(dPa_x * da, dPa_y * da,
                         dPb_x * db, dPb_y * db)
    P = SVector{2, T}(Px, Py)
    return P, J
end
