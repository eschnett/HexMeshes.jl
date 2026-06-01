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
    else
        error("_patch_point_and_jac: PatchDesc.kind must be Inflation, " *
              "Shell, or WarpedCubic; got $(k). Use the trilinear path " *
              "for Cubic / Wedge.")
    end
end

# WarpedCubic — composition of (a) affine element-reference → patch
# parameter and (b) the warp. Differentiate the composition by hand
# (the affine part is constant) so we don't drag ForwardDiff in here.
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
    a = x_lo_e_1 + d1 * ξ
    b = x_lo_e_2 + d2 * η
    c = x_lo_e_3 + d3 * ζ

    # Warp: x = ξ_patch + A · sin(2π (ξ_patch − x_lo) / L) · [cos(…) for coupled].
    L1 = wc.x_hi[1] - wc.x_lo[1]
    L2 = wc.x_hi[2] - wc.x_lo[2]
    L3 = wc.x_hi[3] - wc.x_lo[3]
    ϕ1 = 2 * pi * (a - wc.x_lo[1]) / L1
    ϕ2 = 2 * pi * (b - wc.x_lo[2]) / L2
    ϕ3 = 2 * pi * (c - wc.x_lo[3]) / L3
    A  = wc.amplitude

    if wc.warp_kind === :diagonal
        Px = a + A * sin(ϕ1)
        Py = b + A * sin(ϕ2)
        Pz = c + A * sin(ϕ3)
        # ∂x_a/∂(patch a) — only diagonal nonzero.
        dPx_da = one(T) + A * cos(ϕ1) * (2 * pi / L1)
        dPy_db = one(T) + A * cos(ϕ2) * (2 * pi / L2)
        dPz_dc = one(T) + A * cos(ϕ3) * (2 * pi / L3)
        # Multiply by patch-coord-vs-reference-coord scaling (d1, d2, d3)
        # to get ∂x/∂ξ_ref.
        J = SMatrix{3, 3, T}(
            dPx_da * d1, zero(T),     zero(T),       # column 1: ∂/∂ξ
            zero(T),     dPy_db * d2, zero(T),       # column 2: ∂/∂η
            zero(T),     zero(T),     dPz_dc * d3)   # column 3: ∂/∂ζ
        return SVector{3, T}(Px, Py, Pz), J
    else
        # :coupled — x_a = ξ_a + A sin(ϕ_a) cos(ϕ_b), b = (a mod 3) + 1.
        s1, s2, s3 = sin(ϕ1), sin(ϕ2), sin(ϕ3)
        c1, c2, c3 = cos(ϕ1), cos(ϕ2), cos(ϕ3)
        # x = a + A s1 c2;  y = b + A s2 c3;  z = c + A s3 c1.
        Px = a + A * s1 * c2
        Py = b + A * s2 * c3
        Pz = c + A * s3 * c1
        k1 = 2 * pi / L1; k2 = 2 * pi / L2; k3 = 2 * pi / L3
        # Patch-coord Jacobian (with respect to a, b, c).
        dPxda = one(T) + A * c1 * c2 * k1
        dPxdb =          -A * s1 * s2 * k2
        # dPxdc = 0
        dPydb = one(T) + A * c2 * c3 * k2
        dPydc =          -A * s2 * s3 * k3
        # dPyda = 0
        dPzdc = one(T) + A * c3 * c1 * k3
        dPzda =          -A * s3 * s1 * k1
        # dPzdb = 0
        # Reference-coord Jacobian: column-major (∂x/∂ξ_ref, scaled).
        J = SMatrix{3, 3, T}(
            dPxda * d1, zero(T),    dPzda * d1,     # column 1: ∂/∂ξ
            dPxdb * d2, dPydb * d2, zero(T),        # column 2: ∂/∂η
            zero(T),    dPydc * d3, dPzdc * d3)     # column 3: ∂/∂ζ
        return SVector{3, T}(Px, Py, Pz), J
    end
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
    else
        error("_patch_point_and_jac_2d: PatchDesc.kind must be Inflation or " *
              "Shell; got $(k). Use the bilinear path for Cubic / Wedge.")
    end
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
