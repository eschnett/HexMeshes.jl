# ----------------------------------------------------------------------
# Patch descriptors for multi-block hex/quad/line meshes.
#
# Every `Mesh{D, T}` carries a vector `patch_desc :: Vector{PatchDesc{D, T}}`
# of length `npatches` describing the parametric geometry of each patch,
# plus per-element `patch_id` / `patch_idx` tables (in `Mesh{D, T}`) that
# locate every element inside its patch's structured (Ma × Mb × Mc) grid.
#
# `PatchDesc{D, T}` is a packed "union": one struct holding all four
# variant payloads in fixed offsets, with an `Int8`-backed `kind` tag
# selecting the active one. The unused variants are zero-initialised.
# Wastes ~3× the per-patch memory but keeps the layout fully concrete
# and GPU-safe (no runtime Union dispatch).
#
# Patch geometric families (`PatchKind`):
#
# * `Cubic`     — axis-aligned `[x_lo, x_hi]^D` box. Used as the inner
#                 cube of multi-block meshes and as the sole patch of
#                 uniform line/quad/hex meshes.
# * `Wedge`     — radial wedge with linear scaling `r(a) = R1·(R2/R1)^a`
#                 and physical map `P = r·(±1, b, c)` (with the right-
#                 handed (b, c) swap convention for −x/+y/−z). Used as
#                 the outer patches of cubed-cube / cubed-square meshes.
# * `Inflation` — radial bridge from an inner cube face to an inner
#                 sphere; `f(a) = (1−a)·L + a·R1/Q` with
#                 `Q = √(1 + b² + c²)`. Used as the middle layer of
#                 inflated-cube / inflated-square meshes.
# * `Shell`     — radial spherical/annular shell;
#                 `f(a) = ((1−a)·R1 + a·R2) / Q`. Used as the outer
#                 layer of inflated meshes.
#
# Direction encoding (`dir :: Int8`, only meaningful for non-cubic patches):
#
#   1 → +x        2 → −x
#   3 → +y        4 → −y     (3D only)
#   5 → +z        6 → −z     (3D only)
#
# The right-handed (b, c)-tangent-frame swap for the −x / +y / −z
# patches lives in `_patch_direction_vec` (kept as the implementation
# detail of patch evaluation, not exposed as separate state).

"""
    PatchKind <: Enum{Int8}

Geometric family of a patch. Dimension-neutral: the same four kinds
apply to 1D / 2D / 3D meshes, though in 1D only `Cubic` is used by
the current builders.
"""
@enum PatchKind::Int8 begin
    Cubic        = 1
    Wedge        = 2
    Inflation    = 3
    Shell        = 4
    WarpedCubic  = 5
    BilinearQuad = 6
    TrilinearHex = 7
end

"""
    PatchCubic{D, T}

Axis-aligned box `[x_lo[d], x_hi[d]]` for `d ∈ 1..D`, divided into
`dims[d]` elements along each axis. No direction or curvature state.
"""
struct PatchCubic{D, T}
    dims :: NTuple{D, Int}
    x_lo :: NTuple{D, T}
    x_hi :: NTuple{D, T}
end

"""
    PatchWedge{D, T}

Radial wedge: `r(a) = R1 · (R2/R1)^a`, physical embedding
`P = (r, b·r, c·r)` (rotated according to `dir`). Used as the outer
patches of cubed-cube / cubed-square meshes (which connect an inner
cube face to a flat outer cube face). `c_lo / c_hi` are unused for
`D < 3`.
"""
struct PatchWedge{D, T}
    dims :: NTuple{D, Int}
    dir  :: Int8
    a_lo :: T
    a_hi :: T
    b_lo :: T
    b_hi :: T
    c_lo :: T
    c_hi :: T
    R1   :: T
    R2   :: T
end

"""
    PatchInflation{D, T}

Radial bridge from an inner cube face to an inner sphere; physical
embedding `P = center + f · v(b, c)` with
`f(a) = (1−a)·L + a·R1/Q`, `Q = √(1 + b² + c²)`. Used as the middle
layer of inflated meshes. `c_lo / c_hi` are unused for `D < 3`.

`center` translates the whole patch (the sphere of radius `R1` and the
square face at distance `L` are both centred there). It defaults to the
origin via the convenience constructor, so origin-centred callers can
omit it. Off-centre inflation is used e.g. by `make_two_hole_mesh`, where
each circular hole sits at `(±d/2, 0)`; there the radial range is also
reversed (`a_lo = 1`, `a_hi = 0`) so the structured radial index runs
circle→square and `det J > 0`.
"""
struct PatchInflation{D, T}
    dims   :: NTuple{D, Int}
    dir    :: Int8
    a_lo   :: T
    a_hi   :: T
    b_lo   :: T
    b_hi   :: T
    c_lo   :: T
    c_hi   :: T
    L      :: T
    R1     :: T
    center :: NTuple{D, T}
end

# Convenience constructor: origin-centred inflation (existing callers).
@inline PatchInflation{D, T}(dims::NTuple{D, Int}, dir, a_lo, a_hi,
                             b_lo, b_hi, c_lo, c_hi, L, R1) where {D, T} =
    PatchInflation{D, T}(dims, Int8(dir), a_lo, a_hi, b_lo, b_hi, c_lo, c_hi,
                         L, R1, ntuple(_ -> zero(T), Val(D)))

"""
    PatchShell{D, T}

Radial spherical/annular shell; physical embedding `P = f · v(b, c)`
with `f(a) = ((1−a)·R1 + a·R2) / Q`. Used as the outer layer of
inflated meshes. `c_lo / c_hi` are unused for `D < 3`.
"""
struct PatchShell{D, T}
    dims :: NTuple{D, Int}
    dir  :: Int8
    a_lo :: T
    a_hi :: T
    b_lo :: T
    b_hi :: T
    c_lo :: T
    c_hi :: T
    R1   :: T
    R2   :: T
end

"""
    PatchWarpedCubic{D, T}

Axis-aligned box like `PatchCubic`, but with a smooth sinusoidal
coordinate transformation applied. Used as a diagnostic mesh for
curvilinear behaviour on a periodic topology — every node has a
non-trivial Jacobian without involving outer boundaries.

`warp_kind` selects the spatial pattern:

* `:diagonal` — `x_a(ξ) = ξ_a + A · sin(2π (ξ_a − x_lo[a]) / L_a)`,
  with `L_a = x_hi[a] − x_lo[a]`. Each axis is decoupled, so `J` is
  diagonal but variable.
* `:coupled` — `x_a(ξ) = ξ_a + A · sin(2π (ξ_a − x_lo[a]) / L_a) ·
  cos(2π (ξ_b − x_lo[b]) / L_b)` with `b = (a mod D) + 1`. `J` is
  full 3×3 (cross-coupled axes).

Both maps satisfy `warp(ξ + L_a · ê_a) = warp(ξ) + L_a · ê_a`, so the
periodic identification of opposite faces in the underlying topology
is preserved. Invertibility requires `|A| < min(L_a) / (2π)`.
"""
struct PatchWarpedCubic{D, T}
    dims      :: NTuple{D, Int}
    x_lo      :: NTuple{D, T}
    x_hi      :: NTuple{D, T}
    amplitude :: T
    warp_kind :: Symbol             # :diagonal or :coupled
end

"""
    PatchBilinearQuad{D, T}

General straight-sided quadrilateral (D = 2) given by its four corner
points, with the bilinear / transfinite-interpolation map

    P(ξ, η) = (1−ξ)(1−η)·c00 + ξ(1−η)·c10 + ξη·c11 + (1−ξ)η·c01,   ξ, η ∈ [0, 1].

`corners = (c00, c10, c11, c01)` in Gmsh-canonical winding (the same
order the mesh stores element corners), i.e. ξ runs `c00→c10` and η runs
`c00→c01`. `dims = (Mξ, Mη)`.

The map restricted to any sub-rectangle is again bilinear, so element
position/Jacobian go through the existing `bilinear_map` /
`bilinear_jacobian` corner path (no analytic-Jacobian branch needed).
Used as the intermediate "butterfly" blocks of `make_two_hole_mesh`; the
first general straight-sided quad patch in the repo.
"""
struct PatchBilinearQuad{D, T}
    dims    :: NTuple{D, Int}
    corners :: NTuple{4, NTuple{D, T}}
end

"""
    PatchTrilinearHex{D, T}

General straight-sided hexahedron (D = 3) given by its eight corner points,
with the trilinear / transfinite-interpolation map. `corners` are in
Gmsh-canonical order (the same order the mesh stores element corners):
`((−,−,−), (+,−,−), (+,+,−), (−,+,−), (−,−,+), (+,−,+), (+,+,+), (−,+,+))`, so
ξ runs corner 1→2, η runs 1→4, ζ runs 1→5. `dims = (Mξ, Mη, Mζ)`.

The 3D analog of [`PatchBilinearQuad`](@ref): a trilinear map restricted to
any sub-box is again trilinear, so element position/Jacobian go through the
existing `trilinear_map` / `trilinear_jacobian` corner path (no analytic-
Jacobian branch needed). Used as the intermediate frustum blocks of
`make_two_ball_mesh`.
"""
struct PatchTrilinearHex{D, T}
    dims    :: NTuple{D, Int}
    corners :: NTuple{8, NTuple{D, T}}
end

# Internal zero-initialised dummy patches used to fill the unused
# variants of `PatchDesc`. Not exported.
@inline _dummy_cubic(::Type{T}, ::Val{D}) where {T, D} =
    PatchCubic{D, T}(ntuple(_ -> 0, Val(D)),
                     ntuple(_ -> zero(T), Val(D)),
                     ntuple(_ -> zero(T), Val(D)))
@inline _dummy_wedge(::Type{T}, ::Val{D}) where {T, D} =
    (z = zero(T);
     PatchWedge{D, T}(ntuple(_ -> 0, Val(D)), Int8(0),
                      z, z, z, z, z, z, z, z))
@inline _dummy_inflation(::Type{T}, ::Val{D}) where {T, D} =
    (z = zero(T);
     PatchInflation{D, T}(ntuple(_ -> 0, Val(D)), Int8(0),
                          z, z, z, z, z, z, z, z,
                          ntuple(_ -> z, Val(D))))
@inline _dummy_shell(::Type{T}, ::Val{D}) where {T, D} =
    (z = zero(T);
     PatchShell{D, T}(ntuple(_ -> 0, Val(D)), Int8(0),
                      z, z, z, z, z, z, z, z))
@inline _dummy_warped_cubic(::Type{T}, ::Val{D}) where {T, D} =
    PatchWarpedCubic{D, T}(ntuple(_ -> 0, Val(D)),
                            ntuple(_ -> zero(T), Val(D)),
                            ntuple(_ -> zero(T), Val(D)),
                            zero(T), :diagonal)
@inline _dummy_bilinear_quad(::Type{T}, ::Val{D}) where {T, D} =
    PatchBilinearQuad{D, T}(ntuple(_ -> 0, Val(D)),
                            ntuple(_ -> ntuple(_ -> zero(T), Val(D)), Val(4)))
@inline _dummy_trilinear_hex(::Type{T}, ::Val{D}) where {T, D} =
    PatchTrilinearHex{D, T}(ntuple(_ -> 0, Val(D)),
                            ntuple(_ -> ntuple(_ -> zero(T), Val(D)), Val(8)))

"""
    PatchDesc{D, T}

Packed "tagged union" carrying one of `PatchCubic{D, T}` /
`PatchWedge{D, T}` / `PatchInflation{D, T}` / `PatchShell{D, T}`. The
`kind` field selects the active variant; the other three fields are
zero-initialised. Construct with one of the variant-specific
constructors:

    PatchDesc(c::PatchCubic{D, T})
    PatchDesc(w::PatchWedge{D, T})
    PatchDesc(i::PatchInflation{D, T})
    PatchDesc(s::PatchShell{D, T})

Access by branching on `pd.kind`:

    if pd.kind === Cubic
        c = pd.cubic
        ...
    elseif pd.kind === Shell
        s = pd.shell
        ...
    end

Use [`dims`](@ref) / [`n_elements`](@ref) to read the active variant's
structured-grid extent without manual branching.
"""
struct PatchDesc{D, T}
    kind          :: PatchKind
    cubic         :: PatchCubic{D, T}
    wedge         :: PatchWedge{D, T}
    inflation     :: PatchInflation{D, T}
    shell         :: PatchShell{D, T}
    warped_cubic  :: PatchWarpedCubic{D, T}
    bilinear_quad :: PatchBilinearQuad{D, T}
    trilinear_hex :: PatchTrilinearHex{D, T}
end

PatchDesc(c::PatchCubic{D, T}) where {D, T} =
    PatchDesc{D, T}(Cubic, c,
                    _dummy_wedge(T, Val(D)),
                    _dummy_inflation(T, Val(D)),
                    _dummy_shell(T, Val(D)),
                    _dummy_warped_cubic(T, Val(D)),
                    _dummy_bilinear_quad(T, Val(D)),
                    _dummy_trilinear_hex(T, Val(D)))

PatchDesc(w::PatchWedge{D, T}) where {D, T} =
    PatchDesc{D, T}(Wedge,
                    _dummy_cubic(T, Val(D)), w,
                    _dummy_inflation(T, Val(D)),
                    _dummy_shell(T, Val(D)),
                    _dummy_warped_cubic(T, Val(D)),
                    _dummy_bilinear_quad(T, Val(D)),
                    _dummy_trilinear_hex(T, Val(D)))

PatchDesc(i::PatchInflation{D, T}) where {D, T} =
    PatchDesc{D, T}(Inflation,
                    _dummy_cubic(T, Val(D)),
                    _dummy_wedge(T, Val(D)),
                    i,
                    _dummy_shell(T, Val(D)),
                    _dummy_warped_cubic(T, Val(D)),
                    _dummy_bilinear_quad(T, Val(D)),
                    _dummy_trilinear_hex(T, Val(D)))

PatchDesc(s::PatchShell{D, T}) where {D, T} =
    PatchDesc{D, T}(Shell,
                    _dummy_cubic(T, Val(D)),
                    _dummy_wedge(T, Val(D)),
                    _dummy_inflation(T, Val(D)),
                    s,
                    _dummy_warped_cubic(T, Val(D)),
                    _dummy_bilinear_quad(T, Val(D)),
                    _dummy_trilinear_hex(T, Val(D)))

PatchDesc(wc::PatchWarpedCubic{D, T}) where {D, T} =
    PatchDesc{D, T}(WarpedCubic,
                    _dummy_cubic(T, Val(D)),
                    _dummy_wedge(T, Val(D)),
                    _dummy_inflation(T, Val(D)),
                    _dummy_shell(T, Val(D)),
                    wc,
                    _dummy_bilinear_quad(T, Val(D)),
                    _dummy_trilinear_hex(T, Val(D)))

PatchDesc(bq::PatchBilinearQuad{D, T}) where {D, T} =
    PatchDesc{D, T}(BilinearQuad,
                    _dummy_cubic(T, Val(D)),
                    _dummy_wedge(T, Val(D)),
                    _dummy_inflation(T, Val(D)),
                    _dummy_shell(T, Val(D)),
                    _dummy_warped_cubic(T, Val(D)),
                    bq,
                    _dummy_trilinear_hex(T, Val(D)))

PatchDesc(th::PatchTrilinearHex{D, T}) where {D, T} =
    PatchDesc{D, T}(TrilinearHex,
                    _dummy_cubic(T, Val(D)),
                    _dummy_wedge(T, Val(D)),
                    _dummy_inflation(T, Val(D)),
                    _dummy_shell(T, Val(D)),
                    _dummy_warped_cubic(T, Val(D)),
                    _dummy_bilinear_quad(T, Val(D)),
                    th)

"""
    dims(pd::PatchDesc{D, T}) → NTuple{D, Int}

Element counts along each of the patch's `D` local axes (the
`Ma × Mb × Mc` of the patch's structured grid). Reads from the
active variant.
"""
@inline function dims(pd::PatchDesc{D, T}) where {D, T}
    k = pd.kind
    return k === Cubic        ? pd.cubic.dims         :
           k === Wedge        ? pd.wedge.dims         :
           k === Inflation    ? pd.inflation.dims     :
           k === Shell        ? pd.shell.dims         :
           k === WarpedCubic  ? pd.warped_cubic.dims  :
           k === BilinearQuad ? pd.bilinear_quad.dims :
                                pd.trilinear_hex.dims
end

"""
    n_elements(pd::PatchDesc) → Int

Total element count in this patch, `prod(dims(pd))`.
"""
@inline n_elements(pd::PatchDesc) = prod(dims(pd))
