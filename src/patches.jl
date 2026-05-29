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
    Cubic     = 1
    Wedge     = 2
    Inflation = 3
    Shell     = 4
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
embedding `P = f · v(b, c)` with
`f(a) = (1−a)·L + a·R1/Q`, `Q = √(1 + b² + c²)`. Used as the middle
layer of inflated meshes. `c_lo / c_hi` are unused for `D < 3`.
"""
struct PatchInflation{D, T}
    dims :: NTuple{D, Int}
    dir  :: Int8
    a_lo :: T
    a_hi :: T
    b_lo :: T
    b_hi :: T
    c_lo :: T
    c_hi :: T
    L    :: T
    R1   :: T
end

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
                          z, z, z, z, z, z, z, z))
@inline _dummy_shell(::Type{T}, ::Val{D}) where {T, D} =
    (z = zero(T);
     PatchShell{D, T}(ntuple(_ -> 0, Val(D)), Int8(0),
                      z, z, z, z, z, z, z, z))

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
    kind      :: PatchKind
    cubic     :: PatchCubic{D, T}
    wedge     :: PatchWedge{D, T}
    inflation :: PatchInflation{D, T}
    shell     :: PatchShell{D, T}
end

PatchDesc(c::PatchCubic{D, T}) where {D, T} =
    PatchDesc{D, T}(Cubic, c,
                    _dummy_wedge(T, Val(D)),
                    _dummy_inflation(T, Val(D)),
                    _dummy_shell(T, Val(D)))

PatchDesc(w::PatchWedge{D, T}) where {D, T} =
    PatchDesc{D, T}(Wedge,
                    _dummy_cubic(T, Val(D)), w,
                    _dummy_inflation(T, Val(D)),
                    _dummy_shell(T, Val(D)))

PatchDesc(i::PatchInflation{D, T}) where {D, T} =
    PatchDesc{D, T}(Inflation,
                    _dummy_cubic(T, Val(D)),
                    _dummy_wedge(T, Val(D)),
                    i,
                    _dummy_shell(T, Val(D)))

PatchDesc(s::PatchShell{D, T}) where {D, T} =
    PatchDesc{D, T}(Shell,
                    _dummy_cubic(T, Val(D)),
                    _dummy_wedge(T, Val(D)),
                    _dummy_inflation(T, Val(D)),
                    s)

"""
    dims(pd::PatchDesc{D, T}) → NTuple{D, Int}

Element counts along each of the patch's `D` local axes (the
`Ma × Mb × Mc` of the patch's structured grid). Reads from the
active variant.
"""
@inline function dims(pd::PatchDesc{D, T}) where {D, T}
    k = pd.kind
    return k === Cubic     ? pd.cubic.dims     :
           k === Wedge     ? pd.wedge.dims     :
           k === Inflation ? pd.inflation.dims :
                              pd.shell.dims
end

"""
    n_elements(pd::PatchDesc) → Int

Total element count in this patch, `prod(dims(pd))`.
"""
@inline n_elements(pd::PatchDesc) = prod(dims(pd))
