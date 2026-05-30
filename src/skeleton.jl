# ----------------------------------------------------------------------
# Skeleton-based mesh construction for `Mesh{D, T}` with `D ∈ {2, 3}`.
#
# Two-stage build:
#
# Stage 1 — caller assembles a `SkeletonMesh{D, T}` carrying
#   * a list of patches as `Vector{PatchDesc{D, T}}`, and
#   * an inter-patch face-connectivity table `Matrix{FaceLink}` of
#     shape `(2D, npatches)` saying which patch face borders which
#     (or is on the domain boundary).
#   No floating-point coordinates appear at this stage; everything is
#   combinatorial.
#
# Stage 2 — `_skeleton_to_mesh(skel)` walks the skeleton, enumerates
#   per-patch vertices as integer tuples `(p, idx...)`, unifies
#   face-shared ids via union-find, assigns dense canonical ids,
#   evaluates the per-patch parametric maps to produce coordinates,
#   and builds the per-element neighbour / orientation / patch-info
#   tables. Returns a fully-formed `Mesh{D, T}` with all four patch
#   fields populated.
#
# Integer-only dedup at stage 2 eliminates the ULP-difference problems
# the earlier `Dict{NTuple{D, T}, Int}`-based dedup had for non-power-
# of-2 divisions.
#
# `SkeletonMesh{D, T}` and `FaceLink` are internal-only build-time
# scaffolding; they are not exported from the package. The
# `D = 1` case (uniform line meshes) doesn't go through the skeleton —
# `make_line_mesh` populates the mesh fields directly.

# ----- Direction-dependent unit-ish vectors --------------------------
#
# Used by Inflation and Shell patch parametric maps. The b/c-flip
# pattern for the `-x` / `+y` / `-z` directions is the right-handed-
# frame swap that makes the local `(a, b, c)` frame have `det J > 0`
# in physical space everywhere. Divided by `Q = √(1 + b² + c²)` this
# gives the unit radial direction from origin out to the patch face.

"""
    _patch_direction_vec_2d(dir::Integer, b) → (vx, vy)

Direction-dependent unit-ish vector for 2D inflation / shell patches.

    +x:  v = ( 1,  b)              −x:  v = (-1, -b)
    +y:  v = (-b,  1)              −y:  v = ( b, -1)
"""
@inline function _patch_direction_vec_2d(dir::Integer, b::T) where {T<:Real}
    o = one(T)
    if     dir == 1;  return ( o,  b)
    elseif dir == 2;  return (-o, -b)
    elseif dir == 3;  return (-b,  o)
    else              return ( b, -o)
    end
end

"""
    _patch_direction_vec_2d_and_derivs(dir::Integer, b) → (vx, vy, dvx_db, dvy_db)

`_patch_direction_vec_2d` plus the constant partials `∂v/∂b`. Used by
the analytic-Jacobian path in `make_geometry` for `Mesh{2}` curvilinear
patches.
"""
@inline function _patch_direction_vec_2d_and_derivs(dir::Integer, b::T) where {T<:Real}
    z = zero(T); o = one(T)
    if     dir == 1;  return ( o,  b,    z,  o)    # +x:  v = ( 1,  b), dv/db = (0,  1)
    elseif dir == 2;  return (-o, -b,    z, -o)    # −x:  v = (-1, -b), dv/db = (0, -1)
    elseif dir == 3;  return (-b,  o,   -o,  z)    # +y:  v = (-b,  1), dv/db = (-1, 0)
    else              return ( b, -o,    o,  z)    # −y:  v = ( b, -1), dv/db = (1,  0)
    end
end

"""
    _patch_direction_vec(dir::Integer, b, c) → (vx, vy, vz)

Direction-dependent unit-ish vector for 3D inflation / shell patches.

    +x:  v = ( 1,  b,  c)            -x:  v = (-1,  c,  b)
    +y:  v = ( c,  1,  b)            -y:  v = ( b, -1,  c)
    +z:  v = ( b,  c,  1)            -z:  v = ( c,  b, -1)
"""
@inline function _patch_direction_vec(dir::Integer, b::T, c::T) where {T<:Real}
    o = one(T)
    if     dir == 1;  return ( o,  b,  c)
    elseif dir == 2;  return (-o,  c,  b)
    elseif dir == 3;  return ( c,  o,  b)
    elseif dir == 4;  return ( b, -o,  c)
    elseif dir == 5;  return ( b,  c,  o)
    else              return ( c,  b, -o)
    end
end

"""
    _patch_direction_vec_and_derivs(dir::Integer, b, c)
        → (vx, vy, vz, dvx_db, dvy_db, dvz_db, dvx_dc, dvy_dc, dvz_dc)

`_patch_direction_vec` plus the constant partials `∂v/∂b`, `∂v/∂c`.
Used by the analytic-Jacobian path in `make_geometry` for `Mesh{3}`
curvilinear patches.
"""
@inline function _patch_direction_vec_and_derivs(dir::Integer, b::T, c::T) where {T<:Real}
    z = zero(T); o = one(T)
    if     dir == 1;  return ( o,  b,  c,    z,  o,  z,    z,  z,  o)    # +x
    elseif dir == 2;  return (-o,  c,  b,    z,  z,  o,    z,  o,  z)    # −x
    elseif dir == 3;  return ( c,  o,  b,    z,  z,  o,    o,  z,  z)    # +y
    elseif dir == 4;  return ( b, -o,  c,    o,  z,  z,    z,  z,  o)    # −y
    elseif dir == 5;  return ( b,  c,  o,    o,  z,  z,    z,  o,  z)    # +z
    else              return ( c,  b, -o,    z,  o,  z,    o,  z,  z)    # −z
    end
end

# ----- Per-patch vertex parametric map -------------------------------

"""
    _patch_vertex_position(pd::PatchDesc{D, T}, idx::NTuple{D, Integer})
        → NTuple{D, T}

Family-dispatched coordinate map: given a `PatchDesc` and the 0-indexed
integer vertex coordinates `(idx[1], …, idx[D])` of the patch's
structured grid (`idx[d] ∈ 0..dims(pd)[d]`), return the physical
Cartesian position.

Dispatches on `pd.kind`:

* `Cubic` — affine `[x_lo, x_hi]^D` (any D).
* `Wedge` — radial linear-r wedge, `r(a) = R1·(R2/R1)^a` (D = 2, 3).
* `Inflation` — radial bridge to inner sphere, `f(a) = (1−a)·L + a·R1/Q` (D = 2, 3).
* `Shell` — spherical/annular shell, `f(a) = ((1−a)·R1 + a·R2)/Q` (D = 2, 3).
"""
function _patch_vertex_position(pd::PatchDesc{D, T},
                                  idx::NTuple{D, <:Integer}) where {D, T}
    k = pd.kind
    if k === Cubic
        return _vert_cubic(pd.cubic, idx)
    elseif k === Wedge
        return _vert_wedge(pd.wedge, idx)
    elseif k === Inflation
        return _vert_inflation(pd.inflation, idx)
    else  # Shell
        return _vert_shell(pd.shell, idx)
    end
end

# Cubic — works for any D.
@inline function _vert_cubic(c::PatchCubic{D, T},
                              idx::NTuple{D, <:Integer}) where {D, T}
    ntuple(Val(D)) do d
        ξ = T(idx[d]) / T(c.dims[d])
        return c.x_lo[d] + (c.x_hi[d] - c.x_lo[d]) * ξ
    end
end

# Wedge — D = 2 and 3.
@inline function _vert_wedge(w::PatchWedge{2, T},
                              idx::NTuple{2, <:Integer}) where {T}
    a = w.a_lo + (w.a_hi - w.a_lo) * (T(idx[1]) / T(w.dims[1]))
    b = w.b_lo + (w.b_hi - w.b_lo) * (T(idx[2]) / T(w.dims[2]))
    r = w.R1 * (w.R2 / w.R1)^a
    dir = w.dir
    if     dir == Int8(1);  return ( r,  b * r)
    elseif dir == Int8(2);  return (-r,  b * r)
    elseif dir == Int8(3);  return (b * r,  r)
    else                    return (b * r, -r)
    end
end

@inline function _vert_wedge(w::PatchWedge{3, T},
                              idx::NTuple{3, <:Integer}) where {T}
    a = w.a_lo + (w.a_hi - w.a_lo) * (T(idx[1]) / T(w.dims[1]))
    b = w.b_lo + (w.b_hi - w.b_lo) * (T(idx[2]) / T(w.dims[2]))
    c = w.c_lo + (w.c_hi - w.c_lo) * (T(idx[3]) / T(w.dims[3]))
    r = w.R1 * (w.R2 / w.R1)^a
    dir = w.dir
    if     dir == Int8(1);  return ( r,    b * r, c * r)
    elseif dir == Int8(2);  return (-r,    b * r, c * r)
    elseif dir == Int8(3);  return (b * r,  r,    c * r)
    elseif dir == Int8(4);  return (b * r, -r,    c * r)
    elseif dir == Int8(5);  return (b * r, c * r,  r)
    else                    return (b * r, c * r, -r)
    end
end

# Inflation — D = 2 and 3.
@inline function _vert_inflation(i::PatchInflation{2, T},
                                   idx::NTuple{2, <:Integer}) where {T}
    a = i.a_lo + (i.a_hi - i.a_lo) * (T(idx[1]) / T(i.dims[1]))
    b = i.b_lo + (i.b_hi - i.b_lo) * (T(idx[2]) / T(i.dims[2]))
    Q = sqrt(one(T) + b * b)
    vx, vy = _patch_direction_vec_2d(i.dir, b)
    f = (one(T) - a) * i.L + a * i.R1 / Q
    return (f * vx, f * vy)
end

@inline function _vert_inflation(i::PatchInflation{3, T},
                                   idx::NTuple{3, <:Integer}) where {T}
    a = i.a_lo + (i.a_hi - i.a_lo) * (T(idx[1]) / T(i.dims[1]))
    b = i.b_lo + (i.b_hi - i.b_lo) * (T(idx[2]) / T(i.dims[2]))
    c = i.c_lo + (i.c_hi - i.c_lo) * (T(idx[3]) / T(i.dims[3]))
    Q = sqrt(one(T) + b * b + c * c)
    vx, vy, vz = _patch_direction_vec(i.dir, b, c)
    f = (one(T) - a) * i.L + a * i.R1 / Q
    return (f * vx, f * vy, f * vz)
end

# Shell — D = 2 and 3.
@inline function _vert_shell(s::PatchShell{2, T},
                               idx::NTuple{2, <:Integer}) where {T}
    a = s.a_lo + (s.a_hi - s.a_lo) * (T(idx[1]) / T(s.dims[1]))
    b = s.b_lo + (s.b_hi - s.b_lo) * (T(idx[2]) / T(s.dims[2]))
    Q = sqrt(one(T) + b * b)
    vx, vy = _patch_direction_vec_2d(s.dir, b)
    r = (one(T) - a) * s.R1 + a * s.R2
    f = r / Q
    return (f * vx, f * vy)
end

@inline function _vert_shell(s::PatchShell{3, T},
                               idx::NTuple{3, <:Integer}) where {T}
    a = s.a_lo + (s.a_hi - s.a_lo) * (T(idx[1]) / T(s.dims[1]))
    b = s.b_lo + (s.b_hi - s.b_lo) * (T(idx[2]) / T(s.dims[2]))
    c = s.c_lo + (s.c_hi - s.c_lo) * (T(idx[3]) / T(s.dims[3]))
    Q = sqrt(one(T) + b * b + c * c)
    vx, vy, vz = _patch_direction_vec(s.dir, b, c)
    r = (one(T) - a) * s.R1 + a * s.R2
    f = r / Q
    return (f * vx, f * vy, f * vz)
end

# ----- Internal skeleton types ---------------------------------------

# `FaceLink` describes one entry in a skeleton's `(2D, npatches)`
# face-connectivity table. Two flavours selected by `kind`: `InteriorLink`
# carries `(neigh_patch, neigh_face, orientation)`; `BoundaryLink`
# carries only `boundary_tag ∈ 1..127`.
@enum FaceLinkKind::Int8 begin
    InteriorLink = 1
    BoundaryLink = 2
end

struct FaceLink
    kind         :: FaceLinkKind
    neigh_patch  :: Int
    neigh_face   :: Int
    orientation  :: Int8
    boundary_tag :: Int8
end

interior_link(np::Integer, nf::Integer, o::Integer) =
    FaceLink(InteriorLink, Int(np), Int(nf), Int8(o), Int8(0))
boundary_link(tag::Integer) =
    FaceLink(BoundaryLink, 0, 0, Int8(0), Int8(tag))

"""
    SkeletonMesh{D, T}

Internal build-time skeleton: a list of patches plus their inter-patch
face connectivity. Pass to `_skeleton_to_mesh(skel)` to instantiate the
full `Mesh{D, T}`.
"""
struct SkeletonMesh{D, T}
    patches :: Vector{PatchDesc{D, T}}
    faces   :: Matrix{FaceLink}    # shape (2D, npatches)
end

# ----- Face / orientation helpers ------------------------------------

# Fixed axis for face f ∈ 1..2D — the axis along which f's normal
# points. f = 1, 2 → axis 1; f = 3, 4 → axis 2; f = 5, 6 → axis 3.
@inline _fixed_axis(f::Integer) = ((f - 1) ÷ 2) + 1

# Whether face f is the "low" end of its axis (odd) or the "high" end (even).
@inline _is_low_face(f::Integer) = isodd(f)

# Tangent-axis dimension counts for face f given a patch's dims tuple.
# Returns NTuple{D-1, Int}: dims with the fixed axis removed, preserving
# order.
@inline function _face_tangent_dims(f::Integer, d::NTuple{D, Int}) where {D}
    fa = _fixed_axis(f)
    ntuple(t -> d[t < fa ? t : t + 1], Val(D - 1))
end

# Map face f + 0-indexed tangent coords → full 0-indexed patch-vertex
# index NTuple{D, Int}.
@inline function _face_vert_to_idx(f::Integer,
                                     tangent::NTuple{Dm1, <:Integer},
                                     d::NTuple{D, Int}) where {D, Dm1}
    fa = _fixed_axis(f)
    fixed_val = _is_low_face(f) ? 0 : d[fa]
    ntuple(Val(D)) do ax
        if ax == fa
            fixed_val
        elseif ax < fa
            Int(tangent[ax])
        else
            Int(tangent[ax - 1])
        end
    end
end

# Same, for 1-indexed face cells.
@inline function _face_cell_to_idx(f::Integer,
                                     tangent::NTuple{Dm1, <:Integer},
                                     d::NTuple{D, Int}) where {D, Dm1}
    fa = _fixed_axis(f)
    fixed_val = _is_low_face(f) ? 1 : d[fa]
    ntuple(Val(D)) do ax
        if ax == fa
            fixed_val
        elseif ax < fa
            Int(tangent[ax])
        else
            Int(tangent[ax - 1])
        end
    end
end

# Inverse: 1-indexed element coords → tangent-axis coords on face f.
@inline function _face_cell_to_tangent(f::Integer,
                                         idx::NTuple{D, <:Integer}) where {D}
    fa = _fixed_axis(f)
    ntuple(t -> Int(idx[t < fa ? t : t + 1]), Val(D - 1))
end

# Opposite face along the same axis: 1↔2, 3↔4, 5↔6. As an Int8 tuple
# of length 2D — looked up by face index `f`.
@inline _opposite_face(::Val{D}) where {D} =
    ntuple(f -> Int8(isodd(f) ? f + 1 : f - 1), Val(2 * D))

"""
    OrientationGroup{D}

Type-only marker for the face-orientation group of `Mesh{D}` meshes:

* `OrientationGroup{2}` — D₁ (dihedral group of order 2). Two
  elements `o ∈ {0, 1}` (identity, reversal). Acts on a 1-D face
  segment via [`_neigh_p_vertex`](@ref) / [`_neigh_p_cell`](@ref).
* `OrientationGroup{3}` — D₄ (dihedral group of order 8). Eight
  elements `o ∈ 0..7`. Acts on a 2-D face quad via
  [`_neigh_pq_vertex`](@ref) / [`_neigh_pq_cell`](@ref).

Group elements are passed as `Int8` integers throughout the codebase;
this type is purely a documentation / dispatch anchor. The D-generic
dispatchers `_neigh_tangent_vertex` / `_neigh_tangent_cell` route
through the appropriate single-tangent / two-tangent helper based on
the tangent tuple's arity (which is `D − 1`).
"""
struct OrientationGroup{D} end

"""
    n_orientations(::OrientationGroup{D}) → Int

Cardinality of the orientation group: 2 for D = 2 (D₁), 8 for D = 3 (D₄).
"""
@inline n_orientations(::OrientationGroup{2}) = 2
@inline n_orientations(::OrientationGroup{3}) = 8

# Orientation transform on tangent-axis coords. Dispatches on the
# tangent dimensionality D - 1:
#   D = 2: 1 tangent → D₁ group (o ∈ 0..1).
#   D = 3: 2 tangents → D₄ group (o ∈ 0..7).

"""
    _neigh_p_vertex(o, p, Mt) → p′

D₁ orientation transform on a single 0-indexed vertex coordinate
`p ∈ 0..Mt`, used for `Mesh{2}` face skeletons. `o = 0` is the
identity (`p′ = p`); `o = 1` is the reversal (`p′ = Mt − p`).

The cell-indexed analog is [`_neigh_p_cell`](@ref) (range `1..Mt`);
the runtime kernel variant operating on per-element 1-indexed face
nodes is `HexMeshes._neigh_p`.
"""
@inline _neigh_p_vertex(o::Integer, p::Integer, Mt::Integer) =
    o == 0 ? p : (Mt - p)

"""
    _neigh_p_cell(o, b, Mt) → b′

D₁ orientation transform on a single 1-indexed cell coordinate
`b ∈ 1..Mt`. The vertex-indexed analog is [`_neigh_p_vertex`](@ref).
"""
@inline _neigh_p_cell(o::Integer, b::Integer, Mt::Integer) =
    o == 0 ? b : (Mt + 1 - b)

"""
    _neigh_pq_vertex(o, p, q, Mt1, Mt2) → (p′, q′)

D₄ orientation transform on a pair of 0-indexed vertex coordinates
`(p, q) ∈ 0..Mt1 × 0..Mt2`, used for `Mesh{3}` face skeletons. The
eight elements `o ∈ 0..7` enumerate the four rotations and four
reflections of the square. `o = 0` is the identity; `o = 4..7` are
the reflections.

The cell-indexed analog is [`_neigh_pq_cell`](@ref); the runtime
kernel variant operating on per-element 1-indexed face nodes is
`HexMeshes._neigh_pq`.
"""
@inline function _neigh_pq_vertex(o::Integer, p::Integer, q::Integer,
                                    Mt1::Integer, Mt2::Integer)
    if     o == 0;  return (p,        q       )
    elseif o == 1;  return (q,        Mt1 - p )
    elseif o == 2;  return (Mt1 - p,  Mt2 - q )
    elseif o == 3;  return (Mt2 - q,  p       )
    elseif o == 4;  return (Mt1 - p,  q       )
    elseif o == 5;  return (q,        p       )
    elseif o == 6;  return (p,        Mt2 - q )
    else            return (Mt2 - q,  Mt1 - p )
    end
end

"""
    _neigh_pq_cell(o, b, c, Mt1, Mt2) → (b′, c′)

D₄ orientation transform on a pair of 1-indexed cell coordinates
`(b, c) ∈ 1..Mt1 × 1..Mt2`. The vertex-indexed analog is
[`_neigh_pq_vertex`](@ref).
"""
@inline function _neigh_pq_cell(o::Integer, b::Integer, c::Integer,
                                  Mt1::Integer, Mt2::Integer)
    if     o == 0;  return (b,             c            )
    elseif o == 1;  return (c,             Mt1 + 1 - b  )
    elseif o == 2;  return (Mt1 + 1 - b,   Mt2 + 1 - c  )
    elseif o == 3;  return (Mt2 + 1 - c,   b            )
    elseif o == 4;  return (Mt1 + 1 - b,   c            )
    elseif o == 5;  return (c,             b            )
    elseif o == 6;  return (b,             Mt2 + 1 - c  )
    else            return (Mt2 + 1 - c,   Mt1 + 1 - b  )
    end
end

# D-generic dispatchers
@inline _neigh_tangent_vertex(o, tangent::NTuple{1}, Mt::NTuple{1}) =
    (_neigh_p_vertex(o, tangent[1], Mt[1]),)
@inline _neigh_tangent_vertex(o, tangent::NTuple{2}, Mt::NTuple{2}) =
    _neigh_pq_vertex(o, tangent[1], tangent[2], Mt[1], Mt[2])

@inline _neigh_tangent_cell(o, tangent::NTuple{1}, Mt::NTuple{1}) =
    (_neigh_p_cell(o, tangent[1], Mt[1]),)
@inline _neigh_tangent_cell(o, tangent::NTuple{2}, Mt::NTuple{2}) =
    _neigh_pq_cell(o, tangent[1], tangent[2], Mt[1], Mt[2])

# Lexicographic encoding of (D-tuple, dims) → linear index.
# 0-indexed → 1-indexed: vid = v + 1; 1-indexed cells → e = e_lex + 1.

# Pre-dedup vertex id for vertex `idx` in patch `p`, given the patches'
# vertex-offset table.
@inline function _vid(p::Int, idx::NTuple{D, <:Integer}, d::NTuple{D, Int},
                       vert_offs::Vector{Int}) where {D}
    # Lex: idx[1] + (d[1]+1) * (idx[2] + (d[2]+1) * (idx[3] + ...))
    v = Int(idx[D])
    for ax in (D - 1):-1:1
        v = Int(idx[ax]) + (d[ax] + 1) * v
    end
    return vert_offs[p] + 1 + v
end

# Pre-dedup element id for 1-indexed cell `idx` in patch `p`.
@inline function _eid(p::Int, idx::NTuple{D, <:Integer}, d::NTuple{D, Int},
                       elem_offs::Vector{Int}) where {D}
    e = Int(idx[D]) - 1
    for ax in (D - 1):-1:1
        e = (Int(idx[ax]) - 1) + d[ax] * e
    end
    return elem_offs[p] + 1 + e
end

# Iterator over all 0-indexed vertex tuples for a patch with dims `d`.
@inline _vertex_iter(d::NTuple{D, Int}) where {D} =
    Iterators.product(ntuple(ax -> 0:d[ax], Val(D))...)

# Iterator over all 1-indexed cell tuples for a patch with dims `d`.
@inline _cell_iter(d::NTuple{D, Int}) where {D} =
    Iterators.product(ntuple(ax -> 1:d[ax], Val(D))...)

# Iterator over the 2^D corner offsets of a single cell (each entry in
# {0, 1} for each axis, in Gmsh-canonical tensor-product ordering).
# We just generate from a 0/1 tuple and convert to the Gmsh order at
# the call site.
const _GMSH_CORNERS_1D = ((0,),               (1,))
const _GMSH_CORNERS_2D = ((0, 0), (1, 0), (1, 1), (0, 1))
const _GMSH_CORNERS_3D = ((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
                          (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1))
@inline _gmsh_corners(::Val{1}) = _GMSH_CORNERS_1D
@inline _gmsh_corners(::Val{2}) = _GMSH_CORNERS_2D
@inline _gmsh_corners(::Val{3}) = _GMSH_CORNERS_3D

# ----- Main builder --------------------------------------------------

"""
    _skeleton_to_mesh(skel::SkeletonMesh{D, T}) → Mesh{D, T}

Instantiate the full element-level `Mesh{D, T}` from a skeleton. `D`
must be `2` or `3` (the 1D path is a direct builder in `builders_1d.jl`).

1. Per-patch pre-dedup vertex ids `(p, idx...)`.
2. Union-find over face-shared ids using the skeleton's interior
   `FaceLink`s and the integer D₁ / D₄ orientation transform.
3. Dense canonical vertex ids `1..Nv`.
4. Coordinates from `_patch_vertex_position` at one representative per
   canonical id.
5. Per-element `vertex_idx`, `neighbour`, `neighbour_face`,
   `orientation`, `bdry`, `patch_id`, `patch_idx` tables.
6. Returns a `Mesh{D, T}` with all four patch fields populated.

No floating-point comparison is load-bearing.
"""
function _skeleton_to_mesh(skel::SkeletonMesh{D, T}) where {D, T}
    n_patches = length(skel.patches)
    @assert size(skel.faces) == (2 * D, n_patches)
    @assert D == 2 || D == 3

    # --- Per-patch vertex / element offsets --------------------------
    vert_offs = Vector{Int}(undef, n_patches + 1)
    elem_offs = Vector{Int}(undef, n_patches + 1)
    vert_offs[1] = 0
    elem_offs[1] = 0
    for p in 1:n_patches
        d = dims(skel.patches[p])
        vert_offs[p + 1] = vert_offs[p] + prod(d .+ 1)
        elem_offs[p + 1] = elem_offs[p] + prod(d)
    end
    Nv_pre = vert_offs[end]
    Ne     = elem_offs[end]

    # --- Union-find (path-compression + rank) ------------------------
    parent = collect(1:Nv_pre)
    rank   = zeros(Int, Nv_pre)
    function uf_find(x)
        while parent[x] != x
            parent[x] = parent[parent[x]]
            x = parent[x]
        end
        return x
    end
    function uf_union!(x, y)
        rx, ry = uf_find(x), uf_find(y)
        rx == ry && return
        if     rank[rx] < rank[ry]; parent[rx] = ry
        elseif rank[rx] > rank[ry]; parent[ry] = rx
        else;                        parent[ry] = rx; rank[rx] += 1
        end
        return
    end

    # --- Walk every interior face link once. The `(p, f) < (p2, f2)`
    # guard processes each pair exactly once even when the skeleton
    # lists both halves (which it normally does, for symmetry).
    nfaces = 2 * D
    for p in 1:n_patches, f in 1:nfaces
        link = skel.faces[f, p]
        link.kind === InteriorLink || continue
        p2 = link.neigh_patch
        f2 = link.neigh_face
        ((p, f) < (p2, f2)) || continue
        o   = Int(link.orientation)
        d   = dims(skel.patches[p])
        d2  = dims(skel.patches[p2])
        Mt  = _face_tangent_dims(f, d)     # NTuple{D-1, Int}
        for tangent in Iterators.product(ntuple(t -> 0:Mt[t], Val(D - 1))...)
            idx1 = _face_vert_to_idx(f, tangent, d)
            tangent2 = _neigh_tangent_vertex(o, tangent, Mt)
            idx2 = _face_vert_to_idx(f2, tangent2, d2)
            uf_union!(_vid(p, idx1, d, vert_offs),
                      _vid(p2, idx2, d2, vert_offs))
        end
    end

    # --- Canonical dense ids -----------------------------------------
    Nv = 0
    canon = zeros(Int, Nv_pre)
    for x in 1:Nv_pre
        if uf_find(x) == x
            Nv += 1
            canon[x] = Nv
        end
    end
    final = Vector{Int}(undef, Nv_pre)
    for x in 1:Nv_pre
        final[x] = canon[uf_find(x)]
    end

    # --- Coordinates (one evaluation per canonical id) ---------------
    vertex_coords = Matrix{T}(undef, D, Nv)
    written = falses(Nv)
    for p in 1:n_patches
        pd = skel.patches[p]
        d  = dims(pd)
        for vidx in _vertex_iter(d)
            id = final[_vid(p, vidx, d, vert_offs)]
            written[id] && continue
            x = _patch_vertex_position(pd, vidx)
            for ax in 1:D
                vertex_coords[ax, id] = x[ax]
            end
            written[id] = true
        end
    end
    @assert all(written)

    # --- Per-element tables ------------------------------------------
    ncorners      = 2^D
    vertex_idx    = Matrix{Int}(undef, ncorners, Ne)
    neighbour     = zeros(Int32, nfaces, Ne)
    neighbour_face = zeros(Int8, nfaces, Ne)
    orientation   = zeros(Int8, nfaces, Ne)
    bdry          = zeros(Int8, nfaces, Ne)

    patch_id  = Vector{Int32}(undef, Ne)
    patch_idx = Matrix{Int32}(undef, D, Ne)

    OPP = _opposite_face(Val(D))
    corners = _gmsh_corners(Val(D))

    for p in 1:n_patches
        pd = skel.patches[p]
        d  = dims(pd)
        for cidx in _cell_iter(d)
            # cidx is the 1-indexed (a, b[, c]) of the cell in the patch.
            e = _eid(p, cidx, d, elem_offs)

            # Per-element patch metadata.
            patch_id[e] = Int32(p)
            for ax in 1:D
                patch_idx[ax, e] = Int32(cidx[ax])
            end

            # Gmsh-canonical corner vertex ids.
            for (k, off) in enumerate(corners)
                vidx_corner = ntuple(ax -> Int(cidx[ax]) - 1 + Int(off[ax]), Val(D))
                vertex_idx[k, e] = final[_vid(p, vidx_corner, d, vert_offs)]
            end

            # Per-face neighbour / orientation / bdry.
            for f in 1:nfaces
                fa  = _fixed_axis(f)
                low = _is_low_face(f)
                at_patch_boundary = low ? (cidx[fa] == 1) : (cidx[fa] == d[fa])

                if !at_patch_boundary
                    # Within-patch sibling element.
                    nbr_cidx = ntuple(ax -> ax == fa ?
                                            (low ? cidx[ax] - 1 : cidx[ax] + 1) :
                                            cidx[ax], Val(D))
                    neighbour[f, e]      = _eid(p, nbr_cidx, d, elem_offs)
                    neighbour_face[f, e] = OPP[f]
                    orientation[f, e]    = Int8(0)
                else
                    link = skel.faces[f, p]
                    if link.kind === BoundaryLink
                        bdry[f, e] = link.boundary_tag
                    else
                        p2 = link.neigh_patch
                        f2 = link.neigh_face
                        o  = Int(link.orientation)
                        d2 = dims(skel.patches[p2])
                        Mt = _face_tangent_dims(f, d)
                        tangent_self = _face_cell_to_tangent(f, cidx)
                        tangent_nbr  = _neigh_tangent_cell(o, tangent_self, Mt)
                        cidx2 = _face_cell_to_idx(f2, tangent_nbr, d2)
                        neighbour[f, e]      = _eid(p2, cidx2, d2, elem_offs)
                        neighbour_face[f, e] = Int8(f2)
                        orientation[f, e]    = Int8(o)
                    end
                end
            end
        end
    end

    return Mesh{D, T}(Ne, neighbour, neighbour_face, orientation, bdry,
                     vertex_coords, vertex_idx;
                     patch_id              = patch_id,
                     patch_idx             = patch_idx,
                     patch_desc            = copy(skel.patches),
                     patch_element_offset  = elem_offs)
end
