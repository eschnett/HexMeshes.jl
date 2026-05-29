# ----------------------------------------------------------------------
# Skeleton-based 2D mesh construction.
#
# Two-stage build that mirrors the 3D `skeleton.jl` machinery, with the
# obvious dimensional reductions:
#
#   * Each patch has two local axes `(a, b)` (was three for 3D).
#   * Each patch has 4 faces (was 6), each a 1-D edge (was 2-D quad).
#   * The orientation group on a face is D₁ — identity or reversal —
#     instead of D₄ (8 elements).
#   * Pre-dedup vertices are integer triples `(p, i, j)` indexed over
#     `1..n_patches × 0..Ma × 0..Mb`.
#
# Like the 3D version, this is integer-only union-find dedup — no
# floating-point comparison is ever load-bearing.

"""
    PatchSpec2D{T}

Combinatorial + parametric description of one patch in a 2D multi-block
quad mesh.

# Fields

* `Ma, Mb :: Int` — element counts along the patch's two local axes.
* `family :: PatchKind2D` — selects the parametric vertex map at
  coordinate-assignment time. One of the `PatchKind2D` enum members:
  `Cubical_2D`, `InflationPos/NegX/Y_2D`, `ShellPos/NegX/Y_2D`, or
  `WedgePos/NegX/Y_2D`. The `family` field doubles as the
  `PatchInfo2D.kind` tag stored per element by
  `make_inflated_square_mesh` (only inflation, shell, and cubical
  values appear there in practice).
* `a_lo, a_hi, b_lo, b_hi :: T` — affine ranges in the patch's local
  `(a, b)` parameter space. The reference square `[0, 1]²` is mapped
  to these before the family-specific transform.
* `L, R1, R2 :: T` — analytic constants used by curvilinear families;
  zero for `Cubical_2D`.
"""
struct PatchSpec2D{T}
    Ma     :: Int
    Mb     :: Int
    family :: PatchKind2D
    a_lo   :: T
    a_hi   :: T
    b_lo   :: T
    b_hi   :: T
    L      :: T
    R1     :: T
    R2     :: T
end

"""
    FaceLink2D

One entry in a `SkeletonMesh2D`'s 4×n_patches face-connectivity table.
Two flavours, selected by `kind`:

* `kind = :interior` — face is shared with another patch face. Carries
  `(neigh_patch, neigh_face, orientation ∈ 0..1)`.
* `kind = :boundary` — face is on the domain boundary. Carries only
  `boundary_tag ∈ 1..127`.

Use `interior_link_2d(np, nf, o)` / `boundary_link_2d(tag)` to construct.
"""
struct FaceLink2D
    kind         :: Symbol
    neigh_patch  :: Int
    neigh_face   :: Int
    orientation  :: Int8
    boundary_tag :: Int8
end

interior_link_2d(np::Integer, nf::Integer, o::Integer) =
    FaceLink2D(:interior, Int(np), Int(nf), Int8(o), Int8(0))
boundary_link_2d(tag::Integer) =
    FaceLink2D(:boundary, 0, 0, Int8(0), Int8(tag))

"""
    SkeletonMesh2D{T}

Patch list + 4×n_patches face-link table. `_skeleton_to_mesh_2d(skel)`
instantiates the full `QuadMesh{T}` from this skeleton.
"""
struct SkeletonMesh2D{T}
    patches :: Vector{PatchSpec2D{T}}
    faces   :: Matrix{FaceLink2D}
end

# ----- Per-face index helpers (2D skeleton scope) -------------------

# Element count along the tangent axis of face `f ∈ 1..4` in patch `ps`.
@inline function _face_tangent_count_2d(f::Integer, ps::PatchSpec2D)
    return (f == 1 || f == 2) ? ps.Mb : ps.Ma
end

# Face-local 0-based vertex `pp` → patch vertex `(i, j)`.
@inline function _face_vert_to_ij_2d(f::Integer, pp::Integer,
                                      ps::PatchSpec2D)
    if f == 1
        return (0,     pp)
    elseif f == 2
        return (ps.Ma, pp)
    elseif f == 3
        return (pp, 0)
    else                # f == 4
        return (pp, ps.Mb)
    end
end

# Element `(a, b)` projected onto face `f` → face-cell `p_cell` (1-indexed).
@inline function _face_cell_to_p_2d(f::Integer, a::Integer, b::Integer)
    return (f == 1 || f == 2) ? b : a
end

# Face-cell `p_cell` on face `f` → element `(a, b)` (1-indexed).
@inline function _face_cell_to_ab_2d(f::Integer, p_cell::Integer,
                                       ps::PatchSpec2D)
    if f == 1
        return (1,      p_cell)
    elseif f == 2
        return (ps.Ma,  p_cell)
    elseif f == 3
        return (p_cell, 1)
    else
        return (p_cell, ps.Mb)
    end
end

# D₁ transform on 0-indexed face vertex `p ∈ 0..Mt`.
@inline function _neigh_p_vertex_2d(o::Integer, p::Integer, Mt::Integer)
    return o == 0 ? p : (Mt - p)
end

# Same D₁ transform on 1-indexed face cells `b ∈ 1..Mt`, used to identify
# the neighbour element across a cross-patch face link.
@inline function _neigh_p_cell_2d(o::Integer, b::Integer, Mt::Integer)
    return o == 0 ? b : (Mt + 1 - b)
end

"""
    _patch_vertex_position_2d(ps::PatchSpec2D{T}, i, j) → (x, y)

Family-dispatched coordinate map: given integer vertex `(i, j) ∈
0..Ma × 0..Mb` of patch `ps`, return the physical `(x, y)`. Each family
interprets `(a_lo, a_hi, b_lo, b_hi)` and `(L, R1, R2)` according to
its own parametrisation.
"""
function _patch_vertex_position_2d(ps::PatchSpec2D{T}, i::Integer,
                                    j::Integer) where {T}
    fam = ps.family
    if is_cubical(fam)
        ξ = T(i) / T(ps.Ma)
        η = T(j) / T(ps.Mb)
        return (ps.a_lo + (ps.a_hi - ps.a_lo) * ξ,
                ps.b_lo + (ps.b_hi - ps.b_lo) * η)
    elseif is_wedge(fam)
        # Radial wedge (cubed-square outer patch). Geometric radial
        # spacing `r(a) = R1·(R2/R1)^a` so each cell stays roughly
        # square; angular axis `b ∈ [-1, 1]`.
        a = ps.a_lo + (ps.a_hi - ps.a_lo) * (T(i) / T(ps.Ma))
        b = ps.b_lo + (ps.b_hi - ps.b_lo) * (T(j) / T(ps.Mb))
        r = ps.R1 * (ps.R2 / ps.R1)^a
        dir = direction_of(fam)
        if     dir == Int8(1);  return ( r,  b*r)
        elseif dir == Int8(2);  return (-r,  b*r)
        elseif dir == Int8(3);  return (b*r,  r)
        else                    return (b*r, -r)
        end
    else
        # Inflation / shell families (9-patch inflated square).
        a = ps.a_lo + (ps.a_hi - ps.a_lo) * (T(i) / T(ps.Ma))
        b = ps.b_lo + (ps.b_hi - ps.b_lo) * (T(j) / T(ps.Mb))
        Q = sqrt(one(T) + b * b)
        dir = direction_of(fam)
        vx, vy = _patch_direction_vec_2d(dir, b)
        if is_shell(fam)
            r = (one(T) - a) * ps.R1 + a * ps.R2
            f = r / Q
        else               # inflation
            f = (one(T) - a) * ps.L + a * ps.R1 / Q
        end
        return (f * vx, f * vy)
    end
end

# Direction-dependent unit-ish vector `v(b)` for the 2D inflation /
# shell families. The b-flip pattern for `−x, +y` is the 2D analog of
# the 3D `_patch_direction_vec` axis-swap: it picks the tangent-axis
# orientation per direction so the local `(a, b)` frame is right-handed
# (`det J > 0`) in physical space everywhere. Divided by `Q = √(1 + b²)`
# this gives the unit radial direction from origin out to the patch face.
#
#   +x:  v = ( 1,  b)              −x:  v = (-1, -b)
#   +y:  v = (-b,  1)              −y:  v = ( b, -1)
@inline function _patch_direction_vec_2d(dir::Integer, b::T) where {T<:Real}
    o = one(T)
    if dir == 1                     # +x
        return ( o,  b)
    elseif dir == 2                 # −x
        return (-o, -b)
    elseif dir == 3                 # +y
        return (-b,  o)
    else                            # −y
        return ( b, -o)
    end
end

# Same as `_patch_direction_vec_2d`, plus the constant partial `dv/db`
# (a 2-tuple; entries are `0` or `±1`). Used by the analytic-Jacobian
# path in `make_geometry(::InflatedSquareMesh, …)`.
@inline function _patch_direction_vec_2d_and_derivs(dir::Integer, b::T) where {T<:Real}
    z = zero(T); o = one(T)
    if dir == 1                     # +x: v = (1, b),   dv/db = (0, 1)
        return (o, b,    z, o)
    elseif dir == 2                 # −x: v = (-1, -b), dv/db = (0, -1)
        return (-o, -b,  z, -o)
    elseif dir == 3                 # +y: v = (-b, 1),  dv/db = (-1, 0)
        return (-b, o,   -o, z)
    else                            # −y: v = (b, -1),  dv/db = (1, 0)
        return (b, -o,   o, z)
    end
end

"""
    _skeleton_to_mesh_2d(skel::SkeletonMesh2D{T}) → QuadMesh{T}

Instantiate the full element-level `QuadMesh{T}` from a 2D skeleton:

1. Per-patch pre-dedup vertex ids `(p, i, j)`.
2. Union-find over face-shared ids using the skeleton's interior
   `FaceLink2D`s and the integer D₁ orientation transform.
3. Dense canonical vertex ids `1..Nv`.
4. Coordinates from `_patch_vertex_position_2d` at one representative
   per canonical id.
5. Per-element `vertex_idx`, `neighbour`, `neighbour_face`,
   `orientation`, `bdry` tables.

No floating-point comparison is ever load-bearing.
"""
function _skeleton_to_mesh_2d(skel::SkeletonMesh2D{T}) where {T}
    n_patches = length(skel.patches)
    @assert size(skel.faces) == (4, n_patches)

    # Pre-dedup vertex / element offsets per patch.
    vert_offs = Vector{Int}(undef, n_patches + 1)
    elem_offs = Vector{Int}(undef, n_patches + 1)
    vert_offs[1] = 0
    elem_offs[1] = 0
    for p in 1:n_patches
        ps = skel.patches[p]
        vert_offs[p+1] = vert_offs[p] + (ps.Ma + 1) * (ps.Mb + 1)
        elem_offs[p+1] = elem_offs[p] + ps.Ma * ps.Mb
    end
    Nv_pre = vert_offs[end]
    Ne     = elem_offs[end]

    @inline function vid(p, i, j)
        ps = skel.patches[p]
        return vert_offs[p] + 1 + i + (ps.Ma + 1) * j
    end
    @inline function eid(p, a, b)
        ps = skel.patches[p]
        return elem_offs[p] + a + ps.Ma * (b - 1)
    end

    # --- Union-find (path-compression + rank) -------------------------
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
        else;                        parent[ry] = rx;  rank[rx] += 1
        end
        return
    end

    # Walk every interior face link once. The `(p, f) < (p2, f2)` guard
    # processes each pair exactly once.
    for p in 1:n_patches, f in 1:4
        link = skel.faces[f, p]
        link.kind === :interior || continue
        p2 = link.neigh_patch
        f2 = link.neigh_face
        ((p, f) < (p2, f2)) || continue
        o   = Int(link.orientation)
        ps  = skel.patches[p]
        ps2 = skel.patches[p2]
        Mt = _face_tangent_count_2d(f, ps)
        for pp in 0:Mt
            i1, j1 = _face_vert_to_ij_2d(f, pp, ps)
            pp2 = _neigh_p_vertex_2d(o, pp, Mt)
            i2, j2 = _face_vert_to_ij_2d(f2, pp2, ps2)
            uf_union!(vid(p, i1, j1), vid(p2, i2, j2))
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
    vertex_coords = Matrix{T}(undef, 2, Nv)
    written = falses(Nv)
    for p in 1:n_patches
        ps = skel.patches[p]
        for j in 0:ps.Mb, i in 0:ps.Ma
            id = final[vid(p, i, j)]
            written[id] && continue
            x, y = _patch_vertex_position_2d(ps, i, j)
            vertex_coords[1, id] = x
            vertex_coords[2, id] = y
            written[id] = true
        end
    end
    @assert all(written)

    # --- Per-element tables ------------------------------------------
    vertex_idx     = Matrix{Int}(undef, 4, Ne)
    neighbour      = zeros(Int32, 4, Ne)
    neighbour_face = zeros(Int8,  4, Ne)
    orientation    = zeros(Int8,  4, Ne)
    bdry           = zeros(Int8,  4, Ne)

    OPP = (Int8(2), Int8(1), Int8(4), Int8(3))

    for p in 1:n_patches
        ps = skel.patches[p]
        for b in 1:ps.Mb, a in 1:ps.Ma
            e = eid(p, a, b)
            # Gmsh-canonical 4 corner vertex ids.
            vertex_idx[1, e] = final[vid(p, a - 1, b - 1)]
            vertex_idx[2, e] = final[vid(p, a,     b - 1)]
            vertex_idx[3, e] = final[vid(p, a,     b    )]
            vertex_idx[4, e] = final[vid(p, a - 1, b    )]

            for (f, na, nb, at_patch_boundary) in (
                    (1, a - 1, b,     a == 1     ),
                    (2, a + 1, b,     a == ps.Ma ),
                    (3, a,     b - 1, b == 1     ),
                    (4, a,     b + 1, b == ps.Mb ),
                )
                if !at_patch_boundary
                    neighbour[f, e]      = eid(p, na, nb)
                    neighbour_face[f, e] = OPP[f]
                    orientation[f, e]    = Int8(0)
                else
                    link = skel.faces[f, p]
                    if link.kind === :boundary
                        bdry[f, e] = link.boundary_tag
                    else
                        p2 = link.neigh_patch
                        f2 = link.neigh_face
                        o  = Int(link.orientation)
                        ps2 = skel.patches[p2]
                        Mt = _face_tangent_count_2d(f, ps)
                        p_cell = _face_cell_to_p_2d(f, a, b)
                        p2c = _neigh_p_cell_2d(o, p_cell, Mt)
                        a2, b2 = _face_cell_to_ab_2d(f2, p2c, ps2)
                        neighbour[f, e]      = eid(p2, a2, b2)
                        neighbour_face[f, e] = Int8(f2)
                        orientation[f, e]    = Int8(o)
                    end
                end
            end
        end
    end

    return QuadMesh{T}(Ne, neighbour, neighbour_face, orientation, bdry,
                       vertex_coords, vertex_idx)
end
