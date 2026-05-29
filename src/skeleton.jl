# ----------------------------------------------------------------------
# Skeleton-based mesh construction
#
# Two-stage mesh build that uses *integer-only* dedup for vertex
# identification. Stage 1 builds a `SkeletonMesh` — a small structure
# carrying the patch list and inter-patch face connectivity at the
# combinatorial level (no floating-point coordinates). Stage 2
# (`_skeleton_to_mesh`) enumerates per-patch vertices as integer
# 4-tuples `(p, i, j, k)`, unifies face-shared ids via union-find,
# assigns dense canonical ids, and only then evaluates the family-
# specific parametric map to produce coordinates.
#
# This replaces the earlier position-keyed `Dict{NTuple{3, T}, Int}`
# dedup, which broke at M=4 inflated cube because cube and patch
# vertex positions computed via different floating-point expressions
# rounded to 1-ULP-different values for non-power-of-2 divisions
# (e.g. `0.05000000000000002` vs `0.05` for `L = 0.1`). Integer
# dedup eliminates that class of bug entirely.
#
# Step 1 of the cleanup wires `make_cubical_mesh` through this path;
# the cubed cube and inflated cube builders follow in later steps.

"""
    PatchSpec{T}

Combinatorial + parametric description of one patch in a multi-block
hex mesh.

# Fields

* `Ma, Mb, Mc :: Int` — element counts along the patch's three local
  axes. The patch contains `Ma·Mb·Mc` elements and
  `(Ma+1)·(Mb+1)·(Mc+1)` pre-dedup vertices.
* `family :: PatchKind3D` — selects the parametric vertex map used at
  coordinate-assignment time. One of the `PatchKind3D` enum members:
  `Cubical_3D`, `InflationPos/NegX/Y/Z_3D` (inflation patches in the
  13-patch inflated cube), `ShellPos/NegX/Y/Z_3D` (outer shells of the
  inflated cube), or `WedgePos/NegX/Y/Z_3D` (radial wedges of the
  7-patch cubed cube). The `family` field doubles as the
  `PatchInfo.kind` tag stored per element by `make_inflated_cube_mesh`
  (only inflation, shell, and cubical values appear there in practice).
* `a_lo, a_hi, b_lo, b_hi, c_lo, c_hi :: T` — affine ranges in the
  patch's local `(a, b, c)` parameter space. The reference cube
  `[0, 1]³` is mapped to these before the family-specific transform.
* `L, R1, R2 :: T` — analytic constants used by curvilinear families;
  zero for `:cubical`. Stored on every `PatchSpec` so each patch
  carries everything it needs to evaluate its own vertices.
"""
struct PatchSpec{T}
    Ma     :: Int
    Mb     :: Int
    Mc     :: Int
    family :: PatchKind3D
    a_lo   :: T
    a_hi   :: T
    b_lo   :: T
    b_hi   :: T
    c_lo   :: T
    c_hi   :: T
    L      :: T
    R1     :: T
    R2     :: T
end

"""
    FaceLink

One entry in a `SkeletonMesh`'s 6×n_patches face-connectivity table.
Two flavours, selected by `kind`:

* `kind = :interior` — face is shared with another patch face. Carries
  `(neigh_patch, neigh_face, orientation ∈ 0..7)`.
* `kind = :boundary` — face is on the domain boundary. Carries only
  `boundary_tag ∈ 1..127`.

Use `interior_link(np, nf, o)` / `boundary_link(tag)` to construct.
"""
struct FaceLink
    kind         :: Symbol
    neigh_patch  :: Int
    neigh_face   :: Int
    orientation  :: Int8
    boundary_tag :: Int8
end

interior_link(np::Integer, nf::Integer, o::Integer) =
    FaceLink(:interior, Int(np), Int(nf), Int8(o), Int8(0))
boundary_link(tag::Integer) =
    FaceLink(:boundary, 0, 0, Int8(0), Int8(tag))

"""
    SkeletonMesh{T}

Patch list + 6×n_patches face-link table. `_skeleton_to_mesh(skel)`
instantiates the full `HexMesh{T}` from this skeleton.
"""
struct SkeletonMesh{T}
    patches :: Vector{PatchSpec{T}}
    faces   :: Matrix{FaceLink}
end

# ----- Per-face index helpers (skeleton scope) ------------------------

# Element counts along the two tangent axes of face `f` in patch `ps`.
@inline function _face_tangent_counts(f::Integer, ps::PatchSpec)
    if f == 1 || f == 2
        return (ps.Mb, ps.Mc)
    elseif f == 3 || f == 4
        return (ps.Ma, ps.Mc)
    else                       # f == 5 or 6
        return (ps.Ma, ps.Mb)
    end
end

# Face-local 0-based vertex `(pp, qq)` → patch vertex `(i, j, k)`.
@inline function _face_vert_to_ijk(f::Integer, pp::Integer, qq::Integer,
                                     ps::PatchSpec)
    if f == 1
        return (0,     pp,    qq   )
    elseif f == 2
        return (ps.Ma, pp,    qq   )
    elseif f == 3
        return (pp,    0,     qq   )
    elseif f == 4
        return (pp,    ps.Mb, qq   )
    elseif f == 5
        return (pp,    qq,    0    )
    else
        return (pp,    qq,    ps.Mc)
    end
end

# Element `(a, b, c)` projected onto face `f` → face-cell `(p_cell, q_cell)`.
@inline function _face_cell_to_pq(f::Integer, a::Integer, b::Integer, c::Integer)
    if f == 1 || f == 2
        return (b, c)
    elseif f == 3 || f == 4
        return (a, c)
    else
        return (a, b)
    end
end

# Face-cell `(p_cell, q_cell)` on face `f` → element `(a, b, c)`.
@inline function _face_cell_to_abc(f::Integer, p_cell::Integer, q_cell::Integer,
                                    ps::PatchSpec)
    if f == 1
        return (1,      p_cell, q_cell)
    elseif f == 2
        return (ps.Ma,  p_cell, q_cell)
    elseif f == 3
        return (p_cell, 1,      q_cell)
    elseif f == 4
        return (p_cell, ps.Mb,  q_cell)
    elseif f == 5
        return (p_cell, q_cell, 1     )
    else
        return (p_cell, q_cell, ps.Mc )
    end
end

# D₄ transform on 0-indexed face-vertex coordinates `(p, q) ∈ 0..Mt1 × 0..Mt2`.
# Even `o` preserves the (p, q) dim ordering, odd `o` swaps it
# (so the neighbour's tangent counts come out as `(Mt2, Mt1)`).
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

# Same D₄ transform on 1-indexed face cells, `(b, c) ∈ 1..Mt1 × 1..Mt2`,
# used to identify the neighbour element across a cross-patch face link.
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

# Family-dispatched coordinate map: given integer vertex
# `(i, j, k) ∈ 0..Ma × 0..Mb × 0..Mc` of patch `ps`, return the
# physical `(x, y, z)`. Each family interprets `(a_lo, a_hi, b_lo,
# b_hi, c_lo, c_hi)` and `(L, R1, R2)` according to its own
# parameterisation.
function _patch_vertex_position(ps::PatchSpec{T}, i::Integer,
                                  j::Integer, k::Integer) where {T}
    fam = ps.family
    if is_cubical(fam)
        ξ = T(i) / T(ps.Ma)
        η = T(j) / T(ps.Mb)
        ζ = T(k) / T(ps.Mc)
        return (ps.a_lo + (ps.a_hi - ps.a_lo) * ξ,
                ps.b_lo + (ps.b_hi - ps.b_lo) * η,
                ps.c_lo + (ps.c_hi - ps.c_lo) * ζ)
    elseif is_wedge(fam)
        # Radial wedge (cubed-cube outer patch). Geometric radial
        # spacing `r(a) = R1·(R2/R1)^a` so each cell stays roughly
        # cubical; angular axes `(b, c) ∈ [-1, 1]²`.
        a = ps.a_lo + (ps.a_hi - ps.a_lo) * (T(i) / T(ps.Ma))
        b = ps.b_lo + (ps.b_hi - ps.b_lo) * (T(j) / T(ps.Mb))
        c = ps.c_lo + (ps.c_hi - ps.c_lo) * (T(k) / T(ps.Mc))
        r = ps.R1 * (ps.R2 / ps.R1)^a
        dir = direction_of(fam)
        if     dir == Int8(1);  return ( r,   b*r, c*r)
        elseif dir == Int8(2);  return (-r,   b*r, c*r)
        elseif dir == Int8(3);  return (b*r,  r,   c*r)
        elseif dir == Int8(4);  return (b*r, -r,   c*r)
        elseif dir == Int8(5);  return (b*r,  c*r,  r)
        else                    return (b*r,  c*r, -r)
        end
    else
        # Inflation / shell families (13-patch inflated cube).
        # Uses the right-handed `_patch_direction_vec` (with axis swaps
        # for `-x, +y, -z`) — the corresponding non-trivial D₄
        # orientations are encoded in the skeleton's face-link table.
        a = ps.a_lo + (ps.a_hi - ps.a_lo) * (T(i) / T(ps.Ma))
        b = ps.b_lo + (ps.b_hi - ps.b_lo) * (T(j) / T(ps.Mb))
        c = ps.c_lo + (ps.c_hi - ps.c_lo) * (T(k) / T(ps.Mc))
        Q = sqrt(one(T) + (b * b + c * c))
        dir = direction_of(fam)
        vx, vy, vz = _patch_direction_vec(dir, b, c)
        if is_shell(fam)
            r = (one(T) - a) * ps.R1 + a * ps.R2
            f = r / Q
        else               # inflation
            f = (one(T) - a) * ps.L + a * ps.R1 / Q
        end
        return (f * vx, f * vy, f * vz)
    end
end

"""
    _skeleton_to_mesh(skel::SkeletonMesh{T}) → HexMesh{T}

Instantiate the full element-level `HexMesh{T}` from a `SkeletonMesh`:

1. Per-patch pre-dedup vertex ids `(p, i, j, k)`.
2. Union-find over face-shared ids using the skeleton's interior
   `FaceLink`s and the integer D₄ orientation transform.
3. Dense canonical vertex ids `1..Nv`.
4. Coordinates from `_patch_vertex_position` at one representative
   per canonical id.
5. Per-element `vertex_idx`, `neighbour`, `neighbour_face`,
   `orientation`, `bdry` tables — within-patch faces use trivial
   sibling-element connectivity; cross-patch faces inherit
   `(neighbour_face, orientation)` from the skeleton.

No floating-point comparison is ever load-bearing.
"""
function _skeleton_to_mesh(skel::SkeletonMesh{T}) where {T}
    n_patches = length(skel.patches)
    @assert size(skel.faces) == (6, n_patches)

    # Pre-dedup vertex / element offsets per patch.
    vert_offs = Vector{Int}(undef, n_patches + 1)
    elem_offs = Vector{Int}(undef, n_patches + 1)
    vert_offs[1] = 0
    elem_offs[1] = 0
    for p in 1:n_patches
        ps = skel.patches[p]
        vert_offs[p+1] = vert_offs[p] + (ps.Ma + 1) * (ps.Mb + 1) * (ps.Mc + 1)
        elem_offs[p+1] = elem_offs[p] + ps.Ma * ps.Mb * ps.Mc
    end
    Nv_pre = vert_offs[end]
    Ne     = elem_offs[end]

    # Pre-dedup vertex id for `(p, i, j, k)` and element id for `(p, a, b, c)`.
    @inline function vid(p, i, j, k)
        ps = skel.patches[p]
        return vert_offs[p] + 1 + i + (ps.Ma + 1) * (j + (ps.Mb + 1) * k)
    end
    @inline function eid(p, a, b, c)
        ps = skel.patches[p]
        return elem_offs[p] + a + ps.Ma * ((b - 1) + ps.Mb * (c - 1))
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
    # processes each pair exactly once even when the skeleton lists both
    # halves (which it normally does, for symmetry).
    for p in 1:n_patches, f in 1:6
        link = skel.faces[f, p]
        link.kind === :interior || continue
        p2 = link.neigh_patch
        f2 = link.neigh_face
        ((p, f) < (p2, f2)) || continue
        o   = Int(link.orientation)
        ps  = skel.patches[p]
        ps2 = skel.patches[p2]
        Mt1, Mt2 = _face_tangent_counts(f, ps)
        for qq in 0:Mt2, pp in 0:Mt1
            i1, j1, k1 = _face_vert_to_ijk(f, pp, qq, ps)
            pp2, qq2 = _neigh_pq_vertex(o, pp, qq, Mt1, Mt2)
            i2, j2, k2 = _face_vert_to_ijk(f2, pp2, qq2, ps2)
            uf_union!(vid(p, i1, j1, k1), vid(p2, i2, j2, k2))
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
    vertex_coords = Matrix{T}(undef, 3, Nv)
    written = falses(Nv)
    for p in 1:n_patches
        ps = skel.patches[p]
        for k in 0:ps.Mc, j in 0:ps.Mb, i in 0:ps.Ma
            id = final[vid(p, i, j, k)]
            written[id] && continue
            x, y, z = _patch_vertex_position(ps, i, j, k)
            vertex_coords[1, id] = x
            vertex_coords[2, id] = y
            vertex_coords[3, id] = z
            written[id] = true
        end
    end
    @assert all(written)

    # --- Per-element tables ------------------------------------------
    vertex_idx     = Matrix{Int}(undef, 8, Ne)
    neighbour      = zeros(Int32, 6, Ne)
    neighbour_face = zeros(Int8,  6, Ne)
    orientation    = zeros(Int8,  6, Ne)
    bdry           = zeros(Int8,  6, Ne)

    # Opposite face along the same axis: 1↔2 (-x/+x), 3↔4 (-y/+y), 5↔6 (-z/+z).
    OPP = (Int8(2), Int8(1), Int8(4), Int8(3), Int8(6), Int8(5))

    for p in 1:n_patches
        ps = skel.patches[p]
        for c in 1:ps.Mc, b in 1:ps.Mb, a in 1:ps.Ma
            e = eid(p, a, b, c)
            # Gmsh-canonical 8 corner vertex ids.
            vertex_idx[1, e] = final[vid(p, a - 1, b - 1, c - 1)]
            vertex_idx[2, e] = final[vid(p, a,     b - 1, c - 1)]
            vertex_idx[3, e] = final[vid(p, a,     b,     c - 1)]
            vertex_idx[4, e] = final[vid(p, a - 1, b,     c - 1)]
            vertex_idx[5, e] = final[vid(p, a - 1, b - 1, c    )]
            vertex_idx[6, e] = final[vid(p, a,     b - 1, c    )]
            vertex_idx[7, e] = final[vid(p, a,     b,     c    )]
            vertex_idx[8, e] = final[vid(p, a - 1, b,     c    )]

            # Per-face neighbour / orientation / bdry.
            for (f, na, nb, nc, at_patch_boundary) in (
                    (1, a - 1, b,     c,     a == 1     ),
                    (2, a + 1, b,     c,     a == ps.Ma ),
                    (3, a,     b - 1, c,     b == 1     ),
                    (4, a,     b + 1, c,     b == ps.Mb ),
                    (5, a,     b,     c - 1, c == 1     ),
                    (6, a,     b,     c + 1, c == ps.Mc ),
                )
                if !at_patch_boundary
                    neighbour[f, e]      = eid(p, na, nb, nc)
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
                        Mt1, Mt2 = _face_tangent_counts(f, ps)
                        p_cell, q_cell = _face_cell_to_pq(f, a, b, c)
                        p2c, q2c = _neigh_pq_cell(o, p_cell, q_cell, Mt1, Mt2)
                        a2, b2, c2 = _face_cell_to_abc(f2, p2c, q2c, ps2)
                        neighbour[f, e]      = eid(p2, a2, b2, c2)
                        neighbour_face[f, e] = Int8(f2)
                        orientation[f, e]    = Int8(o)
                    end
                end
            end
        end
    end

    return HexMesh{T}(Ne, neighbour, neighbour_face, orientation, bdry,
                      vertex_coords, vertex_idx)
end

"""
    make_cubical_mesh(::Type{T}, Mx, My, Mz, x0, x1) → HexMesh{T}
    make_cubical_mesh(::Type{T}, M, x0, x1)         → HexMesh{T}

Axis-aligned conforming hex mesh of the cuboid `[x0, x1]³` with
`Mx × My × Mz` (or `M × M × M`) elements. Backed by the skeleton-based
build (`_skeleton_to_mesh`) over a single `:cubical` patch with all
six faces tagged as domain boundary `1..6` (matching the face index
ordering, as before).

Element ordering remains column-major over `(mx, my, mz)`:

    e(mx, my, mz) = mx + (my-1)·Mx + (mz-1)·Mx·My

`orientation` is identically zero (axis-aligned, single patch).
"""
