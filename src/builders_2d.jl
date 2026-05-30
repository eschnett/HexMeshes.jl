# ----------------------------------------------------------------------
# 2D quadrilateral mesh builders.
#
# `make_uniform_quad` is a single-patch uniform mesh, built directly
# without skeleton infrastructure (multi-patch dedup unnecessary for
# a single patch).
#
# `make_cubed_square_mesh` and the inflated-square family use the
# unified D-generic skeleton machinery in `skeleton.jl`.

"""
    make_uniform_quad(::Type{T}, Mx::Int, My::Int, x0, x1;
                      periodic = (false, false)) → Mesh{2, T}

Uniform 2D mesh of `Mx · My` quadrilateral elements on the square
`[x0, x1]²`. Vertices are arranged on a regular `(Mx+1) × (My+1)`
grid; each element's four corners are stored in Gmsh-canonical order
`((−x, −y), (+x, −y), (+x, +y), (−x, +y))`.

# Boundary tags

* Face 1 (−x): elements with `mx = 1`, tagged `Int8(1)`
* Face 2 (+x): elements with `mx = Mx`, tagged `Int8(2)`
* Face 3 (−y): elements with `my = 1`, tagged `Int8(3)`
* Face 4 (+y): elements with `my = My`, tagged `Int8(4)`

`periodic` accepts a single `Bool` (applied to both axes) or a 2-tuple
`(px, py)`. Periodic axes have their outer faces wired into a torus
topology (`bdry == 0`, `neighbour` wraps around modulo the axis
count) while vertex coordinates stay at their physical positions.
"""
function make_uniform_quad(::Type{T}, Mx::Int, My::Int, x0, x1;
                            periodic = (false, false)) where {T}
    @assert Mx ≥ 1 && My ≥ 1
    per = periodic isa Bool ? (periodic, periodic) : periodic
    @assert length(per) == 2
    px, py = per
    Ne = Mx * My
    hx = (T(x1) - T(x0)) / T(Mx)
    hy = (T(x1) - T(x0)) / T(My)

    # Vertex grid (vx ∈ 0..Mx, vy ∈ 0..My), 1-indexed.
    Nv = (Mx + 1) * (My + 1)
    vertex_coords = Matrix{T}(undef, 2, Nv)
    vid(vx, vy) = vx + 1 + vy * (Mx + 1)
    for vy in 0:My, vx in 0:Mx
        v = vid(vx, vy)
        vertex_coords[1, v] = T(x0) + vx * hx
        vertex_coords[2, v] = T(x0) + vy * hy
    end

    # Element connectivity (mx, my) ∈ 1..Mx × 1..My, 1-indexed.
    eid(mx, my) = mx + (my - 1) * Mx
    vertex_idx = Matrix{Int}(undef, 4, Ne)
    for my in 1:My, mx in 1:Mx
        e = eid(mx, my)
        # Gmsh-canonical vertex order: (−x, −y), (+x, −y), (+x, +y), (−x, +y).
        vertex_idx[1, e] = vid(mx - 1, my - 1)
        vertex_idx[2, e] = vid(mx,     my - 1)
        vertex_idx[3, e] = vid(mx,     my)
        vertex_idx[4, e] = vid(mx - 1, my)
    end

    # Face connectivity (face ∈ 1..4 = −x, +x, −y, +y).
    neighbour      = Matrix{Int32}(undef, 4, Ne)
    neighbour_face = Matrix{Int8}(undef, 4, Ne)
    orientation    = Matrix{Int8}(undef, 4, Ne)
    bdry           = Matrix{Int8}(undef, 4, Ne)
    fill!(orientation, Int8(0))      # axis-aligned ⇒ identity
    fill!(bdry,         Int8(0))     # interior by default
    for my in 1:My, mx in 1:Mx
        e = eid(mx, my)
        # Face 1 (−x)
        if mx == 1
            if px
                neighbour[1, e]      = Int32(eid(Mx, my))
                neighbour_face[1, e] = Int8(2)
            else
                neighbour[1, e]      = Int32(0)
                neighbour_face[1, e] = Int8(0)
                bdry[1, e]           = Int8(1)
            end
        else
            neighbour[1, e]      = Int32(eid(mx - 1, my))
            neighbour_face[1, e] = Int8(2)
        end
        # Face 2 (+x)
        if mx == Mx
            if px
                neighbour[2, e]      = Int32(eid(1, my))
                neighbour_face[2, e] = Int8(1)
            else
                neighbour[2, e]      = Int32(0)
                neighbour_face[2, e] = Int8(0)
                bdry[2, e]           = Int8(2)
            end
        else
            neighbour[2, e]      = Int32(eid(mx + 1, my))
            neighbour_face[2, e] = Int8(1)
        end
        # Face 3 (−y)
        if my == 1
            if py
                neighbour[3, e]      = Int32(eid(mx, My))
                neighbour_face[3, e] = Int8(4)
            else
                neighbour[3, e]      = Int32(0)
                neighbour_face[3, e] = Int8(0)
                bdry[3, e]           = Int8(3)
            end
        else
            neighbour[3, e]      = Int32(eid(mx, my - 1))
            neighbour_face[3, e] = Int8(4)
        end
        # Face 4 (+y)
        if my == My
            if py
                neighbour[4, e]      = Int32(eid(mx, 1))
                neighbour_face[4, e] = Int8(3)
            else
                neighbour[4, e]      = Int32(0)
                neighbour_face[4, e] = Int8(0)
                bdry[4, e]           = Int8(4)
            end
        else
            neighbour[4, e]      = Int32(eid(mx, my + 1))
            neighbour_face[4, e] = Int8(3)
        end
    end

    # Patch metadata: single Cubic patch covering [x0, x1]².
    patch_desc = [PatchDesc(PatchCubic{2, T}((Mx, My),
                                              (T(x0), T(x0)),
                                              (T(x1), T(x1))))]
    patch_id   = fill(Int32(1), Ne)
    patch_idx  = Matrix{Int32}(undef, 2, Ne)
    for my in 1:My, mx in 1:Mx
        e = eid(mx, my)
        patch_idx[1, e] = Int32(mx)
        patch_idx[2, e] = Int32(my)
    end
    patch_element_offset = [0, Ne]

    return Mesh{2, T}(Ne, neighbour, neighbour_face, orientation, bdry,
                       vertex_coords, vertex_idx;
                       patch_id              = patch_id,
                       patch_idx             = patch_idx,
                       patch_desc            = patch_desc,
                       patch_element_offset  = patch_element_offset)
end

# Square shorthand: equal element count in both directions.
make_uniform_quad(::Type{T}, M::Int, x0, x1; periodic = false) where {T} =
    make_uniform_quad(T, M, M, x0, x1; periodic = periodic)

# Deprecated alias. New code should call `make_uniform_quad`.
Base.@deprecate make_quad_mesh(::Type{T}, Mx::Int, My::Int, x0, x1) where {T} (
    make_uniform_quad(T, Mx, My, x0, x1))
Base.@deprecate make_quad_mesh(::Type{T}, M::Int, x0, x1) where {T} (
    make_uniform_quad(T, M, x0, x1))

"""
    make_cubed_square_mesh(::Type{T}, M::Int, R::Real) → Mesh{2, T}

Conforming quadrilateral mesh of the square `[-1, 1]²` built from a
"cubed-sphere"-style block topology applied to a square domain: one
central square patch `[-R, R]²` plus four radial-wedge patches
connecting it to the four outer edges. All outer edges are flat — the
overall shape is still a square; the geometry has cubed-square
topology but a square codomain. (The name `inflated_square` is reserved
for the variant where the outer boundary is rounded into a disk.)

`M` is the angular resolution; each wedge has `L` radial cells where
`L = max(1, round(log(1/R) / log(1 + 2/M)))` is chosen so each cell
stays roughly square.

# Geometry

For the +x wedge (others by axis permutation / reflection), with
`a ∈ [0, 1]` (radial) and `b ∈ [-1, 1]` (angular):

    r       = R · (1/R)^a
    (x, y)  = (r, b·r)

so at `a = 0`: the inner-square edge `(R, [-R, R])`; at `a = 1`: the
outer-square edge `(1, [-1, 1])`.

# Element count

`M²` cells in the inner patch + `4·L·M` cells in the four outer wedges.

# Orientation

By construction every shared edge has matching parameter direction,
so `orientation[f, e] = 0` everywhere.
"""
function make_cubed_square_mesh(::Type{T}, M::Int, R::Real) where {T}
    @assert M ≥ 1
    @assert 0 < R < 1
    Rv = T(R)
    L = max(1, round(Int, log(1/R) / log(1 + 2/M)))
    skel = _cubed_square_skeleton(T, M, L, Rv)
    return _skeleton_to_mesh(skel)
end

# Tangential-edge connectivity for the four radial directions of a
# cubed-square topology. Indexed by `dir ∈ 1..4` (+x, −x, +y, −y) and
# face `f ∈ 3..4` of the wedge:
#
#   `_WEDGE_NEIGHBOUR_2D[dir][f - 2] = (neigh_dir, neigh_face)`
#
# All eight corner-edge interfaces have orientation 0 because every
# wedge's tangent direction along the shared edge is parameterised
# by `a` (radial) running 0 → 1 inner → outer.
const _WEDGE_NEIGHBOUR_2D = (
    ((4, 4), (3, 4)),   # +x: faces 3,4 → wedge -y face 4, wedge +y face 4
    ((4, 3), (3, 3)),   # -x:           → wedge -y face 3, wedge +y face 3
    ((2, 4), (1, 4)),   # +y:           → wedge -x face 4, wedge +x face 4
    ((2, 3), (1, 3)),   # -y:           → wedge -x face 3, wedge +x face 3
)

function _cubed_square_skeleton(::Type{T}, M::Int, L::Int, Rv::T) where {T}
    z = zero(T)
    o = one(T)

    # Patch 1: inner square [-R, R]².
    inner = PatchDesc(PatchCubic{2, T}((M, M), (-Rv, -Rv), (Rv, Rv)))

    # Patches 2..5: outer wedges in dir order (+x, -x, +y, -y).
    # All wedges share `a ∈ [0, 1]`, `b ∈ [-1, 1]`; `dir` selects the
    # embedding direction. `(R1, R2) = (R, 1)` for the radial map.
    patches = PatchDesc{2, T}[inner]
    for dir in 1:4
        push!(patches, PatchDesc(PatchWedge{2, T}((L, M), Int8(dir),
                                                    z, o, -o, o, z, z,
                                                    Rv, o)))
    end

    faces = Matrix{FaceLink}(undef, 4, length(patches))

    # ---- Inner square ↔ wedge interfaces. ----
    # Inner face `f` ↔ wedge whose direction matches that face,
    # connecting at the wedge's face 1 (a = 0, inner-radial edge):
    #   f=1 (-x) → -x wedge (patch 3)
    #   f=2 (+x) → +x wedge (patch 2)
    #   f=3 (-y) → -y wedge (patch 5)
    #   f=4 (+y) → +y wedge (patch 4)
    inner_face_to_wedge = (3, 2, 5, 4)
    for f in 1:4
        wp = inner_face_to_wedge[f]
        faces[f, 1]  = interior_link(wp, 1, 0)
        faces[1, wp] = interior_link(1, f, 0)
    end

    # ---- Outer-edge boundary tags on each wedge's face 2. ----
    # Tag convention `-x=1, +x=2, -y=3, +y=4` keyed by wedge dir.
    OUTER_TAG = (Int8(2), Int8(1), Int8(4), Int8(3))
    for dir in 1:4
        faces[2, dir + 1] = boundary_link(OUTER_TAG[dir])
    end

    # ---- Wedge ↔ wedge tangential edges. ----
    for dir in 1:4
        wp = dir + 1
        for f in 3:4
            neigh_dir, neigh_face = _WEDGE_NEIGHBOUR_2D[dir][f - 2]
            faces[f, wp] = interior_link(neigh_dir + 1, neigh_face, 0)
        end
    end

    return SkeletonMesh{2, T}(patches, faces)
end
