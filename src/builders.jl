function make_cubical_mesh(::Type{T}, Mx::Int, My::Int, Mz::Int, x0, x1) where {T}
    @assert Mx ≥ 1 && My ≥ 1 && Mz ≥ 1
    z = zero(T)
    patch = PatchSpec{T}(Mx, My, Mz, Cubical_3D,
                          T(x0), T(x1), T(x0), T(x1), T(x0), T(x1),
                          z, z, z)
    faces = Matrix{FaceLink}(undef, 6, 1)
    for f in 1:6
        faces[f, 1] = boundary_link(f)
    end
    skel = SkeletonMesh{T}([patch], faces)
    return _skeleton_to_mesh(skel)
end

# Cubic convenience: equal element count in each direction.
make_cubical_mesh(::Type{T}, M::Int, x0, x1) where {T} =
    make_cubical_mesh(T, M, M, M, x0, x1)

"""
    make_cubed_cube_mesh(::Type{T}, M::Int, R::Real) → HexMesh{T}

Conforming hex mesh of the cube `[-1, 1]³` built from a "cubed-sphere"
block topology applied to a cubic domain: one central cubic patch
`[-R, R]³` plus six radial-wedge patches connecting it to the six outer
cube faces. All outer faces of the global domain are flat (the overall
shape is still a cube), so the *outer* mesh boundary is `[-1, 1]³`
exactly — the geometry has cubed-sphere topology but a cube codomain.
(The name `inflated_cube` is reserved for a future variant in which the
outer boundary is curved to a sphere.)

`M` is the mesh-resolution parameter: each of the seven patches is
subdivided into `M` cells along each non-radial axis (so the inner patch
has `M³` cells and each outer patch has `L·M²`).

# Geometry

For the +x patch (the other five are obtained by axis permutation /
reflection), with local indices `i ∈ 0..L`, `j ∈ 0..M`, `k ∈ 0..M` and
`s_j = -1 + 2j/M`, `t_k = -1 + 2k/M`:

    r          = R · α^i        (radial coordinate)
    (x, y, z)  = (r,  s_j·r,  t_k·r)

so the cross-section at radial level `i` is the square `[-r, r]²`, which
matches the inner cube's `[-R, R]²` face at `i = 0` and the outer cube's
`[-1, 1]²` face at `i = L`.

# Element count

`M³` cells in the inner patch + `6·L·M²` cells in the six outer patches.

# `L` and radial spacing (step 2 of the construction)

We want each outer-patch cell to be roughly cubical: angular width
`2r/M` should match the radial width `r_{i+1} − r_i`. With geometric
spacing `r_i = R·α^i` the cell aspect is constant in `α`:

    r_{i+1} - r_i = r_i · (α - 1),  angular size = 2 r_i / M

so isotropic ⇒ `α - 1 ≈ 2/M`. The radial endpoint constraint `r_L = 1`
fixes `α = (1/R)^(1/L)`, so we pick

    L = round( log(1/R) / log(1 + 2/M) )

and use the resulting `α`. For `M = 5`, `R = 0.1` this gives `L = 7`,
`α ≈ 1.389`.

# Orientation

By construction, every patch's local axes are oriented so that, at any
shared face, the (p, q) face-node coordinates on the two sides match
directly — `orientation[f, e] = 0` everywhere.
"""
# Tangential-face connectivity for the six radial directions of a
# cubed-cube topology. Indexed by direction `dir ∈ 1..6` and face
# `f ∈ 3..6`:
#
#   `_WEDGE_NEIGHBOUR[dir][f - 2] = (neigh_dir, neigh_face)`
#
# All twelve cube-edge interfaces have orientation 0 because every
# wedge's tangent (p, q) local axes point in the same physical
# (x, y, z) direction at the shared edge (cubed-cube wedges use the
# uniform no-axis-swap convention `v = (±1, b, c)` etc.).
const _WEDGE_NEIGHBOUR = (
    ((4, 4), (3, 4), (6, 4), (5, 4)),   # +x: faces 3,4,5,6 → -y +y -z +z
    ((4, 3), (3, 3), (6, 3), (5, 3)),   # -x:                → -y +y -z +z
    ((2, 4), (1, 4), (6, 6), (5, 6)),   # +y:                → -x +x -z +z
    ((2, 3), (1, 3), (6, 5), (5, 5)),   # -y:                → -x +x -z +z
    ((2, 6), (1, 6), (4, 6), (3, 6)),   # +z:                → -x +x -y +y
    ((2, 5), (1, 5), (4, 5), (3, 5)),   # -z:                → -x +x -y +y
)

function _cubed_cube_skeleton(::Type{T}, M::Int, R::Real) where {T}
    @assert M ≥ 1
    @assert 0 < R < 1
    Rv = T(R)

    # Radial element count `L` per outer patch — same heuristic as the
    # pre-skeleton implementation: pick `L` such that the cell aspect
    # `(α - 1) ≈ 2/M` with `α = (1/R)^(1/L)`.
    L = max(1, round(Int, log(1/R) / log(1 + 2/M)))

    z = zero(T)
    o = one(T)

    # Patch 1: inner cube `[-R, R]³`.
    inner = PatchSpec{T}(M, M, M, Cubical_3D,
                          -Rv, Rv, -Rv, Rv, -Rv, Rv,
                          z, z, z)

    # Patches 2..7: outer wedges in the dir order (+x, -x, +y, -y, +z, -z).
    # All wedges share the parameter ranges `a ∈ [0, 1]`, `b ∈ [-1, 1]`,
    # `c ∈ [-1, 1]`; the family selects the embedding direction. The
    # `(R1, R2)` slots hold the inner-cube half-edge and the outer-cube
    # half-edge (1), used by `_patch_vertex_position` to compute
    # `r(a) = R1 · (R2/R1)^a = R · (1/R)^a`.
    wedge_families = (WedgePosX_3D, WedgeNegX_3D,
                      WedgePosY_3D, WedgeNegY_3D,
                      WedgePosZ_3D, WedgeNegZ_3D)
    patches = PatchSpec{T}[inner]
    for fam in wedge_families
        push!(patches, PatchSpec{T}(L, M, M, fam,
                                     z, o, -o, o, -o, o,
                                     z, Rv, o))
    end

    faces = Matrix{FaceLink}(undef, 6, length(patches))

    # ---- Cube ↔ wedge interfaces. ----
    # Inner cube face `f` → wedge whose direction matches that face,
    # connecting at the wedge's face 1 (`a = 0`, inner-radial face).
    # Vertex layout aligns so all six cube↔wedge interfaces have
    # orientation 0.
    #
    #   f=1 (-x) → -x wedge (patch 3)
    #   f=2 (+x) → +x wedge (patch 2)
    #   f=3 (-y) → -y wedge (patch 5)
    #   f=4 (+y) → +y wedge (patch 4)
    #   f=5 (-z) → -z wedge (patch 7)
    #   f=6 (+z) → +z wedge (patch 6)
    cube_face_to_wedge_patch = (3, 2, 5, 4, 7, 6)
    for f in 1:6
        wp = cube_face_to_wedge_patch[f]
        faces[f, 1]  = interior_link(wp, 1, 0)
        faces[1, wp] = interior_link(1, f, 0)
    end

    # ---- Outer-cube boundary tags on each wedge's face 2. ----
    # Match the original convention: -x=1, +x=2, -y=3, +y=4, -z=5, +z=6,
    # indexed by direction `dir ∈ 1..6` of the wedge.
    OUTER_TAG = (Int8(2), Int8(1), Int8(4), Int8(3), Int8(6), Int8(5))
    for dir in 1:6
        faces[2, dir + 1] = boundary_link(OUTER_TAG[dir])
    end

    # ---- Wedge ↔ wedge tangential faces. ----
    # See `_WEDGE_NEIGHBOUR` (module-level constant) for the table.
    for dir in 1:6
        wp = dir + 1
        for f in 3:6
            neigh_dir, neigh_face = _WEDGE_NEIGHBOUR[dir][f - 2]
            faces[f, wp] = interior_link(neigh_dir + 1, neigh_face, 0)
        end
    end

    return SkeletonMesh{T}(patches, faces)
end

"""
    make_cubed_cube_mesh(::Type{T}, M::Int, R::Real) → HexMesh{T}

Conforming hex mesh of the cube `[-1, 1]³` built from a "cubed-sphere"
block topology applied to a cubic domain: one central cubic patch
`[-R, R]³` plus six radial-wedge patches connecting it to the six outer
cube faces. All outer faces of the global domain are flat (the overall
shape is still a cube), so the *outer* mesh boundary is `[-1, 1]³`
exactly — the geometry has cubed-sphere topology but a cube codomain.

`M` is the mesh-resolution parameter: each of the seven patches is
subdivided into `M` cells along each non-radial axis (so the inner patch
has `M³` cells and each outer patch has `L·M²`).

# Geometry

For the +x patch (the other five are obtained by axis permutation /
reflection), with local indices `i ∈ 0..L`, `j ∈ 0..M`, `k ∈ 0..M`,
parametric coords `a = i/L`, `b = -1 + 2j/M`, `c = -1 + 2k/M`:

    r          = R · (1/R)^a        (radial coordinate)
    (x, y, z)  = (r,  b·r,  c·r)

so the cross-section at radial level `i` is the square `[-r, r]²`,
matching the inner cube's `[-R, R]²` face at `a = 0` and the outer
cube's `[-1, 1]²` face at `a = 1`.

# Element count

`M³ + 6·L·M²` total — `M³` in the inner cube + `L·M²` per outer wedge.

# `L` and radial spacing

To keep each outer-patch cell roughly cubical, set radial-to-tangential
aspect to 1: `α - 1 ≈ 2/M` with `α = (1/R)^(1/L)`. Solving for the
endpoint constraint `r_L = 1` gives

    L = round( log(1/R) / log(1 + 2/M) )

For `M = 5, R = 0.1` this gives `L = 7, α ≈ 1.389`.

# Orientation

By construction, every patch's local axes are oriented so that, at any
shared face, the `(p, q)` face-node coordinates on the two sides match
directly — `orientation[f, e] = 0` everywhere.

Built via the skeleton path (`_cubed_cube_skeleton` →
`_skeleton_to_mesh`), so vertex deduplication and orientations are
combinatorial / integer-keyed — no floating-point comparison is
load-bearing.
"""
make_cubed_cube_mesh(::Type{T}, M::Int, R::Real) where {T} =
    _skeleton_to_mesh(_cubed_cube_skeleton(T, M, R))
