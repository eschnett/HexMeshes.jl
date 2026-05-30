"""
    make_uniform_hex(::Type{T}, Mx::Int, My::Int, Mz::Int, x0, x1;
                     periodic = (false, false, false)) → Mesh{3, T}
    make_uniform_hex(::Type{T}, M::Int, x0, x1; periodic = false) → Mesh{3, T}

Uniform 3D mesh of axis-aligned hexahedral elements on the cube
`[x0, x1]³`, with `Mx · My · Mz` (or `M³` for the equal-count
shorthand) elements. Backed by the skeleton-based build over a single
`PatchCubic{3, T}`.

Non-periodic axes have their two outer faces tagged as domain boundary
(face indices `1..6` matching `(-x, +x, -y, +y, -z, +z)`). Axes flagged
periodic have their two outer faces wired as `PeriodicLink`s into a
torus topology — opposite faces become topological neighbours
(`bdry == 0`, `neighbour` pointing across the seam) while their
vertices stay at their distinct physical positions.

The `periodic` kwarg accepts either a single `Bool` (applied to all
three axes) or a 3-tuple of `Bool`s indexed by axis `(x, y, z)`.

Element ordering is column-major over `(mx, my, mz)`:

    e(mx, my, mz) = mx + (my-1)·Mx + (mz-1)·Mx·My

`orientation` is identically zero (axis-aligned, single patch).
"""
function make_uniform_hex(::Type{T}, Mx::Int, My::Int, Mz::Int, x0, x1;
                            periodic = (false, false, false)) where {T}
    @assert Mx ≥ 1 && My ≥ 1 && Mz ≥ 1
    per = periodic isa Bool ? (periodic, periodic, periodic) : periodic
    @assert length(per) == 3
    cubic = PatchCubic{3, T}((Mx, My, Mz),
                              (T(x0), T(x0), T(x0)),
                              (T(x1), T(x1), T(x1)))
    patch = PatchDesc(cubic)
    faces = Matrix{FaceLink}(undef, 6, 1)
    for f in 1:6
        axis = (f + 1) >> 1                  # 1, 1, 2, 2, 3, 3
        opp  = isodd(f) ? f + 1 : f - 1      # face-axis opposite end
        if per[axis]
            faces[f, 1] = periodic_link(1, opp, 0)
        else
            faces[f, 1] = boundary_link(f)
        end
    end
    skel = SkeletonMesh{3, T}([patch], faces)
    return _skeleton_to_mesh(skel)
end

# Cubic convenience: equal element count in each direction.
make_uniform_hex(::Type{T}, M::Int, x0, x1; periodic = false) where {T} =
    make_uniform_hex(T, M, M, M, x0, x1; periodic = periodic)

# Deprecated aliases. New code should call `make_uniform_hex`.
Base.@deprecate make_cubical_mesh(::Type{T}, Mx::Int, My::Int, Mz::Int, x0, x1) where {T} (
    make_uniform_hex(T, Mx, My, Mz, x0, x1))
Base.@deprecate make_cubical_mesh(::Type{T}, M::Int, x0, x1) where {T} (
    make_uniform_hex(T, M, x0, x1))

"""
    make_cubed_cube_mesh(::Type{T}, M::Int, R::Real) → Mesh{3, T}

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
    inner = PatchDesc(PatchCubic{3, T}((M, M, M),
                                         (-Rv, -Rv, -Rv),
                                         ( Rv,  Rv,  Rv)))

    # Patches 2..7: outer wedges in the dir order (+x, -x, +y, -y, +z, -z).
    # All wedges share the parameter ranges `a ∈ [0, 1]`, `b ∈ [-1, 1]`,
    # `c ∈ [-1, 1]`; `dir` selects the embedding direction. `(R1, R2) =
    # (R, 1)` is used by `_patch_vertex_position` to compute
    # `r(a) = R1 · (R2/R1)^a = R · (1/R)^a`.
    patches = PatchDesc{3, T}[inner]
    for dir in 1:6
        push!(patches, PatchDesc(PatchWedge{3, T}((L, M, M), Int8(dir),
                                                    z, o, -o, o, -o, o,
                                                    Rv, o)))
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

    return SkeletonMesh{3, T}(patches, faces)
end

"""
    make_cubed_cube_mesh(::Type{T}, M::Int, R::Real) → Mesh{3, T}

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
