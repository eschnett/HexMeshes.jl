# ----------------------------------------------------------------------
# 13-patch inflated cube mesh — builder + topology only.
#
# Topology: one axis-aligned inner cube `[-L, L]³` + six inflation
# patches (cube face → inner sphere `r = R1`) + six spherical-shell
# patches (`R1 → R2`). All thirteen patches are conforming at their
# shared faces.
#
# The analytic Jacobian for curvilinear patches lives in
# `patch_jacobian.jl`; the analytic patch ↔ global maps and point
# locators live in `queries.jl`.

# Tangential-face connectivity for the inflation / shell patches of an
# inflated-cube mesh. Same indexing scheme as `_WEDGE_NEIGHBOUR` but
# DIFFERENT entries: the inflated cube's `_patch_direction_vec` is
# right-handed with axis-swaps for the negative-leading-axis directions
# (`-x: v = (-1, c, b)`, `+y: v = (c, 1, b)`, `-z: v = (c, b, -1)`),
# which changes which face of each neighbour patch meets a given cube
# edge. Derived once on paper from the patch parameterisations; all
# twelve cube-edge orientations remain 0 (verified by Gmsh vertex
# correspondence at each edge).
const _INFLATION_NEIGHBOUR = (
    ((4, 4), (3, 6), (6, 6), (5, 4)),   # +x: faces 3,4,5,6 → -y4 +y6 -z6 +z4
    ((6, 5), (5, 3), (4, 3), (3, 5)),   # -x:                → -z5 +z3 -y3 +y5
    ((6, 4), (5, 6), (2, 6), (1, 4)),   # +y:                → -z4 +z6 -x6 +x4
    ((2, 5), (1, 3), (6, 3), (5, 5)),   # -y:                → -x5 +x3 -z3 +z5
    ((2, 4), (1, 6), (4, 6), (3, 4)),   # +z:                → -x4 +x6 -y6 +y4
    ((4, 5), (3, 3), (2, 3), (1, 5)),   # -z:                → -y5 +y3 -x3 +x5
)

"""
    make_inflated_cube_mesh(::Type{T}, L, R1, R2, M; M_i, M_s, outer_bc) → Mesh{3, T}

Build a 13-patch inflated cube mesh of the ball `|x| ≤ R2`:

* an axis-aligned inner cube `[-L, L]³` with `M × M × M` elements,
* six inflation patches that interpolate from each cube face to the
  inner sphere `r = R1`, with `M_i × M × M` elements each,
* six spherical-shell patches that interpolate from the inner sphere to
  the outer sphere `r = R2`, with `M_s × M × M` elements each.

`M_i` defaults to `round((R1 - (1 + √3)/2 · L) / h)` where
`h = 2L/M`. `M_s` defaults to `round((R2 - R1) / h)`. Each defaults to
at least 1.

Passing `R2 = Inf` gives a **compactified** outer boundary: the shell maps
its outer face to spatial infinity i⁰ (`r(a) = R1/(1−a)`); `M_s` then
defaults to `M` radial cells.

The shell patches use `r(ρ) = (1 − ρ)·R1 + ρ·R2` along every radial
ray, so they have exactly constant radial spacing `(R2 − R1) / M_s`
and exactly uniform angular sampling in `(η, ζ) ∈ [-1, 1]²`. The
inflation patches use `r(s, η, ζ) = (1 − s)·L + s · R1 / √(1 + η² + ζ²)`,
so their radial spacing is constant on average (varies between cube-
face center and cube-face corner).

The outer boundary `r = R2` is tagged on every shell-patch outer face
according to `outer_bc`:

* `:dirichlet` (default) — `bdry = 1`.
* `:sommerfeld` — `bdry = 7` (first-order radiative absorbing BC).
* `:excision` (alias `:outflow`) — `bdry = 8` (pure outflow / no SAT).
  Rarely used for the inflated-cube outer boundary, but accepted for
  consistency with `make_radial_shell_mesh`.

The returned `Mesh{3, T}` has all four patch fields populated:
`patch_desc` carries 13 `PatchDesc{3, T}` entries (one `Cubic` + six
`Inflation` + six `Shell`), and the per-element `patch_id` / `patch_idx`
tables drive the analytic-Jacobian path in downstream `make_geometry`.
"""
function make_inflated_cube_mesh(::Type{T}, L::Real, R1::Real, R2::Real, M::Int;
                                  M_i::Union{Nothing, Int}=nothing,
                                  M_s::Union{Nothing, Int}=nothing,
                                  outer_bc::Symbol = :dirichlet) where {T}
    skel = _inflated_cube_skeleton(T, L, R1, R2, M; M_i, M_s, outer_bc)
    return _skeleton_to_mesh(skel)
end

function _inflated_cube_skeleton(::Type{T}, L::Real, R1::Real, R2::Real, M::Int;
                                   M_i::Union{Nothing, Int}=nothing,
                                   M_s::Union{Nothing, Int}=nothing,
                                   outer_bc::Symbol = :dirichlet) where {T}
    outer_bc_tag = outer_bc === :dirichlet ? Int8(1) :
                   outer_bc === :sommerfeld ? Int8(7) :
                   (outer_bc === :excision || outer_bc === :outflow) ? Int8(8) :
                   error("_inflated_cube_skeleton: outer_bc must be " *
                         ":dirichlet, :sommerfeld, or :excision " *
                         "(alias :outflow); got $(repr(outer_bc))")
    @assert M ≥ 1
    @assert L > 0
    @assert L * sqrt(3) < R1 "inner sphere R1 must enclose the cube corner (L·√3)"
    @assert R1 < R2

    Lv  = T(L)
    R1v = T(R1)
    R2v = T(R2)
    h   = 2L / M

    Mi = M_i === nothing ?
         max(1, round(Int, (R1 - (1 + sqrt(3))/2 * L) / h)) :
         M_i
    Ms = M_s !== nothing ? M_s :
         isinf(R2) ? M :                          # compactified: no finite extent
         max(1, round(Int, (R2 - R1) / h))
    @assert Mi ≥ 1
    @assert Ms ≥ 1

    z = zero(T)
    o = one(T)

    # Patch 1: inner cube `[-L, L]³`.
    inner = PatchDesc(PatchCubic{3, T}((M, M, M),
                                         (-Lv, -Lv, -Lv),
                                         ( Lv,  Lv,  Lv)))

    # Patches 2..7: inflation in directions (+x, -x, +y, -y, +z, -z),
    # parameter range `(a, b, c) ∈ [0, 1] × [-1, 1]²`.
    # Patches 8..13: shells in the same direction order.
    patches = PatchDesc{3, T}[inner]
    for dir in 1:6
        push!(patches, PatchDesc(PatchInflation{3, T}((Mi, M, M), Int8(dir),
                                                        z, o, -o, o, -o, o,
                                                        Lv, R1v)))
    end
    for dir in 1:6
        push!(patches, PatchDesc(PatchShell{3, T}((Ms, M, M), Int8(dir),
                                                    z, o, -o, o, -o, o,
                                                    R1v, R2v)))
    end
    @assert length(patches) == 13

    faces = Matrix{FaceLink}(undef, 6, length(patches))

    # ---- Cube ↔ inflation interfaces. ----
    # Inner cube face `f` connects to the inflation patch whose
    # direction matches that face, at the inflation's face 1 (inner-
    # radial, `a = 0`). With the right-handed `_patch_direction_vec`,
    # the conventions for `-x, +y, -z` swap the (b, c) → (η_phys, ζ_phys)
    # axis mapping relative to the cube, giving a D₄ transpose
    # (`o = 5`). The other three directions match directly (`o = 0`).
    CUBE_FACE_TO_DIR         = (2, 1, 4, 3, 6, 5)
    CUBE_FACE_ORIENTATION    = (Int8(5), Int8(0), Int8(0),
                                Int8(5), Int8(5), Int8(0))
    for f in 1:6
        d  = CUBE_FACE_TO_DIR[f]
        ip = d + 1                   # inflation patch id
        oo = CUBE_FACE_ORIENTATION[f]
        faces[f, 1]  = interior_link(ip, 1, oo)
        faces[1, ip] = interior_link(1,  f, oo)
    end

    # ---- Inflation tangential / radial-outer faces. ----
    for d in 1:6
        ip = d + 1                   # inflation patch
        sp = d + 7                   # shell patch (same direction)
        faces[2, ip] = interior_link(sp, 1, 0)
        for f in 3:6
            neigh_dir, neigh_face = _INFLATION_NEIGHBOUR[d][f - 2]
            faces[f, ip] = interior_link(neigh_dir + 1, neigh_face, 0)
        end
    end

    # ---- Shell tangential / outer-sphere faces. ----
    for d in 1:6
        sp = d + 7
        ip = d + 1
        faces[1, sp] = interior_link(ip, 2, 0)
        faces[2, sp] = boundary_link(outer_bc_tag)
        for f in 3:6
            neigh_dir, neigh_face = _INFLATION_NEIGHBOUR[d][f - 2]
            faces[f, sp] = interior_link(neigh_dir + 7, neigh_face, 0)
        end
    end

    return SkeletonMesh{3, T}(patches, faces)
end
