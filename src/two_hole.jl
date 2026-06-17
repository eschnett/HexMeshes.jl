# ----------------------------------------------------------------------
# 28-patch two-hole mesh — a disk `|x| ≤ R2` with two circular holes of
# radius `R1` centred at `(±d/2, 0)`. The canonical 2D binary-excision
# domain. Builder + topology only.
#
# Three regions (inside → outside):
#
#  1. Inner regions (one per hole): 4 `PatchInflation` patches mapping the
#     hole circle `R1` (the excision boundary) out to an axis-aligned
#     square of half-side `L` centred at the hole. The patches are
#     OFF-CENTRE (`center = (±d/2, 0)`) and use a REVERSED radial range
#     (`a_lo = 1`, `a_hi = 0`) so the structured radial index runs
#     circle→square and `det J > 0` (the geometric inverse of
#     `inflated_square`, where the square is *inside* the circle).
#
#  2. Intermediate region: an 8-block "butterfly" of straight-sided
#     `PatchBilinearQuad` blocks (4 per hole) connecting the two hole
#     squares to a big square of half-side `A`, with a shared seam at
#     `x = 0`. Assumes the holes are far enough apart (`L ≪ d/2`) that the
#     two seam blocks are well-shaped.
#
#  3. Outer region: 6 `PatchInflation` (big square `A` → circle `R_mid`) +
#     6 `PatchShell` (`R_mid` → `R2`). The top and bottom edges are split
#     in two at `x = 0` to match the butterfly's seam, so the perimeter is
#     6 segments (left 1, right 1, top 2, bottom 2).
#
# Interior face connectivity is fixed by the mesh TOPOLOGY — it does not
# depend on the geometry parameters, the resolution, or the floating-point
# precision — so it is stored as frozen integer tables (no floating-point
# comparison anywhere in the build; the skeleton stays purely combinatorial,
# which is what makes arbitrary distortion / precision safe). Each entry
# `(p, f, q, g, o)` links face `f` of patch `p` to face `g` of patch `q` with
# D₁ orientation `o ∈ {0, 1}`; the symmetric reverse link is set as well.
#
# The tables were generated once from an endpoint-matching pass and verified
# identical across very different geometries and `R2 = Inf`; the conformity
# (coordinate-consistency) and det-J tests pin them. They are tied to the
# patch build order in `_two_hole_skeleton`: holes 1–8, then the butterfly
# blocks, then 6 outer inflation + 6 shell as the last 12. Boundary faces
# (the two hole circles and the outer circle) are set separately.
# Face numbering: 1 = −x, 2 = +x, 3 = −y, 4 = +y.

# :separated — 28 patches (8 butterfly blocks); 49 interior interfaces.
const _TWO_HOLE_LINKS_SEPARATED = (
    (1, 2, 15, 3, 1), (1, 3, 4, 4, 0), (1, 4, 3, 3, 0), (2, 2, 9, 3, 1),
    (2, 3, 3, 4, 0), (2, 4, 4, 3, 0), (3, 2, 10, 3, 1), (4, 2, 11, 3, 1),
    (5, 2, 12, 3, 1), (5, 3, 8, 4, 0), (5, 4, 7, 3, 0), (6, 2, 16, 3, 1),
    (6, 3, 7, 4, 0), (6, 4, 8, 3, 0), (7, 2, 13, 3, 1), (8, 2, 14, 3, 1),
    (9, 1, 11, 2, 0), (9, 2, 10, 1, 0), (9, 4, 17, 1, 1), (10, 2, 15, 1, 0),
    (10, 4, 19, 1, 1), (11, 1, 15, 2, 0), (11, 4, 21, 1, 1), (12, 1, 13, 2, 0),
    (12, 2, 14, 1, 0), (12, 4, 18, 1, 1), (13, 1, 16, 2, 0), (13, 4, 20, 1, 1),
    (14, 2, 16, 1, 0), (14, 4, 22, 1, 1), (15, 4, 16, 4, 1), (17, 2, 23, 1, 0),
    (17, 3, 19, 4, 0), (17, 4, 21, 3, 0), (18, 2, 24, 1, 0), (18, 3, 22, 4, 0),
    (18, 4, 20, 3, 0), (19, 2, 25, 1, 0), (19, 3, 20, 4, 0), (20, 2, 26, 1, 0),
    (21, 2, 27, 1, 0), (21, 4, 22, 3, 0), (22, 2, 28, 1, 0), (23, 3, 25, 4, 0),
    (23, 4, 27, 3, 0), (24, 3, 28, 4, 0), (24, 4, 26, 3, 0), (25, 3, 26, 4, 0),
    (27, 4, 28, 3, 0))

# :touching — 26 patches (6 butterfly blocks; the hole squares meet at x = 0
# and the two seam blocks are dropped); 45 interior interfaces.
const _TWO_HOLE_LINKS_TOUCHING = (
    (1, 2, 6, 2, 1), (1, 3, 4, 4, 0), (1, 4, 3, 3, 0), (2, 2, 9, 3, 1),
    (2, 3, 3, 4, 0), (2, 4, 4, 3, 0), (3, 2, 10, 3, 1), (4, 2, 11, 3, 1),
    (5, 2, 12, 3, 1), (5, 3, 8, 4, 0), (5, 4, 7, 3, 0), (6, 3, 7, 4, 0),
    (6, 4, 8, 3, 0), (7, 2, 13, 3, 1), (8, 2, 14, 3, 1), (9, 1, 11, 2, 0),
    (9, 2, 10, 1, 0), (9, 4, 15, 1, 1), (10, 2, 13, 1, 0), (10, 4, 17, 1, 1),
    (11, 1, 14, 2, 0), (11, 4, 19, 1, 1), (12, 1, 13, 2, 0), (12, 2, 14, 1, 0),
    (12, 4, 16, 1, 1), (13, 4, 18, 1, 1), (14, 4, 20, 1, 1), (15, 2, 21, 1, 0),
    (15, 3, 17, 4, 0), (15, 4, 19, 3, 0), (16, 2, 22, 1, 0), (16, 3, 20, 4, 0),
    (16, 4, 18, 3, 0), (17, 2, 23, 1, 0), (17, 3, 18, 4, 0), (18, 2, 24, 1, 0),
    (19, 2, 25, 1, 0), (19, 4, 20, 3, 0), (20, 2, 26, 1, 0), (21, 3, 23, 4, 0),
    (21, 4, 25, 3, 0), (22, 3, 26, 4, 0), (22, 4, 24, 3, 0), (23, 3, 24, 4, 0),
    (25, 4, 26, 3, 0))

"""
    make_two_hole_mesh(::Type{T}, R1, R2, d, M; kwargs...) → Mesh{2, T}

Build a 28-patch conforming quad mesh of the disk `|x| ≤ R2` with two
circular holes of radius `R1` centred at `(±d/2, 0)` — the canonical 2D
binary-excision domain.

Regions: each hole is surrounded by 4 `Inflation` patches (circle → square
of half-side `L`); an 8-block `BilinearQuad` "butterfly" connects the two
hole squares to a big square of half-side `A`; and an outer ring of 6
`Inflation` + 6 `Shell` patches maps the big square out through circle
`R_mid` to the outer circle `R2`.

Geometry knobs (all default from `R1, R2, d`; override as needed):

* `L`     — hole-square half-side. Default `1.5·R1`. Requires `R1 < L < d/2`.
* `A`     — big-square half-side. Default `2·(d/2 + L)`. Requires `A > d/2 + L`.
* `R_mid` — inflation/shell interface radius. Default `1.5·√2·A`. Requires
  `√2·A < R_mid < R2`.

Passing `R2 = Inf` gives a **compactified** outer boundary: the outer shell
maps its outer face to spatial infinity i⁰ (`r(a) = R_mid/(1−a)`); `M_s`
then defaults to `M` radial cells.

Resolution: `M` is the number of cells along each hole-square edge (and so
the tangential resolution everywhere). Radial counts `M_h` (hole
inflation), `M_b` (butterfly), `M_i` (outer inflation), `M_s` (shell)
default from the hole-edge spacing `h = 2L/M`.

Boundary tagging (via `_shell_bc_tag`): the two hole circles use
`inner_bc` (default `:excision` → `bdry = 8`), the outer circle uses
`outer_bc` (default `:dirichlet` → `bdry = 1`).

`mode` selects the intermediate-region topology:

* `:separated` (default) — the 28-patch layout above, with two seam blocks
  spanning `x = 0`. Assumes the holes are far enough apart (`L ≪ d/2`) that
  those blocks are well-shaped.
* `:touching` — for close holes: the two hole squares meet directly at
  `x = 0` (which forces `L = d/2`, so `L` may not be set explicitly), the
  two seam blocks are dropped, and the mesh has **26 patches** (6 butterfly
  blocks). Requires `R1 < d/2`. Two valence-6 vertices appear at `(0, ±d/2)`.
"""
function make_two_hole_mesh(::Type{T}, R1::Real, R2::Real, d::Real, M::Int;
                            L::Union{Nothing, Real} = nothing,
                            A::Union{Nothing, Real} = nothing,
                            R_mid::Union{Nothing, Real} = nothing,
                            M_h::Union{Nothing, Int} = nothing,
                            M_b::Union{Nothing, Int} = nothing,
                            M_i::Union{Nothing, Int} = nothing,
                            M_s::Union{Nothing, Int} = nothing,
                            outer_bc::Symbol = :dirichlet,
                            inner_bc::Symbol = :excision,
                            mode::Symbol = :separated) where {T}
    skel = _two_hole_skeleton(T, R1, R2, d, M; L, A, R_mid,
                              M_h, M_b, M_i, M_s, outer_bc, inner_bc, mode)
    return _skeleton_to_mesh(skel)
end

function _two_hole_skeleton(::Type{T}, R1::Real, R2::Real, d::Real, M::Int;
                            L::Union{Nothing, Real} = nothing,
                            A::Union{Nothing, Real} = nothing,
                            R_mid::Union{Nothing, Real} = nothing,
                            M_h::Union{Nothing, Int} = nothing,
                            M_b::Union{Nothing, Int} = nothing,
                            M_i::Union{Nothing, Int} = nothing,
                            M_s::Union{Nothing, Int} = nothing,
                            outer_bc::Symbol = :dirichlet,
                            inner_bc::Symbol = :excision,
                            mode::Symbol = :separated) where {T}
    outer_tag = _shell_bc_tag(outer_bc; outer = true)
    inner_tag = _shell_bc_tag(inner_bc; outer = false)
    mode === :separated || mode === :touching ||
        error("make_two_hole_mesh: mode must be :separated or :touching, " *
              "got $(repr(mode))")
    touching = mode === :touching
    @assert M ≥ 1
    @assert R1 > 0
    @assert R2 > 0

    # Geometry in Float64 (counts use `round`, which MultiFloats lack).
    R1f = Float64(R1);  R2f = Float64(R2)
    sf  = Float64(d) / 2
    # :touching forces L = d/2 so the two hole squares meet at x = 0.
    if touching
        L === nothing ||
            error("make_two_hole_mesh: `L` is fixed to d/2 in :touching mode; " *
                  "do not pass it.")
        Lf = sf
    else
        Lf = L === nothing ? 1.5 * R1f : Float64(L)
    end
    Af  = A     === nothing ? 2.0 * (sf + Lf)    : Float64(A)
    Rmf = R_mid === nothing ? 1.5 * sqrt(2.0) * Af : Float64(R_mid)
    @assert R1f < Lf  "hole-square half-side L must exceed R1 (square encloses circle); in :touching mode this means R1 < d/2"
    touching || @assert Lf < sf  "holes overlap: need L < d/2 (use mode=:touching for close holes where the squares meet)"
    @assert Af > sf + Lf  "big square must contain the hole squares (A > d/2 + L)"
    @assert sqrt(2.0) * Af < Rmf  "R_mid must enclose the big-square corner (√2·A)"
    @assert Rmf < R2f  "R_mid must be inside the outer circle R2"

    hf = 2 * Lf / M
    Mh = M_h === nothing ? max(1, round(Int, (Lf - R1f) / hf)) : M_h
    Rb = M_b === nothing ? max(1, round(Int, (Af - (sf + Lf)) / hf)) : M_b
    Mi = M_i === nothing ? max(1, round(Int, (Rmf - (1 + sqrt(2.0)) / 2 * Af) / hf)) : M_i
    Ms = M_s !== nothing ? M_s :
         isinf(R2f) ? M :                          # compactified: no finite extent
         max(1, round(Int, (R2f - Rmf) / hf))
    @assert Mh ≥ 1 && Rb ≥ 1 && Mi ≥ 1 && Ms ≥ 1

    sv  = T(sf);  Lv = T(Lf);  Av = T(Af);  Rmv = T(Rmf)
    R1v = T(R1f); R2v = T(R2f)
    z = zero(T);  o = one(T)

    patches = PatchDesc{2, T}[]

    # --- 1. Hole inflations (patches 1–8). Reversed radial range so face 1
    #        (a = a_lo = 1) is the circle and face 2 (a = a_hi = 0) the
    #        square; `det J > 0`. Off-centre at (±s, 0). ----------------
    for cx in (-sv, sv)
        for dir in 1:4
            push!(patches,
                  PatchDesc(PatchInflation{2, T}((Mh, M), Int8(dir),
                                                 o, z, -o, o, z, z,
                                                 Lv, R1v, (cx, z))))
        end
    end

    # --- 2. Butterfly bilinear quads (patches 9–16). `bq` takes the two
    #        hole-square-edge endpoints (h0, h1) and the two outer endpoints
    #        (o0 pairs h0, o1 pairs h1) and builds a quad with corners
    #        (c00,c10,c11,c01) = (h0,h1,o1,o0): ξ runs along the hole edge
    #        (M cells), η runs along the spokes (Rb cells). It flips the
    #        ξ direction when needed so the winding is counter-clockwise
    #        (`det J > 0`) regardless of which side of the hole the block
    #        is on. ------------------------------------------------------
    function bq(h0, h1, o0, o1)
        cr = (h1[1] - h0[1]) * (o0[2] - h0[2]) - (h1[2] - h0[2]) * (o0[1] - h0[1])
        if cr < z
            h0, h1 = h1, h0
            o0, o1 = o1, o0
        end
        return PatchDesc(PatchBilinearQuad{2, T}((M, Rb), (h0, h1, o1, o0)))
    end
    # hole 1 (centre −s): left, top, bottom (+ right→seam unless touching)
    push!(patches, bq((-sv - Lv, -Lv), (-sv - Lv,  Lv), (-Av, -Av), (-Av,  Av)))  # L1
    push!(patches, bq((-sv - Lv,  Lv), (-sv + Lv,  Lv), (-Av,  Av), (  z,  Av)))  # T1
    push!(patches, bq((-sv - Lv, -Lv), (-sv + Lv, -Lv), (-Av, -Av), (  z, -Av)))  # B1
    # hole 2 (centre +s): right, top, bottom (+ left→seam unless touching)
    push!(patches, bq(( sv + Lv, -Lv), ( sv + Lv,  Lv), ( Av, -Av), ( Av,  Av)))  # R2
    push!(patches, bq(( sv - Lv,  Lv), ( sv + Lv,  Lv), (  z,  Av), ( Av,  Av)))  # T2
    push!(patches, bq(( sv - Lv, -Lv), ( sv + Lv, -Lv), (  z, -Av), ( Av, -Av)))  # B2
    # Seam blocks — only when the hole squares do NOT meet at x = 0. When
    # touching (L = s) the squares share their x = 0 edge directly, the
    # T1/T2 and B1/B2 spokes meet on x = 0, and these two blocks vanish.
    if !touching
        push!(patches, bq((-sv + Lv, -Lv), (-sv + Lv, Lv), (z, -Av), (z, Av)))   # R1 (seam)
        push!(patches, bq(( sv - Lv, -Lv), ( sv - Lv, Lv), (z, -Av), (z, Av)))   # L2 (seam)
    end

    # --- 3. Outer ring: 6 inflation + 6 shell (always the last 12 patches).
    #        Left/right are full edges; top/bottom split at x = 0 (b = 0)
    #        to match the butterfly seam. ------------------------------
    outer_specs = ((2, -o,  o),   # left  (−x), full
                   (1, -o,  o),   # right (+x), full
                   (3,  z,  o),   # top-left  (+y), x ∈ [−A, 0]
                   (3, -o,  z),   # top-right (+y), x ∈ [0, A]
                   (4, -o,  z),   # bottom-left  (−y), x ∈ [−A, 0]
                   (4,  z,  o))   # bottom-right (−y), x ∈ [0, A]
    for (dir, blo, bhi) in outer_specs
        push!(patches, PatchDesc(PatchInflation{2, T}((Mi, M), Int8(dir),
                                                      z, o, blo, bhi, z, z,
                                                      Av, Rmv)))
    end
    for (dir, blo, bhi) in outer_specs
        push!(patches, PatchDesc(PatchShell{2, T}((Ms, M), Int8(dir),
                                                  z, o, blo, bhi, z, z,
                                                  Rmv, R2v)))
    end
    @assert length(patches) == (touching ? 26 : 28)

    # --- Faces. Boundaries are explicit; interior interfaces come from the
    # frozen connectivity table for the mode — purely combinatorial, with no
    # geometry / precision / `R2 = Inf` concerns. Holes are the first 8
    # patches (face 1 = the circle); the 6 shells are the last 6 (face 2 =
    # the outer circle, or i⁰ when compactified). -------------------------
    faces = Matrix{FaceLink}(undef, 4, length(patches))
    for p in 1:8
        faces[1, p] = boundary_link(inner_tag)
    end
    for p in (length(patches) - 5):length(patches)
        faces[2, p] = boundary_link(outer_tag)
    end
    links = touching ? _TWO_HOLE_LINKS_TOUCHING : _TWO_HOLE_LINKS_SEPARATED
    @assert length(links) == (touching ? 45 : 49)
    for (p, f, q, g, oo) in links
        faces[f, p] = interior_link(q, g, Int8(oo))
        faces[g, q] = interior_link(p, f, Int8(oo))
    end

    return SkeletonMesh{2, T}(patches, faces)
end
