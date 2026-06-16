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
# Interior face connectivity and the D₁ orientation of each interface are
# discovered automatically by matching the physical endpoints of the
# faces (`_two_hole_autowire!`): the only hand-specified links are the
# domain boundaries (the two hole circles and the outer circle). The
# resulting mesh is still purely combinatorial — the floating-point match
# only selects each interface's discrete orientation flag at build time.

# Physical endpoints (tangent = 0 and tangent = Mt) of face `f` of a 2D
# patch, plus the tangential element count `Mt`. Face numbering: 1=−x,
# 2=+x, 3=−y, 4=+y.
@inline function _face_endpoints_2d(pd::PatchDesc{2, T}, f::Int) where {T}
    d = dims(pd)
    if f == 1            # −x: idx[1] = 0, tangent along axis 2
        i0 = (0, 0);      i1 = (0, d[2]);      mt = d[2]
    elseif f == 2        # +x: idx[1] = d[1]
        i0 = (d[1], 0);   i1 = (d[1], d[2]);   mt = d[2]
    elseif f == 3        # −y: idx[2] = 0, tangent along axis 1
        i0 = (0, 0);      i1 = (d[1], 0);      mt = d[1]
    else                 # +y: idx[2] = d[2]
        i0 = (0, d[2]);   i1 = (d[1], d[2]);   mt = d[1]
    end
    p0 = _patch_vertex_position(pd, i0)
    p1 = _patch_vertex_position(pd, i1)
    return (Float64(p0[1]), Float64(p0[2])), (Float64(p1[1]), Float64(p1[2])), mt
end

# Wire every interior (non-boundary) face by matching face endpoints.
# Each interior face has exactly one partner sharing both endpoints (with
# matching tangential count); the orientation flag is 0 if the endpoints
# correspond in order, 1 if reversed.
function _two_hole_autowire!(faces::Matrix{FaceLink},
                             patches::Vector{<:PatchDesc},
                             boundary::Vector{Tuple{Int, Int}},
                             tol::Float64)
    n = length(patches)
    isb = falses(4, n)
    for (p, f) in boundary
        isb[f, p] = true
    end
    # Cache endpoints / tangent counts of every non-boundary face.
    e0 = Matrix{NTuple{2, Float64}}(undef, 4, n)
    e1 = Matrix{NTuple{2, Float64}}(undef, 4, n)
    mt = zeros(Int, 4, n)
    for p in 1:n, f in 1:4
        isb[f, p] && continue
        a, b, m = _face_endpoints_2d(patches[p], f)
        e0[f, p] = a;  e1[f, p] = b;  mt[f, p] = m
    end
    close(a, b) = abs(a[1] - b[1]) ≤ tol && abs(a[2] - b[2]) ≤ tol
    wired = falses(4, n)
    for p in 1:n, f in 1:4
        (isb[f, p] || wired[f, p]) && continue
        a0 = e0[f, p];  a1 = e1[f, p];  m = mt[f, p]
        match_p = 0;  match_f = 0;  ori = 0;  nmatch = 0
        for q in 1:n, g in 1:4
            (q == p && g == f) && continue
            (isb[g, q] || wired[g, q]) && continue
            mt[g, q] == m || continue
            b0 = e0[g, q];  b1 = e1[g, q]
            if close(a0, b0) && close(a1, b1)
                match_p = q;  match_f = g;  ori = 0;  nmatch += 1
            elseif close(a0, b1) && close(a1, b0)
                match_p = q;  match_f = g;  ori = 1;  nmatch += 1
            end
        end
        nmatch == 1 ||
            error("_two_hole_autowire!: face (patch=$p, face=$f) has " *
                  "$(nmatch) matching neighbours (expected 1); endpoints " *
                  "$(a0)..$(a1).")
        faces[f, p]               = interior_link(match_p, match_f, ori)
        faces[match_f, match_p]   = interior_link(p, f, ori)
        wired[f, p]               = true
        wired[match_f, match_p]   = true
    end
    # Every face must now be either a boundary or a wired interior link.
    for p in 1:n, f in 1:4
        @assert isb[f, p] || wired[f, p] "_two_hole_autowire!: face " *
            "(patch=$p, face=$f) left unassigned."
    end
    return faces
end

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

Resolution: `M` is the number of cells along each hole-square edge (and so
the tangential resolution everywhere). Radial counts `M_h` (hole
inflation), `M_b` (butterfly), `M_i` (outer inflation), `M_s` (shell)
default from the hole-edge spacing `h = 2L/M`.

Boundary tagging (via `_shell_bc_tag`): the two hole circles use
`inner_bc` (default `:excision` → `bdry = 8`), the outer circle uses
`outer_bc` (default `:dirichlet` → `bdry = 1`).

The holes are assumed sufficiently far apart (`L ≪ d/2`) that the two
seam blocks straddling `x = 0` are well-shaped.
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
                            inner_bc::Symbol = :excision) where {T}
    skel = _two_hole_skeleton(T, R1, R2, d, M; L, A, R_mid,
                              M_h, M_b, M_i, M_s, outer_bc, inner_bc)
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
                            inner_bc::Symbol = :excision) where {T}
    outer_tag = _shell_bc_tag(outer_bc; outer = true)
    inner_tag = _shell_bc_tag(inner_bc; outer = false)
    @assert M ≥ 1
    @assert R1 > 0
    @assert R2 > 0

    # Geometry in Float64 (counts use `round`, which MultiFloats lack).
    R1f = Float64(R1);  R2f = Float64(R2)
    sf  = Float64(d) / 2
    Lf  = L     === nothing ? 1.5 * R1f          : Float64(L)
    Af  = A     === nothing ? 2.0 * (sf + Lf)    : Float64(A)
    Rmf = R_mid === nothing ? 1.5 * sqrt(2.0) * Af : Float64(R_mid)
    @assert R1f < Lf  "hole-square half-side L must exceed R1 (square encloses circle)"
    @assert Lf < sf   "holes overlap: need L < d/2"
    @assert Af > sf + Lf  "big square must contain the hole squares (A > d/2 + L)"
    @assert sqrt(2.0) * Af < Rmf  "R_mid must enclose the big-square corner (√2·A)"
    @assert Rmf < R2f  "R_mid must be inside the outer circle R2"

    hf = 2 * Lf / M
    Mh = M_h === nothing ? max(1, round(Int, (Lf - R1f) / hf)) : M_h
    Rb = M_b === nothing ? max(1, round(Int, (Af - (sf + Lf)) / hf)) : M_b
    Mi = M_i === nothing ? max(1, round(Int, (Rmf - (1 + sqrt(2.0)) / 2 * Af) / hf)) : M_i
    Ms = M_s === nothing ? max(1, round(Int, (R2f - Rmf) / hf)) : M_s
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
    # hole 1 (centre −s): left, top, right(→seam), bottom
    push!(patches, bq((-sv - Lv, -Lv), (-sv - Lv,  Lv), (-Av, -Av), (-Av,  Av)))  # L1
    push!(patches, bq((-sv - Lv,  Lv), (-sv + Lv,  Lv), (-Av,  Av), (  z,  Av)))  # T1
    push!(patches, bq((-sv + Lv, -Lv), (-sv + Lv,  Lv), (  z, -Av), (  z,  Av)))  # R1 (seam)
    push!(patches, bq((-sv - Lv, -Lv), (-sv + Lv, -Lv), (-Av, -Av), (  z, -Av)))  # B1
    # hole 2 (centre +s): right, top, left(→seam), bottom
    push!(patches, bq(( sv + Lv, -Lv), ( sv + Lv,  Lv), ( Av, -Av), ( Av,  Av)))  # R2
    push!(patches, bq(( sv - Lv,  Lv), ( sv + Lv,  Lv), (  z,  Av), ( Av,  Av)))  # T2
    push!(patches, bq(( sv - Lv, -Lv), ( sv - Lv,  Lv), (  z, -Av), (  z,  Av)))  # L2 (seam)
    push!(patches, bq(( sv - Lv, -Lv), ( sv + Lv, -Lv), (  z, -Av), ( Av, -Av)))  # B2

    # --- 3. Outer inflation (patches 17–22) + shell (patches 23–28).
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
    @assert length(patches) == 28

    # --- Faces: boundaries explicit, interior auto-wired. ---------------
    faces = Matrix{FaceLink}(undef, 4, length(patches))
    boundary = Tuple{Int, Int}[]
    for p in 1:8                       # hole circles (face 1 = a_lo = circle)
        faces[1, p] = boundary_link(inner_tag)
        push!(boundary, (p, 1))
    end
    for p in 23:28                     # outer circle (face 2 = a_hi = R2)
        faces[2, p] = boundary_link(outer_tag)
        push!(boundary, (p, 2))
    end
    autotol = 1e-9 * max(1.0, R2f)
    _two_hole_autowire!(faces, patches, boundary, autotol)

    return SkeletonMesh{2, T}(patches, faces)
end
