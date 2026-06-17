# ----------------------------------------------------------------------
# 9-patch inflated square mesh — 2D analog of `inflated_cube.jl`.
# Builder + topology only.
#
# Topology: one axis-aligned inner square `[-L, L]²` + four inflation
# patches (square edge → inner circle `r = R1`) + four annular-shell
# patches (`R1 → R2`).
#
# Mirrors the 3D inflated cube structurally; the orientation group on
# each inter-patch edge is D₁ (identity / reversal) rather than D₄.
#
# The analytic Jacobian for curvilinear patches lives in
# `patch_jacobian.jl`; the analytic patch ↔ global maps and point
# locators live in `queries_2d.jl`.

# Tangential-edge connectivity for the inflation / shell patches of an
# inflated-square mesh. Same indexing scheme as `_WEDGE_NEIGHBOUR_2D`
# but DIFFERENT entries (right-handed `_patch_direction_vec_2d` with
# sign-flips on `−x, +y`).
const _INFLATION_NEIGHBOUR_2D = (
    ((4, 4), (3, 3)),   # +x: faces 3,4 → -y face 4, +y face 3
    ((3, 4), (4, 3)),   # -x:           → +y face 4, -y face 3
    ((1, 4), (2, 3)),   # +y:           → +x face 4, -x face 3
    ((2, 4), (1, 3)),   # -y:           → -x face 4, +x face 3
)

"""
    make_inflated_square_mesh(::Type{T}, L, R1, R2, M; M_i, M_s, outer_bc) → Mesh{2, T}

Build a 9-patch inflated square mesh of the disk `|x| ≤ R2`:

* an axis-aligned inner square `[-L, L]²` with `M × M` elements,
* four inflation patches interpolating each square edge to the inner
  circle `r = R1`, with `M_i × M` elements each,
* four annular-shell patches interpolating the inner circle to the
  outer circle `r = R2`, with `M_s × M` elements each.

`M_i` defaults to `round((R1 - (1 + √2)/2 · L) / h)` where
`h = 2L/M`. `M_s` defaults to `round((R2 - R1) / h)`. Both clip to
at least 1.

Passing `R2 = Inf` gives a **compactified** outer boundary: the shell maps
its outer face to spatial infinity i⁰ (`r(a) = R1/(1−a)`); `M_s` then
defaults to `M` radial cells (the infinite domain has no `h`-derived scale).

The outer boundary `r = R2` is tagged on every shell-patch outer face
according to `outer_bc`:

* `:dirichlet` (default) — `bdry = 1`.
* `:sommerfeld` — `bdry = 7` (first-order radiative absorbing BC).

The returned `Mesh{2, T}` has all four patch fields populated:
`patch_desc` carries 9 `PatchDesc{2, T}` entries (one `Cubic` + four
`Inflation` + four `Shell`); the per-element `patch_id` / `patch_idx`
tables drive the analytic-Jacobian path in downstream `make_geometry`.
"""
function make_inflated_square_mesh(::Type{T}, L::Real, R1::Real, R2::Real,
                                    M::Int;
                                    M_i::Union{Nothing, Int} = nothing,
                                    M_s::Union{Nothing, Int} = nothing,
                                    outer_bc::Symbol = :dirichlet) where {T}
    skel = _inflated_square_skeleton(T, L, R1, R2, M; M_i, M_s, outer_bc)
    return _skeleton_to_mesh(skel)
end

function _inflated_square_skeleton(::Type{T}, L::Real, R1::Real, R2::Real,
                                    M::Int;
                                    M_i::Union{Nothing, Int} = nothing,
                                    M_s::Union{Nothing, Int} = nothing,
                                    outer_bc::Symbol = :dirichlet) where {T}
    outer_bc_tag = outer_bc === :dirichlet  ? Int8(1) :
                   outer_bc === :sommerfeld ? Int8(7) :
                   error("_inflated_square_skeleton: outer_bc must be " *
                         ":dirichlet or :sommerfeld, got $(repr(outer_bc))")
    @assert M ≥ 1
    @assert L > 0
    @assert L * sqrt(2) < R1 "inner circle R1 must enclose the square corner (L·√2)"
    @assert R1 < R2

    Lv  = T(L)
    R1v = T(R1)
    R2v = T(R2)
    h   = 2L / M

    # Patch counts in Float64 (geometry-independent of `T`; MultiFloats
    # have no `round(Int, ·)`).
    Lf = Float64(L); R1f = Float64(R1); R2f = Float64(R2); hf = 2Lf / M
    Mi = M_i === nothing ?
         max(1, round(Int, (R1f - (1 + sqrt(2.0))/2 * Lf) / hf)) :
         M_i
    Ms = M_s !== nothing ? M_s :
         isinf(R2f) ? M :                          # compactified: no finite extent
         max(1, round(Int, (R2f - R1f) / hf))
    @assert Mi ≥ 1
    @assert Ms ≥ 1

    z = zero(T)
    o = one(T)

    # Patch 1: inner square [-L, L]².
    inner = PatchDesc(PatchCubic{2, T}((M, M), (-Lv, -Lv), (Lv, Lv)))

    # Patches 2..5: inflation in directions (+x, -x, +y, -y).
    # Patches 6..9: shell in same direction order.
    patches = PatchDesc{2, T}[inner]
    for dir in 1:4
        push!(patches, PatchDesc(PatchInflation{2, T}((Mi, M), Int8(dir),
                                                        z, o, -o, o, z, z,
                                                        Lv, R1v)))
    end
    for dir in 1:4
        push!(patches, PatchDesc(PatchShell{2, T}((Ms, M), Int8(dir),
                                                    z, o, -o, o, z, z,
                                                    R1v, R2v)))
    end
    @assert length(patches) == 9

    faces = Matrix{FaceLink}(undef, 4, length(patches))

    # ---- Square ↔ inflation interfaces. ----
    CUBE_FACE_TO_DIR        = (2, 1, 4, 3)
    CUBE_FACE_ORIENTATION   = (Int8(1), Int8(0), Int8(0), Int8(1))
    for f in 1:4
        d  = CUBE_FACE_TO_DIR[f]
        ip = d + 1                       # inflation patch id
        oo = CUBE_FACE_ORIENTATION[f]
        faces[f, 1]  = interior_link(ip, 1, oo)
        faces[1, ip] = interior_link(1,  f, oo)
    end

    # ---- Inflation tangential / outer-radial faces. ----
    for d in 1:4
        ip = d + 1                       # inflation patch
        sp = d + 5                       # shell patch (same direction)
        faces[2, ip] = interior_link(sp, 1, 0)
        for f in 3:4
            neigh_dir, neigh_face = _INFLATION_NEIGHBOUR_2D[d][f - 2]
            faces[f, ip] = interior_link(neigh_dir + 1, neigh_face, 0)
        end
    end

    # ---- Shell tangential / outer-circle faces. ----
    for d in 1:4
        sp = d + 5
        ip = d + 1
        faces[1, sp] = interior_link(ip, 2, 0)
        faces[2, sp] = boundary_link(outer_bc_tag)
        for f in 3:4
            neigh_dir, neigh_face = _INFLATION_NEIGHBOUR_2D[d][f - 2]
            faces[f, sp] = interior_link(neigh_dir + 5, neigh_face, 0)
        end
    end

    return SkeletonMesh{2, T}(patches, faces)
end
