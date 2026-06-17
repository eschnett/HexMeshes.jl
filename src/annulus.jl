# ----------------------------------------------------------------------
# Pure 4-patch annulus mesh — the 2D analog of `make_radial_shell_mesh`.
# Builder + topology only.
#
# Topology: four `PatchShell{2}` patches covering the annulus
# `R1 ≤ |x| ≤ R2`, with no inner square and no inflation layer (unlike
# `make_inflated_square_mesh`, which fills the disk). Intended for 2D
# black-hole-excision-style evolutions: the inner circle at `R1` is the
# excision surface (the hole is *inside*, removed from the computational
# domain) and the outer circle at `R2` is the computational outer
# boundary.
#
# `PatchShell` uses `r(a) = (1 − a)·R1 + a·R2` so the radial element
# spacing is exactly constant `(R2 − R1) / M_r`. Tangential connectivity
# between the four patches reuses `_INFLATION_NEIGHBOUR_2D` (the
# shell-layer topology of `make_inflated_square_mesh`), and the
# inner/outer boundary tagging reuses `_shell_bc_tag` (from
# `radial_shell.jl`).

"""
    make_annulus_mesh(::Type{T}, R1, R2, M;
                       M_r = nothing,
                       outer_bc = :sommerfeld,
                       inner_bc = :excision) → Mesh{2, T}

Build a 4-patch pure annulus mesh covering `R1 ≤ |x| ≤ R2` — the 2D
analog of [`make_radial_shell_mesh`](@ref).

* Four `PatchShell{2}` patches, one per direction `(+x, -x, +y, -y)`,
  each with `M_r × M` elements. `M_r` (radial elements) defaults to
  `max(1, round(Int, (R2 - R1) / h))` with `h = (R2 - R1) / M` (= `M`),
  giving exactly constant radial spacing `Δr = (R2 - R1) / M_r`.
* No inner square: the inner circle `R1` is a boundary (the excision
  surface), not an interior interface.

Passing `R2 = Inf` gives a **compactified** outer boundary (the shell maps
its outer face to spatial infinity i⁰); `M_r` then defaults to `M`.

Boundary-condition tagging (same convention as `make_radial_shell_mesh`):

| kwarg value     | inner face (`bdry`) | outer face (`bdry`) |
| --------------- | ------------------- | ------------------- |
| `:dirichlet`    | `2`                 | `1`                 |
| `:sommerfeld`   | `7`                 | `7`                 |
| `:excision`     | `8`                 | `8`                 |  (alias `:outflow`)

Tag `8` is the "excision / pure outflow" tag: kernels that recognise it
apply **no SAT correction** at the face. Defaults
`outer_bc = :sommerfeld`, `inner_bc = :excision` — the natural setup for
a 2D excision evolution with an absorbing outer boundary.
"""
function make_annulus_mesh(::Type{T}, R1::Real, R2::Real, M::Int;
                            M_r::Union{Nothing, Int} = nothing,
                            outer_bc::Symbol = :sommerfeld,
                            inner_bc::Symbol = :excision) where {T}
    skel = _annulus_skeleton(T, R1, R2, M; M_r, outer_bc, inner_bc)
    return _skeleton_to_mesh(skel)
end

function _annulus_skeleton(::Type{T}, R1::Real, R2::Real, M::Int;
                            M_r::Union{Nothing, Int} = nothing,
                            outer_bc::Symbol = :sommerfeld,
                            inner_bc::Symbol = :excision) where {T}
    @assert M ≥ 1
    @assert R1 > 0
    @assert R1 < R2

    outer_tag = _shell_bc_tag(outer_bc; outer = true)
    inner_tag = _shell_bc_tag(inner_bc; outer = false)

    R1v = T(R1)
    R2v = T(R2)
    h   = (R2 - R1) / M
    Mr  = M_r !== nothing ? M_r :
          isinf(R2) ? M :                          # compactified: no finite extent
          max(1, round(Int, (R2 - R1) / h))
    @assert Mr ≥ 1

    z = zero(T)
    o = one(T)

    # Four shell patches in the canonical direction order (+x, -x, +y, -y)
    # — the same order and `(a, b) ∈ [0,1] × [-1,1]` parametrisation as
    # the annular-shell layer of `make_inflated_square_mesh`.
    patches = PatchDesc{2, T}[
        PatchDesc(PatchShell{2, T}((Mr, M), Int8(dir),
                                     z, o, -o, o, z, z,
                                     R1v, R2v))
        for dir in 1:4
    ]

    faces = Matrix{FaceLink}(undef, 4, length(patches))

    # Shell tangential connectivity: each shell's face 1 (a = a_lo) is
    # the inner-radial boundary (= inner circle); face 2 (a = a_hi) is
    # the outer-radial boundary (= outer circle); faces 3,4 are
    # tangential, wired by `_INFLATION_NEIGHBOUR_2D`. Patch ids equal the
    # direction (no inner-cube offset), so the neighbour patch id is just
    # `neigh_dir`.
    for d in 1:4
        sp = d
        faces[1, sp] = boundary_link(inner_tag)
        faces[2, sp] = boundary_link(outer_tag)
        for f in 3:4
            neigh_dir, neigh_face = _INFLATION_NEIGHBOUR_2D[d][f - 2]
            faces[f, sp] = interior_link(neigh_dir, neigh_face, 0)
        end
    end

    return SkeletonMesh{2, T}(patches, faces)
end
