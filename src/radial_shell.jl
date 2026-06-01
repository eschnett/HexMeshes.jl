# ----------------------------------------------------------------------
# Pure 6-patch spherical-shell mesh — builder + topology only.
#
# Topology: six `PatchShell` patches covering the spherical shell
# `R1 ≤ |x| ≤ R2`, no inner cube and no inflation layer. Intended for
# single-black-hole evolutions with excision: the inner sphere at `R1`
# is the excision surface (the BH is *inside*, removed from the
# computational domain), and the outer sphere at `R2` is the
# computational outer boundary.
#
# `PatchShell` uses `r(a) = (1 − a)·R1 + a·R2` so the radial element
# spacing is exactly constant `(R2 − R1) / M_r`. Tangential connectivity
# between the six patches is identical to the shell-layer topology of
# `make_inflated_cube_mesh` — they reuse `_INFLATION_NEIGHBOUR`.

# Map a BC `Symbol` to the corresponding `bdry` tag value:
#   :dirichlet (outer face) → 1
#   :dirichlet (inner face) → 2  — distinct from outer so kernels can
#       later distinguish them; both fire the existing `1 ≤ tag ≤ 6`
#       Dirichlet branch unchanged.
#   :sommerfeld → 7
#   :excision (alias :outflow) → 8
@inline function _shell_bc_tag(bc::Symbol; outer::Bool)
    if bc === :dirichlet
        return outer ? Int8(1) : Int8(2)
    elseif bc === :sommerfeld
        return Int8(7)
    elseif bc === :excision || bc === :outflow
        return Int8(8)
    else
        error("make_radial_shell_mesh: bc must be :dirichlet, " *
              ":sommerfeld, :excision (or :outflow), got $(repr(bc))")
    end
end

"""
    make_radial_shell_mesh(::Type{T}, R1, R2, M;
                            M_r = nothing,
                            outer_bc = :dirichlet,
                            inner_bc = :excision) → Mesh{3, T}

Build a 6-patch pure spherical-shell mesh covering `R1 ≤ |x| ≤ R2`.

* Six `PatchShell` patches, one per direction `(+x, -x, +y, -y, +z, -z)`,
  each with `M_r × M × M` elements. `M_r` (radial elements) defaults to
  `max(1, round(Int, M · (R2 - R1) / (R2 + R1) · 2))` ≈
  `max(1, round((R2 - R1) / h))` with `h = (R2 - R1) / M` — same
  heuristic as `make_inflated_cube_mesh`'s shell layer, referenced to
  shell thickness instead of cube side.
* Radial spacing is exactly constant: each radial element covers
  `Δr = (R2 - R1) / M_r`. This is the defining feature of the mesh —
  designed for BH excision where the inner sphere `R1` is the excision
  surface and uniform radial resolution is important.

Boundary-condition tagging:

| kwarg value     | inner face (`bdry`) | outer face (`bdry`) |
| --------------- | ------------------- | ------------------- |
| `:dirichlet`    | `2`                 | `1`                 |
| `:sommerfeld`   | `7`                 | `7`                 |
| `:excision`     | `8`                 | `8`                 |
| `:outflow`      | `8`                 | `8`                 |  (alias)

Tag `8` is the new "excision / pure outflow" tag: kernels that
recognise it apply **no SAT correction** at the face (natural SBP
one-sided stencil at the boundary node is the BC). Tags `1..6` all
fire the standard Dirichlet branch; the distinct values for inner
(`2`) vs outer (`1`) Dirichlet are reserved for future inhomogeneous-
data dispatch.

Defaults `outer_bc = :dirichlet`, `inner_bc = :excision` — the
natural setup for BH evolutions.
"""
function make_radial_shell_mesh(::Type{T}, R1::Real, R2::Real, M::Int;
                                  M_r::Union{Nothing, Int} = nothing,
                                  outer_bc::Symbol = :dirichlet,
                                  inner_bc::Symbol = :excision) where {T}
    skel = _radial_shell_skeleton(T, R1, R2, M; M_r, outer_bc, inner_bc)
    return _skeleton_to_mesh(skel)
end

function _radial_shell_skeleton(::Type{T}, R1::Real, R2::Real, M::Int;
                                  M_r::Union{Nothing, Int} = nothing,
                                  outer_bc::Symbol = :dirichlet,
                                  inner_bc::Symbol = :excision) where {T}
    @assert M ≥ 1
    @assert R1 > 0
    @assert R1 < R2

    outer_tag = _shell_bc_tag(outer_bc; outer = true)
    inner_tag = _shell_bc_tag(inner_bc; outer = false)

    R1v = T(R1)
    R2v = T(R2)
    h   = (R2 - R1) / M
    Mr  = M_r === nothing ?
          max(1, round(Int, (R2 - R1) / h)) :
          M_r
    @assert Mr ≥ 1

    z = zero(T)
    o = one(T)

    # Six shell patches in the canonical direction order
    # (+x, -x, +y, -y, +z, -z) — same order as the shell layer in
    # `make_inflated_cube_mesh`.
    patches = PatchDesc{3, T}[
        PatchDesc(PatchShell{3, T}((Mr, M, M), Int8(dir),
                                     z, o, -o, o, -o, o,
                                     R1v, R2v))
        for dir in 1:6
    ]

    faces = Matrix{FaceLink}(undef, 6, length(patches))

    # Shell tangential connectivity: each shell's face 1 (a = a_lo) is
    # the inner-radial boundary (= inner sphere); face 2 (a = a_hi) is
    # the outer-radial boundary (= outer sphere); faces 3..6 are
    # tangential, wired by the cubed-sphere D₄ table.
    for d in 1:6
        sp = d                                            # this patch id
        faces[1, sp] = boundary_link(inner_tag)
        faces[2, sp] = boundary_link(outer_tag)
        for f in 3:6
            neigh_dir, neigh_face = _INFLATION_NEIGHBOUR[d][f - 2]
            faces[f, sp] = interior_link(neigh_dir, neigh_face, 0)
        end
    end

    return SkeletonMesh{3, T}(patches, faces)
end
