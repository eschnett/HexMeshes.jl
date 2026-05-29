
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
    PatchInfo{T}

Per-element parametric description used by `make_geometry` for elements
of an `InflatedCubeMesh` that should be discretised with an analytic
curvilinear map instead of the default trilinear interpolation of the
eight corners.

`kind` selects the patch family:

* `0`     — trilinear (inner cube; the angular / radial fields are unused).
* `1..6`  — inflation patch in directions `(+x, -x, +y, -y, +z, -z)`:
  bridges the inner cube face at `r = L` (with `r = |x|/|y|/|z|`) to the
  inner sphere at radius `R₁`.
* `7..12` — spherical-shell patch in the same directional order:
  bridges the inner sphere at `R₁` to the outer sphere at `R₂`.

For curved patches, `(a_lo, a_hi)` is the radial parameter range
(`s ∈ [0, 1]` for inflation, `ρ ∈ [0, 1]` for shell), `(b_lo, b_hi)` and
`(c_lo, c_hi)` are the two tangent-angular ranges (each `⊂ [-1, 1]`).
"""
struct PatchInfo{T}
    kind :: PatchKind3D
    a_lo :: T
    a_hi :: T
    b_lo :: T
    b_hi :: T
    c_lo :: T
    c_hi :: T
end

"""
    InflatedCubeMesh{T}

A 13-patch conforming hex mesh:

* one axis-aligned inner cube `[-L, L]³` (`M³` elements),
* six inflation patches that bridge each cube face to the inner sphere
  at radius `R₁` (`M_i × M × M` elements each),
* six spherical-shell patches that bridge the inner sphere `R₁` to the
  outer sphere `R₂` (`M_s × M × M` elements each).

Total element count: `M³ + 6 · M² · (M_i + M_s)`.

# Fields

* `base :: HexMesh{T}` — connectivity + trilinear vertex info. Host-side
  queries (`element_vertices`, `nv`, `locate_point`, plotting) work
  through this just like a plain `HexMesh`.
* `patch_info :: Vector{PatchInfo{T}}` of length `base.Ne` — per-element
  parametric description used by `make_geometry` to evaluate the
  analytic Jacobian on the curved patches.
* `L, R1, R2 :: T` — inner-cube half-edge, inner-sphere radius, outer-
  sphere radius. Required: `0 < L · √3 < R1 < R2`.

The radial spacing is **exactly** constant on the spherical shells
(uniform `(R2 - R1)/M_s` along every radial ray); on the inflation
patches the radial spacing is constant on average (varies by a factor
between cube-face center and cube-face corner).
"""
struct InflatedCubeMesh{T, MI, MI8}
    base       :: HexMesh{T, MI, MI8}
    patch_info :: Vector{PatchInfo{T}}
    L          :: T
    R1         :: T
    R2         :: T
    # Element-count metadata, needed by the analytic
    # `locate_point` / `locate_patch` path:
    #   * `M`  — angular resolution on every patch
    #   * `Mi` — radial cells per inflation patch
    #   * `Ms` — radial cells per shell patch
    M          :: Int
    Mi         :: Int
    Ms         :: Int
end

# Forward `HexMesh` accessors through the wrapper so existing host code
# (`mesh.Ne`, `mesh.bdry`, `mesh.vertex_coords`, …) keeps working unchanged.
@inline function Base.getproperty(m::InflatedCubeMesh, name::Symbol)
    if name === :base || name === :patch_info ||
       name === :L || name === :R1 || name === :R2 ||
       name === :M || name === :Mi || name === :Ms
        return getfield(m, name)
    else
        return getproperty(getfield(m, :base), name)
    end
end
Base.propertynames(m::InflatedCubeMesh) =
    (:base, :patch_info, :L, :R1, :R2, :M, :Mi, :Ms,
     :Ne, :conn, :vertex_coords, :vertex_idx,
     :neighbour, :neighbour_face, :orientation, :bdry)

nv(mesh::InflatedCubeMesh) = nv(mesh.base)

# Direction-dependent unit vector `v(b, c)` for the six face directions.
# For each direction `dir ∈ 1..6` (mapping `(+x, -x, +y, -y, +z, -z)`),
# the parameterisation places the physical point `P = f(a, b, c) · v(b, c)`
# (inflation) or `P = (r(a) / Q) · v(b, c)` (shell), where the local
# `(ξ, η, ζ)` frame is right-handed in physical space — picking the
# tangent-axis pair per direction so that `det J > 0`. The corresponding
# tables:
#
#   +x:  v = ( 1,  b,  c)            -x:  v = (-1,  c,  b)
#   +y:  v = ( c,  1,  b)            -y:  v = ( b, -1,  c)
#   +z:  v = ( b,  c,  1)            -z:  v = ( c,  b, -1)
@inline function _patch_direction_vec(dir::Int8, b::T, c::T) where {T}
    if dir == Int8(1)
        return (one(T), b, c)
    elseif dir == Int8(2)
        return (-one(T), c, b)
    elseif dir == Int8(3)
        return (c, one(T), b)
    elseif dir == Int8(4)
        return (b, -one(T), c)
    elseif dir == Int8(5)
        return (b, c, one(T))
    else
        return (c, b, -one(T))
    end
end

# Same as `_patch_direction_vec`, plus the constant partials `∂v/∂b` and
# `∂v/∂c` (each a 3-tuple; entries are `0` or `±1`). Used by the
# analytic-Jacobian path in `make_geometry`.
@inline function _patch_direction_vec_and_derivs(dir::Int8, b::T, c::T) where {T}
    z = zero(T); o = one(T)
    if dir == Int8(1)        # +x
        return (o, b, c,    z, o, z,    z, z, o)
    elseif dir == Int8(2)    # -x
        return (-o, c, b,   z, z, o,    z, o, z)
    elseif dir == Int8(3)    # +y
        return (c, o, b,    z, z, o,    o, z, z)
    elseif dir == Int8(4)    # -y
        return (b, -o, c,   o, z, z,    z, z, o)
    elseif dir == Int8(5)    # +z
        return (b, c, o,    o, z, z,    z, o, z)
    else                     # -z
        return (c, b, -o,   z, o, z,    o, z, z)
    end
end

"""
    make_inflated_cube_mesh(::Type{T}, L, R1, R2, M; M_i, M_s, outer_bc) → InflatedCubeMesh{T}

Build a 13-patch inflated cube mesh of the ball `|x| ≤ R2`:

* an axis-aligned inner cube `[-L, L]³` with `M × M × M` elements,
* six inflation patches that interpolate from each cube face to the
  inner sphere `r = R1`, with `M_i × M × M` elements each,
* six spherical-shell patches that interpolate from the inner sphere to
  the outer sphere `r = R2`, with `M_s × M × M` elements each.

`M_i` defaults to `round((R1 - (1 + √3)/2 · L) / h)` (average radial gap
between the cube and the inner sphere, expressed in cube-edge cells
`h = 2L/M`). `M_s` defaults to `round((R2 - R1) / h)`. Each defaults to
at least 1.

The shell patches use the parameterisation `r(ρ) = (1 - ρ)·R1 + ρ·R2`
along every radial ray, so they have exactly constant radial spacing
`(R2 - R1) / M_s` and exactly uniform angular sampling in
`(η, ζ) ∈ [-1, 1]²`. The inflation patches use
`r(s, η, ζ) = (1 - s)·L + s · R1 / √(1 + η² + ζ²)`, so their radial
spacing is constant on average (varies between cube-face center and
cube-face corner). The geometry matches conformally at every inter-
patch interface (cube → inflation at `r = L`; inflation → shell at
`r = R1`; adjacent inflation/shell patches along the shared cube
edges and great circles on the inner / outer spheres).

The outer boundary `r = R2` is tagged on every shell-patch outer face
according to `outer_bc`:

* `:dirichlet` (default) — `bdry = 1`, value taken from
  `bdry_values[1]` in the kernel (homogeneous if you pass zero there).
* `:sommerfeld` — `bdry = 7`, treated as a first-order radiative /
  absorbing condition `u̇ + ∂_n u = 0` by `rhs3d!`. Energy bleeds out
  of the boundary at rate `∮ u̇² dS`; the symplectic integrator is
  still stable (`L_h` becomes dissipative rather than skew-symmetric)
  and the modified Hamiltonian drifts downward with the physical loss.

`make_geometry(mesh, elem)` dispatches per element on `patch_info[e].kind`:
trilinear for the inner cube; analytic Jacobian on the curved patches.
"""
function _inflated_cube_skeleton(::Type{T}, L::Real, R1::Real, R2::Real, M::Int;
                                   M_i::Union{Nothing, Int}=nothing,
                                   M_s::Union{Nothing, Int}=nothing,
                                   outer_bc::Symbol = :dirichlet) where {T}
    outer_bc_tag = outer_bc === :dirichlet ? Int8(1) :
                   outer_bc === :sommerfeld ? Int8(7) :
                   error("_inflated_cube_skeleton: outer_bc must be " *
                         ":dirichlet or :sommerfeld, got $(repr(outer_bc))")
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
    Ms = M_s === nothing ?
         max(1, round(Int, (R2 - R1) / h)) :
         M_s
    @assert Mi ≥ 1
    @assert Ms ≥ 1

    z = zero(T)
    o = one(T)

    # Patch 1: inner cube `[-L, L]³`.
    inner = PatchSpec{T}(M, M, M, Cubical_3D,
                          -Lv, Lv, -Lv, Lv, -Lv, Lv,
                          z, z, z)

    # Patches 2..7: inflation in directions (+x, -x, +y, -y, +z, -z),
    # parameter range `(a, b, c) ∈ [0, 1] × [-1, 1]²`.
    inflation_families = (InflationPosX_3D, InflationNegX_3D,
                          InflationPosY_3D, InflationNegY_3D,
                          InflationPosZ_3D, InflationNegZ_3D)
    # Patches 8..13: shells in the same direction order.
    shell_families     = (ShellPosX_3D, ShellNegX_3D,
                          ShellPosY_3D, ShellNegY_3D,
                          ShellPosZ_3D, ShellNegZ_3D)

    patches = PatchSpec{T}[inner]
    for fam in inflation_families
        push!(patches, PatchSpec{T}(Mi, M, M, fam,
                                     z, o, -o, o, -o, o,
                                     Lv, R1v, R2v))
    end
    for fam in shell_families
        push!(patches, PatchSpec{T}(Ms, M, M, fam,
                                     z, o, -o, o, -o, o,
                                     Lv, R1v, R2v))
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
    # Both sides of an `o = 5` link carry `o = 5` (transpose is its
    # own inverse).
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
    # Tangential faces 3..6 connect to adjacent inflation patches via
    # `_INFLATION_NEIGHBOUR` (the cube-edge topology *for the right-handed
    # `_patch_direction_vec` convention*; not the same as the cubed-cube
    # `_WEDGE_NEIGHBOUR`). All twelve cube-edge orientations are still 0.
    # Face 2 (outer-radial, `a = 1`) connects to the same-direction shell
    # patch's face 1 with orientation 0.
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
    # Face 1: → inflation patch face 2 (inner-radial side).
    # Face 2: outer sphere domain boundary, tag 1.
    # Faces 3..6: adjacent shell patches via `_INFLATION_NEIGHBOUR` (same
    # direction conventions, same connectivity).
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

    return SkeletonMesh{T}(patches, faces)
end

# Build the per-element `PatchInfo` table that
# `make_geometry(::InflatedCubeMesh)` reads to dispatch on patch kind
# and recover the parameter-space extent of each element. Order matches
# `_skeleton_to_mesh`'s element enumeration (column-major over `(a, b, c)`
# inside each patch, patches walked in order).
function _build_inflated_cube_patch_info(skel::SkeletonMesh{T}, Ne::Int) where {T}
    patch_info = Vector{PatchInfo{T}}(undef, Ne)
    e = 0
    zT = zero(T)
    for ps in skel.patches
        kind = ps.family
        for c in 1:ps.Mc, b in 1:ps.Mb, a in 1:ps.Ma
            e += 1
            if is_cubical(kind)
                # Inner cube: trilinear path; parameter-extent slots unused.
                patch_info[e] = PatchInfo{T}(Cubical_3D, zT, zT, zT, zT, zT, zT)
            else
                a_lo = ps.a_lo + (ps.a_hi - ps.a_lo) * T(a - 1) / T(ps.Ma)
                a_hi = ps.a_lo + (ps.a_hi - ps.a_lo) * T(a)     / T(ps.Ma)
                b_lo = ps.b_lo + (ps.b_hi - ps.b_lo) * T(b - 1) / T(ps.Mb)
                b_hi = ps.b_lo + (ps.b_hi - ps.b_lo) * T(b)     / T(ps.Mb)
                c_lo = ps.c_lo + (ps.c_hi - ps.c_lo) * T(c - 1) / T(ps.Mc)
                c_hi = ps.c_lo + (ps.c_hi - ps.c_lo) * T(c)     / T(ps.Mc)
                patch_info[e] = PatchInfo{T}(kind, a_lo, a_hi, b_lo, b_hi, c_lo, c_hi)
            end
        end
    end
    @assert e == Ne
    return patch_info
end

function make_inflated_cube_mesh(::Type{T}, L::Real, R1::Real, R2::Real, M::Int;
                                  M_i::Union{Nothing, Int}=nothing,
                                  M_s::Union{Nothing, Int}=nothing,
                                  outer_bc::Symbol = :dirichlet) where {T}
    skel       = _inflated_cube_skeleton(T, L, R1, R2, M; M_i, M_s, outer_bc)
    base       = _skeleton_to_mesh(skel)
    patch_info = _build_inflated_cube_patch_info(skel, base.Ne)
    # Recover the (possibly auto-defaulted) radial cell counts directly
    # from the skeleton — `skel.patches[2]` is the first inflation patch
    # (`Ma = M_i`), `skel.patches[8]` is the first shell patch (`Ma = M_s`).
    Mi = skel.patches[2].Ma
    Ms = skel.patches[8].Ma
    return InflatedCubeMesh(base, patch_info, T(L), T(R1), T(R2), M, Mi, Ms)
end


"""
    _patch_point_and_jac(pi, ξ, η_ref, ζ_ref, L, R1, R2) → (P, J)

Analytic element map for one node of an `InflatedCubeMesh` curved patch.
Given a per-element `PatchInfo` and a reference-cube coordinate
`(ξ, η_ref, ζ_ref) ∈ [0, 1]³`, returns the physical point `P` and the
`SMatrix{3, 3}` Jacobian `J[i, a] = ∂P_i / ∂ξₐ_ref`. The reference-to-
parameter affine map and the parameter-to-physical map are composed
analytically; no finite differencing is involved.

Used by `make_geometry(::InflatedCubeMesh)`. For `pi.kind == Cubical_3D` (inner
cube), use the trilinear path instead — this routine assumes a curved
patch.
"""
@inline function _patch_point_and_jac(pi::PatchInfo{T},
                                       ξ::T, η_ref::T, ζ_ref::T,
                                       L::T, R1::T, R2::T) where {T}
    # Reference-cube [0, 1]³ → parameter-space (a, b, c).
    da = pi.a_hi - pi.a_lo
    db = pi.b_hi - pi.b_lo
    dc = pi.c_hi - pi.c_lo
    a  = pi.a_lo + da * ξ
    b  = pi.b_lo + db * η_ref
    c  = pi.c_lo + dc * ζ_ref

    Q  = sqrt(one(T) + (b*b + c*c))
    Q3 = Q * Q * Q
    pi_is_shell = is_shell(pi.kind)
    dir         = direction_of(pi.kind)

    # Scalar `f(a, b, c)` such that `P = f · v(b, c)`.
    if pi_is_shell
        rval  = (one(T) - a) * R1 + a * R2
        f     = rval / Q
        df_da = (R2 - R1) / Q
        df_db = -rval * b / Q3
        df_dc = -rval * c / Q3
    else
        f     = (one(T) - a) * L + a * R1 / Q
        df_da = -L + R1 / Q
        df_db = -a * R1 * b / Q3
        df_dc = -a * R1 * c / Q3
    end

    vx, vy, vz, dvxb, dvyb, dvzb, dvxc, dvyc, dvzc =
        _patch_direction_vec_and_derivs(dir, b, c)

    Px = f * vx; Py = f * vy; Pz = f * vz

    dPa_x = df_da * vx;            dPa_y = df_da * vy;            dPa_z = df_da * vz
    dPb_x = df_db * vx + f * dvxb; dPb_y = df_db * vy + f * dvyb; dPb_z = df_db * vz + f * dvzb
    dPc_x = df_dc * vx + f * dvxc; dPc_y = df_dc * vy + f * dvyc; dPc_z = df_dc * vz + f * dvzc

    # Reference-cube Jacobian: column `a` is `∂P/∂ξ_ref_a`, scaled by the
    # affine ref→parameter derivative.
    J = SMatrix{3, 3, T}(
        dPa_x * da, dPa_y * da, dPa_z * da,
        dPb_x * db, dPb_y * db, dPb_z * db,
        dPc_x * dc, dPc_y * dc, dPc_z * dc)
    P = SVector{3, T}(Px, Py, Pz)
    return P, J
end

# ----------------------------------------------------------------------
# Analytic patch ↔ global maps for `InflatedCubeMesh` (3D analog of
# the 2D versions in `inflated_square.jl`).
#
# Per-direction inverse of `_patch_direction_vec(dir, b, c)`. The b/c
# swaps on `-x, +y, -z` (right-handed local frames) carry through to
# the inverse:
#
#   +x: v = ( 1, b, c)   →  f = x,   b = y/x,   c = z/x
#   -x: v = (-1, c, b)   →  f = -x,  c = -y/x,  b = -z/x
#   +y: v = ( c, 1, b)   →  f = y,   c = x/y,   b = z/y
#   -y: v = ( b,-1, c)   →  f = -y,  b = -x/y,  c = -z/y
#   +z: v = ( b, c, 1)   →  f = z,   b = x/z,   c = y/z
#   -z: v = ( c, b,-1)   →  f = -z,  c = -x/z,  b = -y/z
@inline function _inverse_dir_vec_3d(dir::Integer,
                                      x::T, y::T, z::T) where {T}
    if dir == 1                     # +x
        return  x,  y / x,  z / x
    elseif dir == 2                 # -x
        return -x, -z / x, -y / x   # returns (f, b, c) with the b/c swap baked in
    elseif dir == 3                 # +y
        return  y,  z / y,  x / y
    elseif dir == 4                 # -y
        return -y, -x / y, -z / y
    elseif dir == 5                 # +z
        return  z,  x / z,  y / z
    else                            # -z
        return -z, -y / z, -x / z
    end
end

"""
    patch_to_global(mesh::InflatedCubeMesh{T}, patch_index, ξ_patch)
        → SVector{3, T}

Forward parametric map: given a 1-indexed patch (`1` for the inner
cube, `2..7` for inflation `(+x, -x, +y, -y, +z, -z)`, `8..13` for the
matching shell patches) and patch-local coordinates `ξ_patch ∈ [0, 1]³`,
return the physical point `P`. Exact inverse of [`global_to_patch`](@ref).
"""
function patch_to_global(mesh::InflatedCubeMesh{T}, patch_index::Integer,
                          ξ_patch::SVector{3, T}) where {T}
    ξ_a, ξ_b, ξ_c = ξ_patch[1], ξ_patch[2], ξ_patch[3]
    L = mesh.L;  R1 = mesh.R1;  R2 = mesh.R2

    if patch_index == 1
        x = -L + 2L * ξ_a
        y = -L + 2L * ξ_b
        z = -L + 2L * ξ_c
        return SVector{3, T}(x, y, z)
    elseif 2 ≤ patch_index ≤ 7
        dir = Int8(patch_index - 1)
        a = ξ_a
        b = -one(T) + T(2) * ξ_b
        c = -one(T) + T(2) * ξ_c
        Q = sqrt(one(T) + b * b + c * c)
        f = (one(T) - a) * L + a * R1 / Q
        vx, vy, vz = _patch_direction_vec(dir, b, c)
        return SVector{3, T}(f * vx, f * vy, f * vz)
    elseif 8 ≤ patch_index ≤ 13
        dir = Int8(patch_index - 7)
        a = ξ_a
        b = -one(T) + T(2) * ξ_b
        c = -one(T) + T(2) * ξ_c
        Q = sqrt(one(T) + b * b + c * c)
        r = (one(T) - a) * R1 + a * R2
        f = r / Q
        vx, vy, vz = _patch_direction_vec(dir, b, c)
        return SVector{3, T}(f * vx, f * vy, f * vz)
    else
        return SVector{3, T}(T(NaN), T(NaN), T(NaN))
    end
end

"""
    global_to_patch(mesh::InflatedCubeMesh{T}, patch_index, p; tol)
        → SVector{3, T}

Inverse of [`patch_to_global`](@ref): closed-form recovery of
`ξ_patch ∈ [0, 1]³` from a physical point `p` and a chosen patch.
Returns `SVector(NaN, NaN, NaN)` if `p` is outside the patch by more
than `tol`. Round-trip exact to machine precision.
"""
function global_to_patch(mesh::InflatedCubeMesh{T}, patch_index::Integer,
                          p::SVector{3, T};
                          tol = sqrt(eps(T))) where {T}
    L = mesh.L;  R1 = mesh.R1;  R2 = mesh.R2
    x, y, z = p[1], p[2], p[3]
    NaN_ξ = SVector{3, T}(T(NaN), T(NaN), T(NaN))

    if patch_index == 1
        ξ_a = (x + L) / (2L)
        ξ_b = (y + L) / (2L)
        ξ_c = (z + L) / (2L)
        if -tol ≤ ξ_a ≤ one(T) + tol &&
           -tol ≤ ξ_b ≤ one(T) + tol &&
           -tol ≤ ξ_c ≤ one(T) + tol
            return SVector{3, T}(clamp(ξ_a, zero(T), one(T)),
                                  clamp(ξ_b, zero(T), one(T)),
                                  clamp(ξ_c, zero(T), one(T)))
        end
        return NaN_ξ
    elseif 2 ≤ patch_index ≤ 13
        shell_patch = patch_index ≥ 8
        dir = shell_patch ? patch_index - 7 : patch_index - 1
        f_val, b, c = _inverse_dir_vec_3d(dir, x, y, z)
        if !(isfinite(b) && isfinite(c) && isfinite(f_val) && f_val > -tol)
            return NaN_ξ
        end
        Q = sqrt(one(T) + b * b + c * c)
        if shell_patch
            r = sqrt(x * x + y * y + z * z)
            a = (r - R1) / (R2 - R1)
        else
            a = (f_val - L) / (R1 / Q - L)
        end
        ξ_a = a
        ξ_b = (b + one(T)) / 2
        ξ_c = (c + one(T)) / 2
        if -tol ≤ ξ_a ≤ one(T) + tol &&
           -tol ≤ ξ_b ≤ one(T) + tol &&
           -tol ≤ ξ_c ≤ one(T) + tol
            return SVector{3, T}(clamp(ξ_a, zero(T), one(T)),
                                  clamp(ξ_b, zero(T), one(T)),
                                  clamp(ξ_c, zero(T), one(T)))
        end
        return NaN_ξ
    else
        return NaN_ξ
    end
end

"""
    locate_patch(mesh::InflatedCubeMesh{T}, p::SVector{3, T}; tol)
        → patch_index :: Int

Patch finder: return the 1-indexed patch (`1` inner cube, `2..7`
inflation `(+x, -x, +y, -y, +z, -z)`, `8..13` matching shell patches)
containing the physical point `p`. Returns `0` if `p` is outside the
ball `|x| ≤ R2 + tol`.
"""
function locate_patch(mesh::InflatedCubeMesh{T}, p::SVector{3, T};
                       tol = sqrt(eps(T))) where {T}
    x, y, z = p[1], p[2], p[3]
    L  = mesh.L
    R1 = mesh.R1
    R2 = mesh.R2

    absx = abs(x); absy = abs(y); absz = abs(z)

    if absx ≤ L + tol && absy ≤ L + tol && absz ≤ L + tol
        return 1
    end

    r = sqrt(x*x + y*y + z*z)
    if r > R2 + tol
        return 0
    end

    dir = if absx ≥ absy && absx ≥ absz
        x > 0 ? 1 : 2
    elseif absy ≥ absz
        y > 0 ? 3 : 4
    else
        z > 0 ? 5 : 6
    end
    return r ≤ R1 + tol ? dir + 1 : dir + 7
end

"""
    locate_element_in_patch(mesh::InflatedCubeMesh{T}, patch_index, ξ_patch)
        → (element_index, ξ_in_element)

Given a patch (`1..13`) and patch-local coordinates `ξ_patch ∈ [0, 1]³`,
return the global element index and the within-element reference
coordinate. Both coordinate systems are the *analytic* patch parameter
space — matches GLL-node placement by `_patch_point_and_jac` and is
the correct ξ to feed to downstream Lagrange interpolation.
"""
function locate_element_in_patch(mesh::InflatedCubeMesh{T},
                                  patch_index::Integer,
                                  ξ_patch::SVector{3, T}) where {T}
    ξ_a, ξ_b, ξ_c = ξ_patch[1], ξ_patch[2], ξ_patch[3]
    M = mesh.M; Mi = mesh.Mi; Ms = mesh.Ms
    Ma, Mb, Mc, elem_off = if patch_index == 1
        M, M, M, 0
    elseif 2 ≤ patch_index ≤ 7
        Mi, M, M, M*M*M + (patch_index - 2) * Mi * M * M
    else  # 8..13
        Ms, M, M, M*M*M + 6 * Mi * M * M + (patch_index - 8) * Ms * M * M
    end

    s_a = clamp(ξ_a, zero(T), one(T)) * Ma
    s_b = clamp(ξ_b, zero(T), one(T)) * Mb
    s_c = clamp(ξ_c, zero(T), one(T)) * Mc
    a_cell = min(Ma, max(1, floor(Int, s_a) + 1))
    b_cell = min(Mb, max(1, floor(Int, s_b) + 1))
    c_cell = min(Mc, max(1, floor(Int, s_c) + 1))
    ξ_elem_a = s_a - (a_cell - 1)
    ξ_elem_b = s_b - (b_cell - 1)
    ξ_elem_c = s_c - (c_cell - 1)
    e = elem_off + a_cell + Ma * ((b_cell - 1) + Mb * (c_cell - 1))
    return e, SVector{3, T}(ξ_elem_a, ξ_elem_b, ξ_elem_c)
end

