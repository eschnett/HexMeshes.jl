# 2D analog of `inflated_cube.jl`. 9-patch conforming quad mesh of the
# disk `|x| ≤ R2`:
#
#   * one axis-aligned inner square `[-L, L]²` (`M²` elements),
#   * four inflation patches that bridge each square edge to the inner
#     circle at radius `R1` (`M_i × M` elements each),
#   * four annular-shell patches that bridge the inner circle to the
#     outer circle at `R2` (`M_s × M` elements each).
#
# Total element count: `M² + 4·M·(M_i + M_s)`.
#
# Mirrors the 3D inflated cube structurally; the orientation group on
# each inter-patch edge is D₁ (identity / reversal) rather than D₄.

# Tangential-edge connectivity for the inflation / shell patches of an
# inflated-square mesh. Same indexing scheme as `_WEDGE_NEIGHBOUR_2D`
# but DIFFERENT entries: the inflated square's `_patch_direction_vec_2d`
# is right-handed with sign-flips for `−x, +y`, which changes which face
# of each neighbour patch meets a given square-corner edge. Derived
# once on paper from the patch parametrisations; all four corner-edge
# orientations remain 0.
const _INFLATION_NEIGHBOUR_2D = (
    ((4, 4), (3, 3)),   # +x: faces 3,4 → -y face 4, +y face 3
    ((3, 4), (4, 3)),   # -x:           → +y face 4, -y face 3
    ((1, 4), (2, 3)),   # +y:           → +x face 4, -x face 3
    ((2, 4), (1, 3)),   # -y:           → -x face 4, +x face 3
)

"""
    PatchInfo2D{T}

Per-element parametric description used by `make_geometry` for elements
of an `InflatedSquareMesh` that should be discretised with an analytic
curvilinear map instead of the default bilinear interpolation of the
four corners.

`kind` selects the patch family:

* `0`     — bilinear (inner square; the angular / radial fields are unused).
* `1..4`  — inflation patch in directions `(+x, -x, +y, -y)`: bridges the
  inner-square edge at `r = L` (where `r` is the dominant coordinate) to
  the inner circle at radius `R1`.
* `5..8`  — annular-shell patch in the same directional order: bridges
  the inner circle at `R1` to the outer circle at `R2`.

For curved patches, `(a_lo, a_hi)` is the radial parameter range
(`s ∈ [0, 1]` for inflation, `ρ ∈ [0, 1]` for shell) and `(b_lo, b_hi)`
is the tangent-angular range (`⊂ [-1, 1]`).
"""
struct PatchInfo2D{T}
    kind :: PatchKind2D
    a_lo :: T
    a_hi :: T
    b_lo :: T
    b_hi :: T
end

"""
    InflatedSquareMesh{T}

A 9-patch conforming quad mesh:

* one axis-aligned inner square `[-L, L]²` (`M²` elements),
* four inflation patches bridging each square edge to the inner circle
  at radius `R1` (`M_i × M` elements each),
* four annular-shell patches bridging the inner circle `R1` to the
  outer circle `R2` (`M_s × M` elements each).

Total element count: `M² + 4·M·(M_i + M_s)`.

# Fields

* `base :: QuadMesh{T}` — connectivity + bilinear vertex info. Host-side
  queries (`element_vertices`, `nv`, `locate_point`, plotting) work
  through this just like a plain `QuadMesh`.
* `patch_info :: Vector{PatchInfo2D{T}}` of length `base.Ne` —
  per-element parametric description used by `make_geometry` to evaluate
  the analytic Jacobian on the curved patches.
* `L, R1, R2 :: T` — inner-square half-edge, inner-circle radius, outer-
  circle radius. Required: `0 < L · √2 < R1 < R2`.
"""
struct InflatedSquareMesh{T, MI, MI8}
    base       :: QuadMesh{T, MI, MI8}
    patch_info :: Vector{PatchInfo2D{T}}
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

# Forward `QuadMesh` accessors through the wrapper so existing host code
# (`mesh.Ne`, `mesh.bdry`, `mesh.vertex_coords`, …) keeps working
# unchanged.
@inline function Base.getproperty(m::InflatedSquareMesh, name::Symbol)
    if name === :base || name === :patch_info ||
       name === :L || name === :R1 || name === :R2 ||
       name === :M || name === :Mi || name === :Ms
        return getfield(m, name)
    else
        return getproperty(getfield(m, :base), name)
    end
end
Base.propertynames(m::InflatedSquareMesh) =
    (:base, :patch_info, :L, :R1, :R2, :M, :Mi, :Ms,
     :Ne, :conn, :vertex_coords, :vertex_idx,
     :neighbour, :neighbour_face, :orientation, :bdry)

nv(mesh::InflatedSquareMesh) = nv(mesh.base)

# Forward `element_vertices` queries through the wrapper.
@inline function element_vertices(mesh::InflatedSquareMesh{T}, e::Integer) where {T}
    return element_vertices(mesh.base, e)
end

"""
    make_inflated_square_mesh(::Type{T}, L, R1, R2, M; M_i, M_s, outer_bc) → InflatedSquareMesh{T}

Build a 9-patch inflated square mesh of the disk `|x| ≤ R2`:

* an axis-aligned inner square `[-L, L]²` with `M × M` elements,
* four inflation patches interpolating each square edge to the inner
  circle `r = R1`, with `M_i × M` elements each,
* four annular-shell patches interpolating the inner circle to the
  outer circle `r = R2`, with `M_s × M` elements each.

`M_i` defaults to `round((R1 - (1 + √2)/2 · L) / h)` (average radial gap
between square and inner circle, expressed in inner-square-edge cells
`h = 2L/M`). `M_s` defaults to `round((R2 - R1) / h)`. Both clip to at
least 1.

The shell patches use the parameterisation `r(ρ) = (1 − ρ)·R1 + ρ·R2`
along every radial ray, so they have exactly constant radial spacing
`(R2 − R1) / M_s`. The inflation patches use
`r(s, b) = (1 − s)·L + s · R1 / √(1 + b²)`, so their radial spacing is
constant on average (varies between edge centre and corner).

The outer boundary `r = R2` is tagged on every shell-patch outer face
according to `outer_bc`:

* `:dirichlet` (default) — `bdry = 1`.
* `:sommerfeld` — `bdry = 7`, downstream Sommerfeld-aware kernels
  interpret as a first-order radiative BC.
"""
function make_inflated_square_mesh(::Type{T}, L::Real, R1::Real, R2::Real,
                                    M::Int;
                                    M_i::Union{Nothing, Int} = nothing,
                                    M_s::Union{Nothing, Int} = nothing,
                                    outer_bc::Symbol = :dirichlet) where {T}
    skel       = _inflated_square_skeleton(T, L, R1, R2, M; M_i, M_s, outer_bc)
    base       = _skeleton_to_mesh_2d(skel)
    patch_info = _build_inflated_square_patch_info(skel, base.Ne)
    # Recover the (possibly auto-defaulted) radial cell counts directly
    # from the skeleton — `skel.patches[2]` is the first inflation patch
    # (`Ma = Mi`), `skel.patches[6]` is the first shell patch (`Ma = Ms`).
    Mi = skel.patches[2].Ma
    Ms = skel.patches[6].Ma
    return InflatedSquareMesh(base, patch_info, T(L), T(R1), T(R2),
                              M, Mi, Ms)
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

    Mi = M_i === nothing ?
         max(1, round(Int, (R1 - (1 + sqrt(2))/2 * L) / h)) :
         M_i
    Ms = M_s === nothing ?
         max(1, round(Int, (R2 - R1) / h)) :
         M_s
    @assert Mi ≥ 1
    @assert Ms ≥ 1

    z = zero(T)
    o = one(T)

    # Patch 1: inner square [-L, L]².
    inner = PatchSpec2D{T}(M, M, Cubical_2D,
                            -Lv, Lv, -Lv, Lv,
                            z, z, z)

    # Patches 2..5: inflation in directions (+x, -x, +y, -y).
    inflation_families = (InflationPosX_2D, InflationNegX_2D,
                           InflationPosY_2D, InflationNegY_2D)
    # Patches 6..9: shell in same direction order.
    shell_families     = (ShellPosX_2D, ShellNegX_2D,
                           ShellPosY_2D, ShellNegY_2D)

    patches = PatchSpec2D{T}[inner]
    for fam in inflation_families
        push!(patches, PatchSpec2D{T}(Mi, M, fam,
                                       z, o, -o, o,
                                       Lv, R1v, R2v))
    end
    for fam in shell_families
        push!(patches, PatchSpec2D{T}(Ms, M, fam,
                                       z, o, -o, o,
                                       Lv, R1v, R2v))
    end
    @assert length(patches) == 9

    faces = Matrix{FaceLink2D}(undef, 4, length(patches))

    # ---- Square ↔ inflation interfaces. ----
    # Inner-square face `f` connects to the inflation patch whose
    # direction matches that face, at the inflation's face 1 (inner-
    # radial, `a = 0`). The right-handed `_patch_direction_vec_2d`
    # uses b-sign-flips for `-x` and `+y`, giving orientation 1 (D₁
    # reversal) on those two cube↔inflation links. The other two
    # match directly (orientation 0).
    CUBE_FACE_TO_DIR        = (2, 1, 4, 3)
    CUBE_FACE_ORIENTATION   = (Int8(1), Int8(0), Int8(0), Int8(1))
    for f in 1:4
        d  = CUBE_FACE_TO_DIR[f]
        ip = d + 1                       # inflation patch id
        oo = CUBE_FACE_ORIENTATION[f]
        faces[f, 1]  = interior_link_2d(ip, 1, oo)
        faces[1, ip] = interior_link_2d(1,  f, oo)
    end

    # ---- Inflation tangential / outer-radial faces. ----
    # Tangential faces 3..4 connect to adjacent inflation patches via
    # `_INFLATION_NEIGHBOUR_2D` (all corner-edge orientations 0).
    # Face 2 (outer-radial, `a = 1`) connects to the same-direction
    # shell patch's face 1 with orientation 0.
    for d in 1:4
        ip = d + 1                       # inflation patch
        sp = d + 5                       # shell patch (same direction)
        faces[2, ip] = interior_link_2d(sp, 1, 0)
        for f in 3:4
            neigh_dir, neigh_face = _INFLATION_NEIGHBOUR_2D[d][f - 2]
            faces[f, ip] = interior_link_2d(neigh_dir + 1, neigh_face, 0)
        end
    end

    # ---- Shell tangential / outer-circle faces. ----
    # Face 1: → inflation face 2 (inner-radial side).
    # Face 2: outer-circle boundary, tag from `outer_bc`.
    # Faces 3..4: adjacent shell patches via `_INFLATION_NEIGHBOUR_2D`.
    for d in 1:4
        sp = d + 5
        ip = d + 1
        faces[1, sp] = interior_link_2d(ip, 2, 0)
        faces[2, sp] = boundary_link_2d(outer_bc_tag)
        for f in 3:4
            neigh_dir, neigh_face = _INFLATION_NEIGHBOUR_2D[d][f - 2]
            faces[f, sp] = interior_link_2d(neigh_dir + 5, neigh_face, 0)
        end
    end

    return SkeletonMesh2D{T}(patches, faces)
end

# Build the per-element `PatchInfo2D` table that
# `make_geometry(::InflatedSquareMesh)` will read to dispatch on patch
# kind and recover the parameter-space extent of each element.
function _build_inflated_square_patch_info(skel::SkeletonMesh2D{T}, Ne::Int) where {T}
    patch_info = Vector{PatchInfo2D{T}}(undef, Ne)
    e = 0
    zT = zero(T)
    for ps in skel.patches
        kind = ps.family
        for b in 1:ps.Mb, a in 1:ps.Ma
            e += 1
            if is_cubical(kind)
                patch_info[e] = PatchInfo2D{T}(Cubical_2D, zT, zT, zT, zT)
            else
                a_lo = ps.a_lo + (ps.a_hi - ps.a_lo) * T(a - 1) / T(ps.Ma)
                a_hi = ps.a_lo + (ps.a_hi - ps.a_lo) * T(a)     / T(ps.Ma)
                b_lo = ps.b_lo + (ps.b_hi - ps.b_lo) * T(b - 1) / T(ps.Mb)
                b_hi = ps.b_lo + (ps.b_hi - ps.b_lo) * T(b)     / T(ps.Mb)
                patch_info[e] = PatchInfo2D{T}(kind, a_lo, a_hi, b_lo, b_hi)
            end
        end
    end
    @assert e == Ne
    return patch_info
end

"""
    _patch_point_and_jac_2d(pi, ξ, η_ref, L, R1, R2) → (P, J)

Analytic element map for one node of an `InflatedSquareMesh` curved
patch. Given a per-element `PatchInfo2D` and a reference-square
coordinate `(ξ, η_ref) ∈ [0, 1]²`, returns the physical point
`P :: SVector{2, T}` and the `SMatrix{2, 2}` Jacobian
`J[i, a] = ∂P_i / ∂ξₐ_ref`. For `pi.kind == Cubical_2D` (inner square),
use the bilinear path instead — this routine assumes a curved patch.
"""
@inline function _patch_point_and_jac_2d(pi::PatchInfo2D{T},
                                          ξ::T, η_ref::T,
                                          L::T, R1::T, R2::T) where {T}
    da = pi.a_hi - pi.a_lo
    db = pi.b_hi - pi.b_lo
    a  = pi.a_lo + da * ξ
    b  = pi.b_lo + db * η_ref

    Q  = sqrt(one(T) + b * b)
    Q3 = Q * Q * Q
    pi_is_shell = is_shell(pi.kind)
    dir         = direction_of(pi.kind)

    if pi_is_shell
        rval  = (one(T) - a) * R1 + a * R2
        f     = rval / Q
        df_da = (R2 - R1) / Q
        df_db = -rval * b / Q3
    else
        f     = (one(T) - a) * L + a * R1 / Q
        df_da = -L + R1 / Q
        df_db = -a * R1 * b / Q3
    end

    vx, vy, dvxb, dvyb = _patch_direction_vec_2d_and_derivs(dir, b)

    Px = f * vx
    Py = f * vy

    dPa_x = df_da * vx;            dPa_y = df_da * vy
    dPb_x = df_db * vx + f * dvxb; dPb_y = df_db * vy + f * dvyb

    J = SMatrix{2, 2, T}(dPa_x * da, dPa_y * da,
                         dPb_x * db, dPb_y * db)
    P = SVector{2, T}(Px, Py)
    return P, J
end

# ----------------------------------------------------------------------
# Analytic point location for `InflatedSquareMesh`
#
# The 9-patch topology has a closed-form inverse for the patch-finder
# step (no tree search): determine which patch contains a point by
# comparing `|x|, |y|, r = √(x² + y²)` against the geometric thresholds
# `L`, `R1`, `R2`, and which of the four radial directions by the
# dominant axis. Within a patch, the parametric inverse is also closed
# form. Finally a single Newton iteration on the bilinear element map
# recovers the precise reference coordinate inside the element.
#
# This drops point-location cost from `O(Ne)` (brute-force scan) to
# `O(1)` per query, making `interpolate_field` usable for hot loops.

"""
    locate_patch(mesh::InflatedSquareMesh{T}, p::SVector{2, T}; tol)
        → patch_index :: Int

Patch finder: return the 1-indexed patch (`1` for the inner square,
`2..5` for inflation `(+x, -x, +y, -y)`, `6..9` for matching shell
patches) containing the physical point `p`. Returns `0` if `p` is
outside the disk `|x| ≤ R2 + tol`.

Patch selection is by geometric region — `|x|, |y| ≤ L` → inner;
`|P| ≤ R1` → inflation; `|P| ≤ R2` → shell — combined with a
dominant-axis test to pick the radial direction. For points exactly
on inter-patch boundaries the inner patch wins, then x-direction wins
over y-direction (deterministic).
"""
function locate_patch(mesh::InflatedSquareMesh{T}, p::SVector{2, T};
                       tol = sqrt(eps(T))) where {T}
    x, y = p[1], p[2]
    L  = mesh.L
    R1 = mesh.R1
    R2 = mesh.R2

    absx = abs(x)
    absy = abs(y)

    if absx ≤ L + tol && absy ≤ L + tol
        return 1
    end

    r = sqrt(x * x + y * y)
    if r > R2 + tol
        return 0
    end

    # Dominant-axis direction.
    dir = if absx ≥ absy
        x > 0 ? 1 : 2
    else
        y > 0 ? 3 : 4
    end

    # `r ≤ R1` selects inflation; `r > R1` selects shell.
    return r ≤ R1 + tol ? dir + 1 : dir + 5
end

"""
    locate_element_in_patch(mesh::InflatedSquareMesh{T}, patch_index, ξ_patch)
        → (element_index, ξ_in_element)

Given a patch (`1..9`) and patch-local coordinates `ξ_patch ∈ [0, 1]²`,
return the global element index and the within-element reference
coordinate `ξ_in_element ∈ [0, 1]²`. Both `ξ_patch` and `ξ_in_element`
live in the *analytic* patch parameter space — this matches how GLL
nodes are placed inside curvilinear elements (via
`_patch_point_and_jac_2d`), so the recovered `ξ_in_element` can be fed
directly to Lagrange interpolation against the patch GLL grid without
any further Newton refinement.
"""
function locate_element_in_patch(mesh::InflatedSquareMesh{T},
                                  patch_index::Integer,
                                  ξ_patch::SVector{2, T}) where {T}
    ξ_a, ξ_b = ξ_patch[1], ξ_patch[2]
    M = mesh.M; Mi = mesh.Mi; Ms = mesh.Ms
    Ma, Mb, elem_off = if patch_index == 1
        M, M, 0
    elseif 2 ≤ patch_index ≤ 5
        Mi, M, M*M + (patch_index - 2) * Mi * M
    else                # 6..9
        Ms, M, M*M + 4 * Mi * M + (patch_index - 6) * Ms * M
    end

    s_a = clamp(ξ_a, zero(T), one(T)) * Ma
    s_b = clamp(ξ_b, zero(T), one(T)) * Mb
    a_cell = min(Ma, max(1, floor(Int, s_a) + 1))
    b_cell = min(Mb, max(1, floor(Int, s_b) + 1))
    ξ_elem_a = s_a - (a_cell - 1)
    ξ_elem_b = s_b - (b_cell - 1)
    e = elem_off + a_cell + Ma * (b_cell - 1)
    return e, SVector{2, T}(ξ_elem_a, ξ_elem_b)
end

# ----------------------------------------------------------------------
# Analytic patch ↔ global coordinate maps.
#
# `patch_to_global` and `global_to_patch` are exact inverses of each
# other on each patch's parameter domain — no Newton iteration needed.
# Together with `locate_patch`, they support O(1) point location and
# precise element-ξ recovery for downstream interpolation.

# Per-direction inverse of the inflation/shell parametric direction
# vector `_patch_direction_vec_2d`. Given (x, y) and a direction
# `dir ∈ 1..4`, returns `(f, b)` such that  `P = f · v(b)`  for the
# corresponding inflation/shell map. The b/sign-flip pattern matches
# the right-handed `_patch_direction_vec_2d`:
#   +x:  v = ( 1,  b)  →  f = x,   b = y/x
#   -x:  v = (-1, -b)  →  f = -x,  b = y/x
#   +y:  v = (-b,  1)  →  f = y,   b = -x/y
#   -y:  v = ( b, -1)  →  f = -y,  b = -x/y
@inline function _inverse_dir_vec_2d(dir::Integer, x::T, y::T) where {T}
    if dir == 1                     # +x
        return  x,  y / x
    elseif dir == 2                 # -x
        return -x,  y / x
    elseif dir == 3                 # +y
        return  y, -x / y
    else                            # -y
        return -y, -x / y
    end
end

"""
    patch_to_global(mesh::InflatedSquareMesh{T}, patch_index, ξ_patch)
        → SVector{2, T}

Forward parametric map: given a 1-indexed patch (`1` for the inner
square, `2..5` for inflation `(+x, -x, +y, -y)`, `6..9` for the matching
shell patches) and patch-local coordinates `ξ_patch ∈ [0, 1]²`, return
the physical point `P`.

This is the exact inverse of [`global_to_patch`](@ref) on each patch's
parameter domain.
"""
function patch_to_global(mesh::InflatedSquareMesh{T}, patch_index::Integer,
                          ξ_patch::SVector{2, T}) where {T}
    ξ_a, ξ_b = ξ_patch[1], ξ_patch[2]
    L = mesh.L;  R1 = mesh.R1;  R2 = mesh.R2

    if patch_index == 1
        # Inner square [-L, L]²: affine in (ξ_a, ξ_b)
        x = -L + 2L * ξ_a
        y = -L + 2L * ξ_b
        return SVector{2, T}(x, y)
    elseif 2 ≤ patch_index ≤ 5
        # Inflation patch in direction dir = patch_index − 1.
        dir = patch_index - 1
        a = ξ_a
        b = -one(T) + T(2) * ξ_b
        Q = sqrt(one(T) + b * b)
        f = (one(T) - a) * L + a * R1 / Q
        vx, vy = _patch_direction_vec_2d(dir, b)
        return SVector{2, T}(f * vx, f * vy)
    elseif 6 ≤ patch_index ≤ 9
        # Shell patch in direction dir = patch_index − 5.
        dir = patch_index - 5
        a = ξ_a
        b = -one(T) + T(2) * ξ_b
        Q = sqrt(one(T) + b * b)
        r = (one(T) - a) * R1 + a * R2
        f = r / Q
        vx, vy = _patch_direction_vec_2d(dir, b)
        return SVector{2, T}(f * vx, f * vy)
    else
        return SVector{2, T}(T(NaN), T(NaN))
    end
end

"""
    global_to_patch(mesh::InflatedSquareMesh{T}, patch_index, p; tol)
        → SVector{2, T}

Inverse of [`patch_to_global`](@ref): given a physical point `p` and a
patch index, return the patch-local coordinates `ξ_patch ∈ [0, 1]²`,
or `SVector(NaN, NaN)` if `p` is outside the patch by more than `tol`
(FP-noise slack). Round-trip exact:
`global_to_patch(i, patch_to_global(i, ξ)) ≈ ξ` to machine precision.

The patch family selects a closed-form inverse — no Newton iteration.
"""
function global_to_patch(mesh::InflatedSquareMesh{T}, patch_index::Integer,
                          p::SVector{2, T};
                          tol = sqrt(eps(T))) where {T}
    L = mesh.L;  R1 = mesh.R1;  R2 = mesh.R2
    x, y = p[1], p[2]
    NaN_ξ = SVector{2, T}(T(NaN), T(NaN))

    if patch_index == 1
        ξ_a = (x + L) / (2L)
        ξ_b = (y + L) / (2L)
        if -tol ≤ ξ_a ≤ one(T) + tol && -tol ≤ ξ_b ≤ one(T) + tol
            return SVector{2, T}(clamp(ξ_a, zero(T), one(T)),
                                  clamp(ξ_b, zero(T), one(T)))
        end
        return NaN_ξ
    elseif 2 ≤ patch_index ≤ 9
        shell_patch = patch_index ≥ 6
        dir = shell_patch ? patch_index - 5 : patch_index - 1
        f_val, b = _inverse_dir_vec_2d(dir, x, y)
        # f must be strictly positive on the correct side; b within
        # [-1, 1] places the point in the right angular wedge.
        if !(isfinite(b) && isfinite(f_val) && f_val > -tol)
            return NaN_ξ
        end
        Q = sqrt(one(T) + b * b)
        if shell_patch
            r = sqrt(x * x + y * y)
            a = (r - R1) / (R2 - R1)
        else
            denom = R1 / Q - L
            a = (f_val - L) / denom
        end
        ξ_a = a
        ξ_b = (b + one(T)) / 2
        if -tol ≤ ξ_a ≤ one(T) + tol && -tol ≤ ξ_b ≤ one(T) + tol
            return SVector{2, T}(clamp(ξ_a, zero(T), one(T)),
                                  clamp(ξ_b, zero(T), one(T)))
        end
        return NaN_ξ
    else
        return NaN_ξ
    end
end
