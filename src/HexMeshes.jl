"""
    HexMeshes

Topology, parametric geometry, and host-side queries for conforming
tensor-product (line / quad / hex) meshes in `D = 1, 2, 3` spatial
dimensions. The core type is `Mesh{D, T}`; convenience aliases
`LineMesh = Mesh{1}`, `QuadMesh = Mesh{2}`, `HexMesh = Mesh{3}` are
provided (transitional; new code should prefer `Mesh{D}` directly).

Decoupled from any particular element / quadrature / operator type —
those live in downstream packages.

# Mesh families

| Family    | 1D                  | 2D                               | 3D                                                   |
|-----------|---------------------|----------------------------------|------------------------------------------------------|
| Uniform   | `make_uniform_line` | `make_uniform_quad`              | `make_uniform_hex`                                   |
| Cubed     | n/a                 | `make_cubed_square_mesh`         | `make_cubed_cube_mesh`                               |
| Inflated  | n/a                 | `make_inflated_square_mesh(...)` | `make_inflated_cube_mesh(...; outer_bc=:dirichlet)`  |

# Storage layout

Every `Mesh{D, T}` carries:

* `conn :: MeshConnectivity{D, MI, MI8}` — kernel-resident
  `(neighbour, neighbour_face, orientation, bdry)`, each `(2D, Ne)`.
* `vertex_coords :: Matrix{T}` of shape `(D, Nv)` — host-only.
* `vertex_idx :: Matrix{Int}` of shape `(2^D, Ne)` — host-only.
* `patch_desc :: Vector{PatchDesc{D, T}}` — one entry per patch in
  the multi-block topology.
* `patch_id :: Vector{Int32}` of length `Ne`, `patch_idx :: Matrix{Int32}`
  of shape `(D, Ne)`, `patch_element_offset :: Vector{Int}` — per-element
  patch metadata supporting `O(1)` analytic point location and
  analytic-Jacobian dispatch in `make_geometry`.

This package is pure-CPU: no `KernelAbstractions`, no `Adapt`. The
GPU-migration code (`to_device`, `Adapt.adapt_structure` rules) and
the operator-aware `MeshGeometry` type live downstream.
"""
module HexMeshes

using LinearAlgebra
using StaticArrays

# Patch-descriptor types: `PatchKind` enum + `PatchCubic` / `PatchWedge`
# / `PatchInflation` / `PatchShell` + the packed `PatchDesc{D, T}`
# union. Must be loaded before `topology.jl` because `Mesh{D, T}` now
# carries `patch_desc :: Vector{PatchDesc{D, T}}`.
include("patches.jl")
# Topology types — parametric `Mesh{D, T}` and `MeshConnectivity{D, …}`.
include("topology.jl")
# Unified D-generic skeleton builder: `SkeletonMesh{D, T}`, `FaceLink`,
# `_skeleton_to_mesh`, the per-patch parametric vertex map
# `_patch_vertex_position`, and the direction vectors. Internal-only.
include("skeleton.jl")
# Analytic per-element position + Jacobian for `Inflation` / `Shell`
# patches (both 2D and 3D); used by downstream `make_geometry`.
include("patch_jacobian.jl")
# Multi-patch mesh builders (skeleton + topology only). Analytic patch
# maps / locators live in `queries.jl` / `queries_2d.jl`.
include("builders.jl")            # 3D `make_uniform_hex`, `make_cubed_cube_mesh`
include("inflated_cube.jl")       # 3D `make_inflated_cube_mesh`
include("radial_shell.jl")        # 3D `make_radial_shell_mesh` (BH excision)
# 3D pure geometric maps: `trilinear_shape`/`_dshape`/`_map`/`_jacobian`.
include("geometry.jl")
# 3D host-side queries: `element_vertices`, `locate_point`,
# `invert_element_map`, `lagrange_basis`, `tensor_interp`,
# `interpolate_field`, plus the analytic patch ↔ global maps
# (`patch_to_global`, `global_to_patch`, `locate_patch`,
# `locate_element_in_patch`).
include("queries.jl")
# 1D linear shape functions: `linear_shape`/`_dshape`/`_map`/`_jacobian`.
include("geometry_1d.jl")
# 1D `make_uniform_line(T, M, x0, x1)`.
include("builders_1d.jl")
# 1D host-side queries.
include("queries_1d.jl")
# 2D bilinear shape functions: `bilinear_shape`/`_dshape`/`_map`/`_jacobian`.
include("geometry_2d.jl")
# 2D builders: `make_uniform_quad` (single-patch direct) and
# `make_cubed_square_mesh` (5-patch via the unified D-generic skeleton).
include("builders_2d.jl")
# 2D `make_inflated_square_mesh`.
include("inflated_square.jl")
# 2D host-side queries + 2D analytic patch maps.
include("queries_2d.jl")

export
    # Topology
    Mesh, LineMesh, QuadMesh, HexMesh, MeshConnectivity, nv, npatches,
    # Patch-descriptor types + the dimension-neutral PatchKind enum.
    # `Cubic` / `Wedge` / `Inflation` / `Shell` are the four geometric
    # families a `PatchDesc{D, T}` may carry; constructors take one of
    # the variant types and zero-initialise the others.
    PatchKind, Cubic, Wedge, Inflation, Shell, WarpedCubic,
    PatchCubic, PatchWedge, PatchInflation, PatchShell, PatchWarpedCubic,
    PatchDesc,
    dims, n_elements,
    # Face-orientation group marker (`OrientationGroup{2}` = D₁,
    # `OrientationGroup{3}` = D₄). Type-only; documents the dispatch
    # structure of the internal `_neigh_*` orientation transforms.
    OrientationGroup, n_orientations,
    # Public constructors. Uniform builders are
    # `make_uniform_{line,quad,hex}`; `make_line_mesh` / `make_quad_mesh`
    # / `make_cubical_mesh` are deprecated aliases.
    make_uniform_line, make_uniform_quad, make_uniform_hex,
    make_warped_uniform_hex,
    make_line_mesh, make_quad_mesh, make_cubical_mesh,
    make_cubed_square_mesh, make_inflated_square_mesh,
    make_cubed_cube_mesh, make_inflated_cube_mesh,
    make_radial_shell_mesh,
    # Host-side queries
    element_vertices, locate_point, invert_element_map, interpolate_field,
    default_tol,
    # Analytic patch ↔ global coordinate maps and point location
    patch_to_global, global_to_patch,
    locate_patch, locate_element_in_patch,
    # Pure geometric maps (useful for downstream `make_geometry`-style code)
    trilinear_shape, trilinear_dshape, trilinear_map, trilinear_jacobian,
    # 1D analogs
    linear_shape, linear_dshape, linear_map, linear_jacobian,
    # 2D analogs
    bilinear_shape, bilinear_dshape, bilinear_map, bilinear_jacobian,
    lagrange_basis, tensor_interp

end # module HexMeshes
