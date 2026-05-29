"""
    HexMeshes

Topology, parametric geometry, and host-side queries for conforming
tensor-product (line / quad / hex) meshes. Despite the historical name,
the package supports `D = 1, 2, 3` via the parametric type `Mesh{D, T}`
with the convenience aliases:

* `LineMesh{T} = Mesh{1, T}` (1D line elements)
* `QuadMesh{T} = Mesh{2, T}` (2D quadrilateral elements)
* `HexMesh{T}  = Mesh{3, T}` (3D hexahedral elements — historical default)

Decoupled from any particular element / quadrature / operator type —
those live in downstream packages (e.g. `WaveToySecondOrder` for 3D),
which combine a mesh with their own element to produce per-node
geometric data.

# Mesh families

Per-dimension uniform builders use the shape's natural name in the
function name (`line`, `quad`, `cubical`); the curvilinear families
follow the same `_<shape>_` convention.

| Family | 1D | 2D | 3D |
|---|---|---|---|
| Uniform | `make_line_mesh(T, M, x0, x1)` | `make_quad_mesh(T, Mx, My, x0, x1)` (Phase 3) | `make_cubical_mesh(T, Mx, My, Mz, x0, x1)` |
| Cubed-shape | n/a | `make_cubed_square_mesh(T, M, R)` (Phase 3) | `make_cubed_cube_mesh(T, M, R)` |
| Inflated-shape | n/a | `make_inflated_square_mesh(...)` (Phase 3) | `make_inflated_cube_mesh(T, L, R1, R2, M; outer_bc)` |

# Storage layout

Every `Mesh{D, T}` carries a kernel-resident `MeshConnectivity{D, MI, MI8}`
(four `(2D, Ne)` matrices: `neighbour`, `neighbour_face`, `orientation`,
`bdry`) plus host-only `vertex_coords::Matrix{T}` of shape `(D, Nv)`
and `vertex_idx::Matrix{Int}` of shape `(2^D, Ne)`. `InflatedCubeMesh{T}`
wraps a `HexMesh{T}` and adds a per-element `PatchInfo{T}` table that
downstream code uses to evaluate analytic Jacobians on the curvilinear
patches.

This package is pure-CPU: no `KernelAbstractions`, no `Adapt`. The
GPU-migration code (`to_device`, `Adapt.adapt_structure` rules) and
the operator-aware `MeshGeometry` type live downstream.
"""
module HexMeshes

using LinearAlgebra
using StaticArrays

# Topology types — parametric `Mesh{D, T}` and `MeshConnectivity{D, …}`
# with `LineMesh`, `QuadMesh`, `HexMesh` aliases for D = 1, 2, 3.
include("topology.jl")
# `PatchKind3D` / `PatchKind2D` enums (Int8-backed) and predicates,
# used as the `kind` field type of `PatchInfo` / `PatchInfo2D`. Must
# be loaded before `inflated_cube.jl` and `inflated_square.jl`.
include("patch_kinds.jl")
# `PatchSpec` / `FaceLink` / `SkeletonMesh` + the union-find vertex-dedup
# `_skeleton_to_mesh` builder. Currently 3D-specific; 2D will get its own
# skeleton machinery in Phase 3.
include("skeleton.jl")
# 3D `make_cubical_mesh` and `make_cubed_cube_mesh` (plus the cubed-cube
# inter-patch neighbour table).
include("builders.jl")
# `InflatedCubeMesh` / `PatchInfo` and `make_inflated_cube_mesh`,
# including the analytic patch parametrisation `_patch_point_and_jac`
# used by downstream geometry code.
include("inflated_cube.jl")
# 3D pure geometric maps: `trilinear_shape`/`_dshape`/`_map`/`_jacobian`.
include("geometry.jl")
# 3D host-side queries: `element_vertices`, `locate_point`,
# `invert_element_map`, `lagrange_basis`, `tensor_interp`,
# `interpolate_field`.
include("queries.jl")
# 1D linear shape functions: `linear_shape`/`_dshape`/`_map`/`_jacobian`.
include("geometry_1d.jl")
# 1D `make_line_mesh(T, M, x0, x1)`.
include("builders_1d.jl")
# 1D host-side queries.
include("queries_1d.jl")
# 2D bilinear shape functions: `bilinear_shape`/`_dshape`/`_map`/`_jacobian`.
include("geometry_2d.jl")
# `PatchSpec2D` / `FaceLink2D` / `SkeletonMesh2D` + `_skeleton_to_mesh_2d`.
include("skeleton_2d.jl")
# 2D `make_quad_mesh(T, Mx, My, x0, x1)` (single-patch direct) and
# `make_cubed_square_mesh(T, M, R)` (5-patch via 2D skeleton).
include("builders_2d.jl")
# `InflatedSquareMesh` / `PatchInfo2D` and `make_inflated_square_mesh`,
# including the analytic patch parametrisation `_patch_point_and_jac_2d`
# used by downstream geometry code.
include("inflated_square.jl")
# 2D host-side queries (Newton-iteration `invert_element_map`).
include("queries_2d.jl")

export
    # Topology — `Mesh{D, T}` is the underlying parametric type;
    # `LineMesh = Mesh{1}`, `QuadMesh = Mesh{2}`, `HexMesh = Mesh{3}`
    # are convenience aliases. All four names are exported.
    Mesh, LineMesh, QuadMesh, HexMesh, MeshConnectivity, nv,
    # Inflated cube wrapper + per-element analytic-patch tag
    InflatedCubeMesh, PatchInfo,
    # Inflated square (2D analog)
    InflatedSquareMesh, PatchInfo2D,
    # Patch-kind enums (Int8-backed) and predicates
    PatchKind3D, PatchKind2D,
    Cubical_3D, InflationPosX_3D, InflationNegX_3D, InflationPosY_3D,
    InflationNegY_3D, InflationPosZ_3D, InflationNegZ_3D,
    ShellPosX_3D, ShellNegX_3D, ShellPosY_3D, ShellNegY_3D,
    ShellPosZ_3D, ShellNegZ_3D,
    WedgePosX_3D, WedgeNegX_3D, WedgePosY_3D, WedgeNegY_3D,
    WedgePosZ_3D, WedgeNegZ_3D,
    Cubical_2D, InflationPosX_2D, InflationNegX_2D, InflationPosY_2D,
    InflationNegY_2D, ShellPosX_2D, ShellNegX_2D, ShellPosY_2D, ShellNegY_2D,
    WedgePosX_2D, WedgeNegX_2D, WedgePosY_2D, WedgeNegY_2D,
    is_cubical, is_inflation, is_shell, is_wedge, direction_of,
    # Skeleton-level types (advanced users / extension authors)
    PatchSpec, FaceLink, SkeletonMesh, interior_link, boundary_link,
    # Skeleton-level types (advanced users / extension authors) — 2D
    PatchSpec2D, FaceLink2D, SkeletonMesh2D,
    interior_link_2d, boundary_link_2d,
    # Public constructors
    make_cubical_mesh, make_cubed_cube_mesh, make_inflated_cube_mesh,
    make_line_mesh,
    make_quad_mesh, make_cubed_square_mesh, make_inflated_square_mesh,
    # Host-side queries
    element_vertices, locate_point, invert_element_map, interpolate_field,
    # Analytic patch ↔ global coordinate maps and point location
    # (multi-patch meshes only — InflatedSquareMesh, InflatedCubeMesh)
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
