# HexMeshes.jl

Topology, parametric geometry, and host-side queries for conforming
hex meshes. Pure CPU; no element / quadrature / operator structure
(those live in downstream packages such as
[WaveToySecondOrder.jl](https://github.com/eschnetter/WaveToySecondOrder.jl)).

## Supported mesh families

| Constructor | Domain | Topology |
|---|---|---|
| `make_cubical_mesh(T, M[x,y,z], x0, x1)` | `[x0, x1]³` | uniform axis-aligned brick |
| `make_cubed_cube_mesh(T, M, R)` | `[-1, 1]³` | 1 central cube + 6 radial wedges (cubed-cube) |
| `make_inflated_cube_mesh(T, L, R1, R2, M; outer_bc)` | ball `|x| ≤ R2` | 1 cube + 6 inflation patches + 6 spherical-shell patches; `outer_bc ∈ (:dirichlet, :sommerfeld)` |

All meshes are conforming: every face is shared by exactly two
elements (or is on the domain boundary), and the connectivity
includes the D₄ orientation that maps face-local `(p, q)` between
the two sides.

## Construction approach

Two-stage skeleton-based build, with **integer-only vertex dedup**
via union-find:

1. A small `SkeletonMesh{T}` describes patches (`PatchSpec`) + inter-
   patch face links (`FaceLink`). No floating-point coordinates at
   this stage.
2. `_skeleton_to_mesh(skel)` enumerates per-patch vertices as integer
   4-tuples `(p, i, j, k)`, unifies face-shared ids via union-find,
   assigns dense canonical ids, then evaluates the family-specific
   parametric map to produce coordinates. Avoids the 1-ULP dedup
   failures that broke a position-keyed `Dict{NTuple{3, T}, Int}` at
   `L = 0.1, M = 4` (cube and patch arithmetic disagreed by 1 ULP on
   non-power-of-2 divisions).

## Storage layout

```julia
struct HexMesh{T, MI, MI8}
    Ne            :: Int
    conn          :: MeshConnectivity{MI, MI8}    # kernel-readable
    vertex_coords :: Matrix{T}                    # (3, Nv), host-only
    vertex_idx    :: Matrix{Int}                  # (8, Ne), host-only
end

struct MeshConnectivity{MI, MI8}
    neighbour      :: MI    # (6, Ne)  Int32
    neighbour_face :: MI8   # (6, Ne)  Int8
    orientation    :: MI8   # (6, Ne)  Int8 (D₄ index 0..7)
    bdry           :: MI8   # (6, Ne)  Int8 boundary-condition tag
end
```

The `MeshConnectivity` matrices are bitstype-friendly so downstream
packages can migrate them to a GPU backend with `Adapt`/`KernelAbstractions`.

## License

MIT.
