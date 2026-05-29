# Mesh topology + geometry for a conforming hexahedral element mesh.
#
# The mesh is the data layer that decouples the 3D kernels from any
# particular grid arrangement: instead of indexing neighbours via
# `(mx ± 1, my, mz)` tuples on a 3D lattice, the per-element loop walks
# a 1D list of elements and asks the mesh for each element's six
# neighbours and the orientation of each face. This is the prerequisite
# for unstructured meshes (cubed sphere, multi-block topologies, future
# adaptive refinement).
#
# Vertex storage follows the standard finite-element / Gmsh convention:
# one shared coordinate table `vertex_coords` of shape `(3, Nv)` over the
# `Nv` distinct mesh vertices, plus a per-element connectivity table
# `vertex_idx` of shape `(8, Ne)` giving the index into `vertex_coords`
# of each of the eight corners of each hex. Shared vertices on common
# faces of adjacent elements are stored once, not duplicated.
#
# Face index convention (used throughout for `neighbour`, `orientation`,
# `bdry`):
#
#     1 → −x face        2 → +x face
#     3 → −y face        4 → +y face
#     5 → −z face        6 → +z face
#
# Vertex index convention (8 corners of each hex, Gmsh-canonical ordering):
#
#     1: (−x, −y, −z)    2: (+x, −y, −z)
#     3: (+x, +y, −z)    4: (−x, +y, −z)
#     5: (−x, −y, +z)    6: (+x, −y, +z)
#     7: (+x, +y, +z)    8: (−x, +y, +z)

"""
    HexMesh{T}

Connectivity + geometry of a conforming hexahedral mesh.

# Fields

* `Ne :: Int` — number of elements.
* `neighbour :: Matrix{Int32}` of shape `(6, Ne)` — element ID of the
  neighbour across each of the six faces (face ordering as above). `0`
  marks an outer-boundary face. Stored as `Int32` rather than `Int` so
  that `nbr` reads from inside the GPU kernel stay 32-bit (Apple
  Silicon GPUs emulate 64-bit integer arithmetic, which costs ~4× on
  every 4-D array offset computation in `_add_face_sat!`). 2^31 ≫
  any realistic element count, so the narrower type is safe.
* `neighbour_face :: Matrix{Int8}` of shape `(6, Ne)` — face index of
  the neighbour element that abuts face `f` of `e`. For an axis-aligned
  cubical mesh this is just the opposite face index along the same axis
  (`(2,1,4,3,6,5)[f]`); for multi-patch meshes where the two sides have
  different orthogonal-axis conventions it can be any of the six values.
  `0` on outer-boundary faces (where `neighbour == 0`).
* `orientation :: Matrix{Int8}` of shape `(6, Ne)` — `0..7` encoding of
  the D₄ transform that maps this face's local face-quadrature `(p, q)`
  coordinates to the matching face on the neighbour. `0` is the identity
  (used everywhere on axis-aligned meshes and on the cubed-cube mesh
  by construction). The transform table is documented in `_neigh_pq`
  below.
* `bdry :: Matrix{Int8}` of shape `(6, Ne)` — boundary-condition tag,
  nonzero only on outer faces.
* `vertex_coords :: Matrix{T}` of shape `(3, Nv)` — Cartesian coordinates
  of every distinct vertex in the mesh. Shared between adjacent elements.
* `vertex_idx :: Matrix{Int}` of shape `(8, Ne)` — for each element, the
  indices into `vertex_coords` of its eight corners (in the canonical
  vertex ordering above).
"""
# `MeshConnectivity{D, MI, MI8}` bundles only what the kernel reads from
# a mesh: the four connectivity matrices, indexed by `(face, element)`,
# with `face ∈ 1..2D` faces per element. Whoever launches the kernel
# sees this as one bitstype-friendly argument, so GPU adaptation can
# replace its arrays with `MtlDeviceMatrix` (etc.) in one shot.
# `Mesh{D, T}` (below) embeds a `MeshConnectivity{D, ...}` plus the
# host-only vertex metadata that the kernel never touches.
#
# The `D` type parameter is the spatial dimension (1, 2, or 3). All
# three are supported; `HexMesh = Mesh{3}` is the historical alias.
struct MeshConnectivity{D, MI, MI8}
    neighbour      :: MI    # (2D, Ne)
    neighbour_face :: MI8   # (2D, Ne)
    orientation    :: MI8   # (2D, Ne)
    bdry           :: MI8   # (2D, Ne)
end

# Outer constructor for `MeshConnectivity{D}(args...)` that infers MI,
# MI8 from the argument types (Julia doesn't auto-derive these from a
# partially-specified parametric constructor across all versions).
MeshConnectivity{D}(neighbour::MI, neighbour_face::MI8,
                    orientation::MI8, bdry::MI8) where {D, MI, MI8} =
    MeshConnectivity{D, MI, MI8}(neighbour, neighbour_face, orientation, bdry)

# `Mesh{D, T}` is parametrised on the *concrete* storage types of the
# kernel-read connectivity matrices (via `conn::MeshConnectivity`) so
# that they may live on a GPU as `CuArray`, `MtlArray`, `ROCArray`, etc.
# Host-only fields (`vertex_coords`, `vertex_idx`) stay concrete
# `Matrix` — they are read by plotting and diagnostics, never by the
# kernel, so there is no benefit to migrating them onto a device.
# A `Base.getproperty` forwarder below preserves `mesh.neighbour` /
# `mesh.bdry` / etc. so existing host code does not need updating.
#
# Array shapes (with `D` the spatial dimension):
#
# * `vertex_coords :: Matrix{T}` is `(D, Nv)`.
# * `vertex_idx    :: Matrix{Int}` is `(2^D, Ne)` — `2^D` corners per
#   element in Gmsh-canonical tensor-product ordering.
# * Connectivity matrices are all `(2D, Ne)` — `2D` faces per element.
struct Mesh{D, T, MI, MI8}
    Ne            :: Int
    conn          :: MeshConnectivity{D, MI, MI8}
    vertex_coords :: Matrix{T}
    vertex_idx    :: Matrix{Int}

    function Mesh{D, T}(Ne::Int,
                        conn::MeshConnectivity{D, MI, MI8},
                        vertex_coords::Matrix{T},
                        vertex_idx::Matrix{Int}) where {D, T, MI, MI8}
        new{D, T, MI, MI8}(Ne, conn, vertex_coords, vertex_idx)
    end

    # Back-compat constructor matching the old flat-field signature
    # (used internally by `_skeleton_to_mesh` and friends).
    function Mesh{D, T}(Ne::Int,
                        neighbour::MI, neighbour_face::MI8,
                        orientation::MI8, bdry::MI8,
                        vertex_coords::Matrix{T},
                        vertex_idx::Matrix{Int}) where {D, T, MI, MI8}
        new{D, T, MI, MI8}(Ne,
                           MeshConnectivity{D, MI, MI8}(neighbour, neighbour_face,
                                                        orientation, bdry),
                           vertex_coords, vertex_idx)
    end
end

# Convenience aliases — these preserve the historical 3D API surface
# (`HexMesh{T}(...)` still works, `m isa HexMesh{T}` still works) and
# introduce the 1D / 2D variants in the same pattern.
const LineMesh = Mesh{1}
const QuadMesh = Mesh{2}
const HexMesh  = Mesh{3}

# `mesh.neighbour` etc. forward to `mesh.conn.*` so existing call sites
# (mesh-build code, tests, diagnostics) keep working without churn.
@inline function Base.getproperty(m::Mesh, name::Symbol)
    if name === :neighbour || name === :neighbour_face ||
       name === :orientation || name === :bdry
        getfield(getfield(m, :conn), name)
    else
        getfield(m, name)
    end
end
Base.propertynames(m::Mesh) = (:Ne, :conn, :vertex_coords, :vertex_idx,
                               :neighbour, :neighbour_face, :orientation, :bdry)

"""
    nv(mesh::Mesh) → Int

Number of distinct mesh vertices.
"""
nv(mesh::Mesh) = size(mesh.vertex_coords, 2)


# Face-local coordinate transforms per dimension:
#
# * 1D — face is a 0-D point; no coordinates. `_neigh_pq` is unused.
# * 2D — face is a 1-D segment with one local coord `p ∈ 1..N`.
#        Orientation group is D₁ (2 elements: identity, reverse).
#        Encoded by `o ∈ 0..1`.
# * 3D — face is a 2-D quad with `(p, q) ∈ 1..N²`. Orientation group
#        is D₄ (8 elements). Encoded by `o ∈ 0..7`.

"""
    _neigh_p(o, p, N) → p′

D₁ orientation transform for 2D meshes: maps self's face-local `p` into
the neighbour's `p` using 1-indexed coordinates in `1..N`.

* `o = 0` — identity (`p′ = p`)
* `o = 1` — reversal (`p′ = N + 1 − p`)

Used by downstream 2D kernels to read across an inter-element edge
when the two sides' tangent direction is reversed.
"""
@inline function _neigh_p(o::Integer, p::Integer, N::Integer)
    return o == 0 ? p : (N + one(N) - p)
end

# D₄ orientation transform: maps self's face-local `(p, q)` into the
# neighbour's `(p, q)` using 1-indexed coordinates in `1..N`. Resolves
# the eight rotations + reflections of the unit square. Used by the
# kernel `_face_sat_compute!` to read across an inter-element face
# when the two sides' tangent axes don't line up directly.
#
# This is the 1-indexed cousin of `_neigh_pq_vertex` (0-indexed,
# 0..M) and `_neigh_pq_cell` (1-indexed cells, 1..M); the three are
# the same group operation re-expressed for the index range each
# caller works in.
@inline function _neigh_pq(o::Integer, p::Integer, q::Integer, N::Integer)
    # Mirror in whatever integer type `p, q, N` are passed as. Callers
    # in the kernel pass `Int32` to keep the index chain off the
    # 64-bit slow path on NVIDIA / Apple Silicon. The `one(N)` keeps
    # the "+1" in the same type as `N`.
    np1 = N + one(N)
    if     o == 0;  return (p,        q       )
    elseif o == 1;  return (q,        np1 - p)
    elseif o == 2;  return (np1 - p,  np1 - q)
    elseif o == 3;  return (np1 - q,  p       )
    elseif o == 4;  return (np1 - p,  q       )
    elseif o == 5;  return (q,        p       )
    elseif o == 6;  return (p,        np1 - q)
    else            return (np1 - q,  np1 - p)
    end
end

