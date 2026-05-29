# Mesh topology + geometry for a conforming tensor-product element mesh
# in `D = 1, 2, 3` spatial dimensions.
#
# A `Mesh{D, T}` is the data layer that decouples kernels from any
# particular grid arrangement: instead of indexing neighbours via
# `(mx ± 1, my, mz)` tuples on a `D`-dimensional lattice, the per-
# element loop walks a 1D list of elements and asks the mesh for each
# element's `2D` neighbours and the orientation of each face. This is
# the prerequisite for unstructured meshes (cubed sphere, multi-block
# topologies, future adaptive refinement).
#
# Vertex storage follows the standard finite-element / Gmsh convention:
# one shared coordinate table `vertex_coords` of shape `(D, Nv)` over
# the `Nv` distinct mesh vertices, plus a per-element connectivity
# table `vertex_idx` of shape `(2^D, Ne)` giving the index into
# `vertex_coords` of each corner of each element. Shared vertices on
# common faces of adjacent elements are stored once, not duplicated.
#
# Face index convention (used throughout for `neighbour`, `orientation`,
# `bdry`), in `1..2D`:
#
#     1 → −x face        2 → +x face        (always present)
#     3 → −y face        4 → +y face        (D ≥ 2)
#     5 → −z face        6 → +z face        (D = 3)
#
# Vertex index convention (`2^D` corners per element, Gmsh-canonical
# tensor-product ordering). In 3D the eight corners are:
#
#     1: (−x, −y, −z)    2: (+x, −y, −z)
#     3: (+x, +y, −z)    4: (−x, +y, −z)
#     5: (−x, −y, +z)    6: (+x, −y, +z)
#     7: (+x, +y, +z)    8: (−x, +y, +z)
#
# The 2D ordering omits the `±z` axis (corners 1..4 above); the 1D
# ordering keeps only `1: (−x)` and `2: (+x)`.

# `MeshConnectivity{D, MI, MI8}` bundles only what the kernel reads
# from a mesh: the four connectivity matrices, indexed by
# `(face, element)` with `face ∈ 1..2D`. Whoever launches the kernel
# sees this as one bitstype-friendly argument, so GPU adaptation can
# replace its arrays with `MtlDeviceMatrix` (etc.) in one shot.
# `Mesh{D, T}` (below) embeds a `MeshConnectivity{D, ...}` plus the
# host-only vertex and patch metadata that the kernel never touches.
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

"""
    Mesh{D, T, MI, MI8}

Conforming tensor-product element mesh in `D = 1, 2, 3` spatial
dimensions, with `T`-typed vertex coordinates. The kernel-read
connectivity arrays have concrete types `MI, MI8` (see below).

# Type parameters

* `D` — spatial dimension (`1`, `2`, or `3`).
* `T` — element type of `vertex_coords` (typically `Float32` /
  `Float64`).
* `MI`, `MI8` — concrete storage types of the kernel-read
  connectivity matrices (`AbstractMatrix{Int32}` / `Int8`
  respectively). Parametrised so the connectivity may live on a GPU
  as `CuArray`, `MtlArray`, `ROCArray`, etc. while the host-only
  fields stay plain `Matrix`.

# Fields

* `Ne :: Int` — number of elements.
* `conn :: MeshConnectivity{D, MI, MI8}` — kernel-read connectivity
  bundle. Read its fields directly as `mesh.conn.neighbour`,
  `mesh.conn.neighbour_face`, `mesh.conn.orientation`,
  `mesh.conn.bdry` (each of shape `(2D, Ne)`).
* `vertex_coords :: Matrix{T}` of shape `(D, Nv)` — Cartesian
  coordinates of every distinct mesh vertex. Host-only.
* `vertex_idx :: Matrix{Int}` of shape `(2^D, Ne)` — for each
  element, indices into `vertex_coords` of its `2^D` corners in
  Gmsh-canonical tensor-product ordering. Host-only.
* `patch_id :: Vector{Int32}` of length `Ne` — owning patch
  (1-indexed into `patch_desc`) of each element. Host-only.
* `patch_idx :: Matrix{Int32}` of shape `(D, Ne)` — 1-indexed
  position of each element inside its patch's structured grid.
  Host-only.
* `patch_desc :: Vector{PatchDesc{D, T}}` of length `npatches` —
  parametric geometry of each patch (`Cubic` / `Wedge` /
  `Inflation` / `Shell`). Host-only.
* `patch_element_offset :: Vector{Int}` of length `npatches + 1` —
  cumulative element offset per patch; element ids inside patch `p`
  occupy the range `(offset[p] + 1) : offset[p+1]`. Host-only.

The four patch fields enable `O(1)` analytic point location and
analytic-Jacobian dispatch in `make_geometry` — kernels never read
them.
"""
struct Mesh{D, T, MI, MI8}
    Ne                   :: Int
    conn                 :: MeshConnectivity{D, MI, MI8}
    vertex_coords        :: Matrix{T}
    vertex_idx           :: Matrix{Int}
    patch_id             :: Vector{Int32}
    patch_idx            :: Matrix{Int32}
    patch_desc           :: Vector{PatchDesc{D, T}}
    patch_element_offset :: Vector{Int}

    # Primary constructor. Patch fields are keyword args with empty
    # defaults for transition-period call sites that don't yet supply
    # them; once all builders are updated (Phase 4) the defaults will
    # be dropped.
    function Mesh{D, T}(Ne::Int,
                        conn::MeshConnectivity{D, MI, MI8},
                        vertex_coords::Matrix{T},
                        vertex_idx::Matrix{Int};
                        patch_id::Vector{Int32} = Int32[],
                        patch_idx::Matrix{Int32} = zeros(Int32, D, 0),
                        patch_desc::Vector{PatchDesc{D, T}} = PatchDesc{D, T}[],
                        patch_element_offset::Vector{Int} = Int[0],
                        ) where {D, T, MI, MI8}
        new{D, T, MI, MI8}(Ne, conn, vertex_coords, vertex_idx,
                           patch_id, patch_idx, patch_desc,
                           patch_element_offset)
    end

    # Back-compat constructor matching the old flat-field signature
    # (used internally by `_skeleton_to_mesh` and friends).
    function Mesh{D, T}(Ne::Int,
                        neighbour::MI, neighbour_face::MI8,
                        orientation::MI8, bdry::MI8,
                        vertex_coords::Matrix{T},
                        vertex_idx::Matrix{Int};
                        patch_id::Vector{Int32} = Int32[],
                        patch_idx::Matrix{Int32} = zeros(Int32, D, 0),
                        patch_desc::Vector{PatchDesc{D, T}} = PatchDesc{D, T}[],
                        patch_element_offset::Vector{Int} = Int[0],
                        ) where {D, T, MI, MI8}
        new{D, T, MI, MI8}(Ne,
                           MeshConnectivity{D, MI, MI8}(neighbour, neighbour_face,
                                                        orientation, bdry),
                           vertex_coords, vertex_idx,
                           patch_id, patch_idx, patch_desc,
                           patch_element_offset)
    end
end

"""
    npatches(mesh::Mesh) → Int

Number of patches in the mesh's `patch_desc` table. Returns `0` for
meshes that haven't yet been populated with patch metadata
(transition-period state).
"""
@inline npatches(mesh::Mesh) = length(mesh.patch_desc)

# Convenience aliases for the three supported dimensions. **Deprecated**
# — new code should prefer `Mesh{1}` / `Mesh{2}` / `Mesh{3}` directly.
# Each access emits a `depwarn` once per session.
Base.@deprecate_binding LineMesh Mesh{1} true
Base.@deprecate_binding QuadMesh Mesh{2} true
Base.@deprecate_binding HexMesh  Mesh{3} true

# Mesh has no `getproperty` forwarder; the kernel-read fields live on
# `mesh.conn.{neighbour, neighbour_face, orientation, bdry}` and must
# be accessed there. The transitional `mesh.X` shorthand emitted a
# deprecation warning and has been removed.

"""
    nv(mesh::Mesh) → Int

Number of distinct mesh vertices.
"""
nv(mesh::Mesh) = size(mesh.vertex_coords, 2)


# Face-local coordinate transforms for `Mesh{D}`:
#
# * `D = 1` — face is a 0-D point; no coordinates. Neither `_neigh_p`
#             nor `_neigh_pq` is used.
# * `D = 2` — face is a 1-D segment with one local coord `p ∈ 1..N`.
#             Orientation group is D₁ (2 elements: identity, reverse).
#             Encoded by `o ∈ 0..1`. See `_neigh_p`.
# * `D = 3` — face is a 2-D quad with `(p, q) ∈ 1..N²`. Orientation
#             group is D₄ (8 elements). Encoded by `o ∈ 0..7`. See
#             `_neigh_pq`.

"""
    _neigh_p(o, p, N) → p′

D₁ orientation transform for `Mesh{2}`: maps self's face-local `p`
into the neighbour's `p` using 1-indexed coordinates in `1..N`.

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

