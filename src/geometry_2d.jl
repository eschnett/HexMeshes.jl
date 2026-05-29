# ----------------------------------------------------------------------
# Per-element geometric map — 2D
#
# Each quadrilateral element is the image of the reference square
# `[0, 1]²` under the bilinear map defined by its four Gmsh-ordered
# corner vertices `v₁..v₄`. Writing `mξ = 1 − ξ`, `mη = 1 − η`, the
# four shape functions in Gmsh order are
#
#     N₁ = mξ·mη    (corner (−x, −y))
#     N₂ =  ξ·mη    (corner (+x, −y))
#     N₃ =  ξ· η    (corner (+x, +y))
#     N₄ = mξ· η    (corner (−x, +y))
#
# and `x(ξ, η) = Σ Nᵥ · vᵥ`. The Jacobian matrix `J[i, a] = ∂xᵢ/∂ξₐ`
# (with `ξ₁ = ξ`, `ξ₂ = η`) is constant for axis-aligned rectangles
# (diagonal) and varies with `(ξ, η)` for distorted / curved quads.

"""
    bilinear_shape(ξ, η) → NTuple{4, T}

The four bilinear shape functions of a 2D quadrilateral element at the
reference point `(ξ, η) ∈ [0, 1]²`, in Gmsh-canonical vertex order
`(v₁ at (0,0), v₂ at (1,0), v₃ at (1,1), v₄ at (0,1))`.
"""
@inline function bilinear_shape(ξ::T, η::T) where {T}
    mξ, mη = one(T) - ξ, one(T) - η
    return (mξ * mη,
             ξ * mη,
             ξ *  η,
            mξ *  η)
end

"""
    bilinear_dshape(ξ, η) → (NTuple{4, T}, NTuple{4, T})

Partial derivatives of the four bilinear shape functions at one
reference point. Returns two 4-tuples for `∂/∂ξ` and `∂/∂η`.
"""
@inline function bilinear_dshape(ξ::T, η::T) where {T}
    mξ, mη = one(T) - ξ, one(T) - η
    dξ = (-mη,  mη,  η, -η)
    dη = (-mξ, -ξ,   ξ,  mξ)
    return dξ, dη
end

"""
    bilinear_map(verts, ξ, η) → SVector{2, T}

Image of the reference-square point `(ξ, η) ∈ [0, 1]²` under the
bilinear map defined by the four Gmsh-ordered corner vertices `verts`.
"""
@inline function bilinear_map(verts::NTuple{4, SVector{2, T}},
                              ξ, η) where {T}
    Nv = bilinear_shape(T(ξ), T(η))
    p = zero(SVector{2, T})
    @inbounds for v in 1:4
        p += Nv[v] * verts[v]
    end
    return p
end

"""
    bilinear_jacobian(verts, ξ, η) → SMatrix{2, 2, T, 4}

Jacobian of the bilinear element map at reference-point `(ξ, η)`:
column `a` is `∂x / ∂ξₐ` (with `ξ₁ = ξ`, `ξ₂ = η`).
"""
@inline function bilinear_jacobian(verts::NTuple{4, SVector{2, T}},
                                   ξ, η) where {T}
    dξ, dη = bilinear_dshape(T(ξ), T(η))
    col1 = zero(SVector{2, T})
    col2 = zero(SVector{2, T})
    @inbounds for v in 1:4
        col1 += dξ[v] * verts[v]
        col2 += dη[v] * verts[v]
    end
    return SMatrix{2, 2, T}(col1[1], col1[2],
                            col2[1], col2[2])
end
