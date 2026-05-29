# ----------------------------------------------------------------------
# Per-element geometric map
#
# Each hex element is the image of the reference cube `[0, 1]³` under the
# trilinear map defined by its eight Gmsh-ordered corner vertices `v₁..v₈`.
# Writing `m_a = 1 - a` for `a ∈ {ξ, η, ζ}`, the eight Gmsh-ordered shape
# functions are
#
#     N₁ = m_ξ·m_η·m_ζ      (corner (−x, −y, −z))
#     N₂ =  ξ ·m_η·m_ζ      (corner (+x, −y, −z))
#     N₃ =  ξ · η ·m_ζ      (corner (+x, +y, −z))
#     N₄ = m_ξ· η ·m_ζ      (corner (−x, +y, −z))
#     N₅ = m_ξ·m_η· ζ       (corner (−x, −y, +z))
#     N₆ =  ξ ·m_η· ζ       (corner (+x, −y, +z))
#     N₇ =  ξ · η · ζ       (corner (+x, +y, +z))
#     N₈ = m_ξ· η · ζ       (corner (−x, +y, +z))
#
# and the map is `x(ξ, η, ζ) = Σ Nᵥ(ξ, η, ζ) · vᵥ`. The Jacobian matrix
# `J[i, a] = ∂xᵢ / ∂ξₐ` (with `ξ₁ = ξ`, `ξ₂ = η`, `ξ₃ = ζ`) drops the
# right-side fall-through but otherwise follows the same shape derivatives.
# For axis-aligned hexes `J` is diagonal-constant; for curved/distorted
# hexes (cubed-cube outer patches, future cubed-sphere blocks) `J` varies
# with position and must be inverted per node when applying operators.

# Trilinear shape functions at one reference point. Returns the 8-tuple
# of values in Gmsh corner order, fully stack-allocated.
@inline function trilinear_shape(ξ::T, η::T, ζ::T) where {T}
    mξ, mη, mζ = one(T) - ξ, one(T) - η, one(T) - ζ
    return (mξ * mη * mζ,    ξ * mη * mζ,
             ξ *  η * mζ,   mξ *  η * mζ,
            mξ * mη *  ζ,    ξ * mη *  ζ,
             ξ *  η *  ζ,   mξ *  η *  ζ)
end

# Partial derivatives of the eight shape functions at one reference
# point. Returns three 8-tuples for `∂/∂ξ`, `∂/∂η`, `∂/∂ζ`.
@inline function trilinear_dshape(ξ::T, η::T, ζ::T) where {T}
    mξ, mη, mζ = one(T) - ξ, one(T) - η, one(T) - ζ
    dξ = (-mη * mζ,  mη * mζ,
           η * mζ,  -η * mζ,
          -mη *  ζ,  mη *  ζ,
           η *  ζ,  -η *  ζ)
    dη = (-mξ * mζ, -ξ * mζ,
           ξ * mζ,  mξ * mζ,
          -mξ *  ζ, -ξ *  ζ,
           ξ *  ζ,  mξ *  ζ)
    dζ = (-mξ * mη, -ξ * mη,
          -ξ *  η, -mξ *  η,
           mξ * mη,  ξ * mη,
           ξ *  η,  mξ *  η)
    return dξ, dη, dζ
end

"""
    trilinear_map(verts, ξ, η, ζ) → SVector{3, T}

Image of the reference-cube point `(ξ, η, ζ) ∈ [0, 1]³` under the trilinear
map defined by the eight Gmsh-ordered corner vertices `verts`.
"""
@inline function trilinear_map(verts::NTuple{8, SVector{3, T}},
                                ξ, η, ζ) where {T}
    Nv = trilinear_shape(T(ξ), T(η), T(ζ))
    p = zero(SVector{3, T})
    @inbounds for v in 1:8
        p += Nv[v] * verts[v]
    end
    return p
end

"""
    trilinear_jacobian(verts, ξ, η, ζ) → SMatrix{3, 3, T, 9}

Jacobian of the trilinear element map at reference-point `(ξ, η, ζ)`:
column `a` is `∂x / ∂ξₐ` (with `ξ₁ = ξ`, `ξ₂ = η`, `ξ₃ = ζ`).
"""
@inline function trilinear_jacobian(verts::NTuple{8, SVector{3, T}},
                                     ξ, η, ζ) where {T}
    dξ, dη, dζ = trilinear_dshape(T(ξ), T(η), T(ζ))
    col1 = zero(SVector{3, T})
    col2 = zero(SVector{3, T})
    col3 = zero(SVector{3, T})
    @inbounds for v in 1:8
        col1 += dξ[v] * verts[v]
        col2 += dη[v] * verts[v]
        col3 += dζ[v] * verts[v]
    end
    return SMatrix{3, 3, T}(col1[1], col1[2], col1[3],
                            col2[1], col2[2], col2[3],
                            col3[1], col3[2], col3[3])
end
