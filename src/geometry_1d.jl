# ----------------------------------------------------------------------
# Per-element geometric map — 1D
#
# Each line element is the image of the reference interval `[0, 1]`
# under the linear map defined by its two Gmsh-ordered endpoint
# vertices `v₁, v₂`:
#
#     N₁ = 1 − ξ    (vertex at ξ = 0)
#     N₂ =     ξ    (vertex at ξ = 1)
#
# and `x(ξ) = N₁(ξ)·v₁ + N₂(ξ)·v₂`. The Jacobian is the scalar
# `J = dx / dξ = v₂ − v₁`, constant along the element. For axis-aligned
# unit-length elements `J` is just the element width.

"""
    linear_shape(ξ) → NTuple{2, T}

The two linear shape functions of a 1D line element at the reference
point `ξ ∈ [0, 1]`, in Gmsh-canonical vertex order `(v₁ at ξ = 0,
v₂ at ξ = 1)`.
"""
@inline function linear_shape(ξ::T) where {T}
    return (one(T) - ξ, ξ)
end

"""
    linear_dshape(ξ) → NTuple{2, T}

Derivatives `∂Nᵥ / ∂ξ` of the linear shape functions. Constant in `ξ`,
returned as a 2-tuple to match the API shape of `bilinear_dshape` and
`trilinear_dshape`.
"""
@inline function linear_dshape(ξ::T) where {T}
    return (-one(T), one(T))
end

"""
    linear_map(verts, ξ) → SVector{1, T}

Image of the reference-interval point `ξ ∈ [0, 1]` under the linear
map defined by the two Gmsh-ordered endpoint vertices `verts`.
"""
@inline function linear_map(verts::NTuple{2, SVector{1, T}}, ξ) where {T}
    Nv = linear_shape(T(ξ))
    return Nv[1] * verts[1] + Nv[2] * verts[2]
end

"""
    linear_jacobian(verts, ξ) → SMatrix{1, 1, T, 1}

Jacobian of the linear element map at reference-point `ξ`: a 1×1
matrix whose single entry is `dx / dξ = v₂ − v₁`. The `SMatrix` wrapper
keeps the API uniform with `bilinear_jacobian` and `trilinear_jacobian`.
"""
@inline function linear_jacobian(verts::NTuple{2, SVector{1, T}}, ξ) where {T}
    _, dξ = linear_dshape(T(ξ))
    j = dξ * verts[2] + (-dξ) * verts[1]  # = verts[2] - verts[1]
    return SMatrix{1, 1, T, 1}(j[1])
end
