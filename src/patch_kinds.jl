# Per-element patch-kind enums for the curvilinear inflated meshes.
#
# Both `PatchKind3D` (for `InflatedCubeMesh.patch_info`) and `PatchKind2D`
# (for `InflatedSquareMesh.patch_info`) are `@enum`-typed with `Int8`
# backing — storage is exactly one byte per element, but comparisons
# and dispatch read as named values rather than magic numbers. The
# numeric values themselves are load-bearing: `direction_of` recovers
# the per-direction code via `(Int8(k) - 1) % 2D + 1`, so reshuffling
# the enum members is a breaking change.
#
# Naming convention: member names are suffixed with `_3D` / `_2D` because
# Julia's `@enum` puts member names at module top-level — without the
# suffix `Cubical` and friends would collide between the two enums.

"""
    PatchKind3D <: Enum{Int8}

Per-element patch kind for `InflatedCubeMesh`. Stored as `Int8` in
`PatchInfo.kind`. The numeric encoding is:

* `Cubical_3D`         = 0  — inner cube (bilinear/trilinear path)
* `InflationPosX_3D`   = 1
* `InflationNegX_3D`   = 2
* `InflationPosY_3D`   = 3
* `InflationNegY_3D`   = 4
* `InflationPosZ_3D`   = 5
* `InflationNegZ_3D`   = 6
* `ShellPosX_3D`       = 7
* `ShellNegX_3D`       = 8
* `ShellPosY_3D`       = 9
* `ShellNegY_3D`       = 10
* `ShellPosZ_3D`       = 11
* `ShellNegZ_3D`       = 12
* `WedgePosX_3D`       = 13
* `WedgeNegX_3D`       = 14
* `WedgePosY_3D`       = 15
* `WedgeNegY_3D`       = 16
* `WedgePosZ_3D`       = 17
* `WedgeNegZ_3D`       = 18

The integer encoding is used arithmetically by `direction_of` and by
`_patch_point_and_jac` — don't reshuffle.

The wedge values are used in `PatchSpec.family` for the cubed-cube
skeleton family; they never appear in `PatchInfo.kind` (the per-element
classification of an `InflatedCubeMesh`).
"""
@enum PatchKind3D::Int8 begin
    Cubical_3D       = 0
    InflationPosX_3D = 1
    InflationNegX_3D = 2
    InflationPosY_3D = 3
    InflationNegY_3D = 4
    InflationPosZ_3D = 5
    InflationNegZ_3D = 6
    ShellPosX_3D     = 7
    ShellNegX_3D     = 8
    ShellPosY_3D     = 9
    ShellNegY_3D     = 10
    ShellPosZ_3D     = 11
    ShellNegZ_3D     = 12
    WedgePosX_3D     = 13
    WedgeNegX_3D     = 14
    WedgePosY_3D     = 15
    WedgeNegY_3D     = 16
    WedgePosZ_3D     = 17
    WedgeNegZ_3D     = 18
end

"""
    PatchKind2D <: Enum{Int8}

2D analog of `PatchKind3D` for `InflatedSquareMesh.patch_info`:

* `Cubical_2D`         = 0  — inner square (bilinear path)
* `InflationPosX_2D`   = 1
* `InflationNegX_2D`   = 2
* `InflationPosY_2D`   = 3
* `InflationNegY_2D`   = 4
* `ShellPosX_2D`       = 5
* `ShellNegX_2D`       = 6
* `ShellPosY_2D`       = 7
* `ShellNegY_2D`       = 8
* `WedgePosX_2D`       = 9
* `WedgeNegX_2D`       = 10
* `WedgePosY_2D`       = 11
* `WedgeNegY_2D`       = 12

The wedge values are used in `PatchSpec2D.family` for the cubed-square
skeleton family; they never appear in `PatchInfo2D.kind`.
"""
@enum PatchKind2D::Int8 begin
    Cubical_2D       = 0
    InflationPosX_2D = 1
    InflationNegX_2D = 2
    InflationPosY_2D = 3
    InflationNegY_2D = 4
    ShellPosX_2D     = 5
    ShellNegX_2D     = 6
    ShellPosY_2D     = 7
    ShellNegY_2D     = 8
    WedgePosX_2D     = 9
    WedgeNegX_2D     = 10
    WedgePosY_2D     = 11
    WedgeNegY_2D     = 12
end

# ─── Predicates ──────────────────────────────────────────────────────

"""
    is_cubical(k)   → Bool
    is_inflation(k) → Bool
    is_shell(k)     → Bool
    is_wedge(k)     → Bool

Patch-family predicates for `PatchKind3D` / `PatchKind2D`. Replace
ad-hoc range checks like `pi.kind == Int8(0)` or
`pi.kind ≥ Int8(7)` in user code.
"""
@inline is_cubical(k::PatchKind3D)   = k === Cubical_3D
@inline is_inflation(k::PatchKind3D) = InflationPosX_3D ≤ k ≤ InflationNegZ_3D
@inline is_shell(k::PatchKind3D)     = ShellPosX_3D     ≤ k ≤ ShellNegZ_3D
@inline is_wedge(k::PatchKind3D)     = WedgePosX_3D     ≤ k ≤ WedgeNegZ_3D

@inline is_cubical(k::PatchKind2D)   = k === Cubical_2D
@inline is_inflation(k::PatchKind2D) = InflationPosX_2D ≤ k ≤ InflationNegY_2D
@inline is_shell(k::PatchKind2D)     = ShellPosX_2D     ≤ k ≤ ShellNegY_2D
@inline is_wedge(k::PatchKind2D)     = WedgePosX_2D     ≤ k ≤ WedgeNegY_2D

"""
    direction_of(k) → Int8

Recover the per-direction code (`1 = +x`, `2 = −x`, `3 = +y`, …) from a
non-cubical patch kind. Result is undefined for the cubical kind.

The arithmetic uses `(Int8(k) - 1) % 2D + 1`, so it works uniformly for
inflation, shell, and wedge values: `InflationPosX_3D`, `ShellPosX_3D`,
and `WedgePosX_3D` all collapse to direction `1`.
"""
@inline direction_of(k::PatchKind3D) =
    Int8(((Int8(k) - Int8(1)) % Int8(6)) + Int8(1))
@inline direction_of(k::PatchKind2D) =
    Int8(((Int8(k) - Int8(1)) % Int8(4)) + Int8(1))
