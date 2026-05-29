# Precision-genericity tests: every mesh family and analytic-locator
# path is exercised across `T ∈ (Float32, Float64, BigFloat)` to confirm
# the package is type-agnostic. Round-trip and containment tolerances
# scale with `default_tol(T) = sqrt(eps(T))`.

using HexMeshes
using HexMeshes: Mesh, PatchDesc, Cubic, Inflation, Shell,
                 make_uniform_line, make_uniform_quad, make_uniform_hex,
                 make_cubed_square_mesh, make_cubed_cube_mesh,
                 make_inflated_square_mesh, make_inflated_cube_mesh,
                 nv, npatches, locate_point, locate_patch,
                 patch_to_global, global_to_patch,
                 default_tol
using StaticArrays
using Test

@testset "precision-genericity (T ∈ Float32, Float64, BigFloat)" begin

    @testset "default_tol(T) = sqrt(eps(T))" begin
        for T in (Float32, Float64, BigFloat)
            @test default_tol(T) == sqrt(eps(T))
        end
    end

    @testset "make_uniform_line: vertex types and locate_point" begin
        for T in (Float32, Float64, BigFloat)
            m = make_uniform_line(T, 4, T(0), T(1))
            @test m isa Mesh{1, T}
            @test eltype(m.vertex_coords) === T
            @test m.patch_desc[1].kind === Cubic
            # locate_point at quarter-points snaps to the right element.
            for (xq, e_expected) in ((T(1)/8, 1), (T(3)/8, 2),
                                       (T(5)/8, 3), (T(7)/8, 4))
                e, ξ = locate_point(m, SVector{1, T}(xq))
                @test e == e_expected
                @test 0 ≤ Float64(ξ[1]) ≤ 1
            end
        end
    end

    @testset "make_uniform_quad: vertex types and locate_point" begin
        for T in (Float32, Float64, BigFloat)
            m = make_uniform_quad(T, 3, 3, T(0), T(1))
            @test m isa Mesh{2, T}
            @test eltype(m.vertex_coords) === T
            e, ξ = locate_point(m, SVector{2, T}(T(0.5), T(0.5)))
            @test e == 5     # center element of 3×3 grid
        end
    end

    @testset "make_uniform_hex: vertex types and locate_point" begin
        for T in (Float32, Float64, BigFloat)
            m = make_uniform_hex(T, 3, T(0), T(1))
            @test m isa Mesh{3, T}
            @test eltype(m.vertex_coords) === T
            e, ξ = locate_point(m, SVector{3, T}(T(0.5), T(0.5), T(0.5)))
            @test e == 2 + 1 * 3 + 1 * 9   # element (2, 2, 2) of 3³ grid = 14
        end
    end

    @testset "make_cubed_square_mesh: vertex types and shape" begin
        for T in (Float32, Float64, BigFloat)
            m = make_cubed_square_mesh(T, 3, T(1)/5)
            @test m isa Mesh{2, T}
            @test eltype(m.vertex_coords) === T
            @test npatches(m) == 5
            # Origin lies in the inner Cubic patch.
            @test locate_patch(m, SVector{2, T}(T(0), T(0))) == 1
        end
    end

    @testset "make_cubed_cube_mesh: vertex types and shape" begin
        for T in (Float32, Float64, BigFloat)
            m = make_cubed_cube_mesh(T, 2, T(1)/5)
            @test m isa Mesh{3, T}
            @test eltype(m.vertex_coords) === T
            @test npatches(m) == 7
            @test locate_patch(m, SVector{3, T}(T(0), T(0), T(0))) == 1
        end
    end

    @testset "make_inflated_square_mesh: patch round-trip" begin
        for T in (Float32, Float64, BigFloat)
            m = make_inflated_square_mesh(T, T(1)/10, T(3)/10, one(T), 3)
            @test m isa Mesh{2, T}
            @test npatches(m) == 9
            # Random round-trip ξ → global → ξ on every patch.
            tol = 5 * default_tol(T)
            for patch_index in 1:9
                pd = m.patch_desc[patch_index]
                for trial in 1:10
                    ξ = SVector{2, T}(T(rand()), T(rand()))
                    p  = patch_to_global(pd, ξ)
                    ξ2 = global_to_patch(pd, p)
                    @test !isnan(ξ2[1])
                    @test abs(ξ2[1] - ξ[1]) ≤ tol
                    @test abs(ξ2[2] - ξ[2]) ≤ tol
                end
            end
            # locate_point on a few representative physical points.
            for (xy, _) in (((T(0), T(0)), 1),
                             ((T(2)/10, T(0)), 2),
                             ((T(7)/10, T(0)), 6))
                e, ξ = locate_point(m, SVector{2, T}(xy...))
                @test e ≠ 0
                @test all(-default_tol(T) ≤ Float64(ξ_i) ≤ 1 + Float64(default_tol(T)) for ξ_i in ξ)
            end
        end
    end

    # ─── Rational precision: exact arithmetic on axis-aligned meshes ──
    #
    # `Rational{BigInt}` is exact under +, −, ×, ÷. For axis-aligned
    # `Cubic` patches the vertex map, `patch_to_global`, `global_to_patch`,
    # and `locate_point` all use only those operations, so vertices stay
    # bitwise-exact and `global_to_patch ∘ patch_to_global == id` is
    # exact (no `≤ tol` slack). Curvilinear patches use `sqrt` / `^`
    # which type-promote to `Float64`, so this testset deliberately
    # exercises only the cubical builders.

    @testset "make_uniform_line: Rational{BigInt} exact" begin
        T = Rational{BigInt}
        m = make_uniform_line(T, 4, T(0//1), T(1//1))
        @test m isa Mesh{1, T}
        @test eltype(m.vertex_coords) === T
        # Vertices are bitwise-exact rationals.
        @test m.vertex_coords[1, :] == [T(0//1), T(1//4), T(1//2), T(3//4), T(1//1)]
        # locate_point at exact rationals returns exact rational ξ.
        e, ξ = locate_point(m, SVector{1, T}(T(3//8)))
        @test e == 2
        @test ξ == SVector{1, T}(T(1//2))
        # Round-trip is bitwise exact.
        pd = m.patch_desc[1]
        for ξ_in in (SVector{1, T}(T(1//7)),
                      SVector{1, T}(T(3//11)),
                      SVector{1, T}(T(0)),
                      SVector{1, T}(T(1)))
            p_out  = patch_to_global(pd, ξ_in)
            ξ_back = global_to_patch(pd, p_out)
            @test ξ_back == ξ_in
        end
        # Point outside the patch reports element 0.
        @test locate_point(m, SVector{1, T}(T(5//4)))[1] == 0
    end

    @testset "make_uniform_quad: Rational{BigInt} exact" begin
        T = Rational{BigInt}
        m = make_uniform_quad(T, 3, 3, T(0//1), T(1//1))
        @test m isa Mesh{2, T}
        @test eltype(m.vertex_coords) === T
        # All vertex coordinates lie on the rational grid `{0, 1/3, 2/3, 1}²`.
        @test all(v -> v in (T(0//1), T(1//3), T(2//3), T(1//1)), m.vertex_coords)
        # Center: element (2, 2) = 5 of the 3×3 grid.
        e, ξ = locate_point(m, SVector{2, T}(T(1//2), T(1//2)))
        @test e == 5
        @test ξ == SVector{2, T}(T(1//2), T(1//2))
        # Round-trip.
        pd = m.patch_desc[1]
        for ξ_in in (SVector{2, T}(T(1//7),  T(3//11)),
                      SVector{2, T}(T(5//13), T(8//19)))
            p_out  = patch_to_global(pd, ξ_in)
            ξ_back = global_to_patch(pd, p_out)
            @test ξ_back == ξ_in
        end
    end

    @testset "make_uniform_hex: Rational{BigInt} exact" begin
        T = Rational{BigInt}
        m = make_uniform_hex(T, 4, T(0//1), T(1//1))
        @test m isa Mesh{3, T}
        @test eltype(m.vertex_coords) === T
        # Vertices on the 4×4×4 rational grid.
        allowed = (T(0//1), T(1//4), T(1//2), T(3//4), T(1//1))
        @test all(v -> v in allowed, m.vertex_coords)
        # locate_point on rational inputs returns exact ξ.
        e, ξ = locate_point(m, SVector{3, T}(T(1//3), T(2//3), T(7//8)))
        @test e ≠ 0
        @test ξ == SVector{3, T}(T(1//3), T(2//3), T(1//2))
        # Round-trip is bitwise exact.
        pd = m.patch_desc[1]
        for ξ_in in (SVector{3, T}(T(1//7),  T(3//11), T(5//13)),
                      SVector{3, T}(T(0),     T(1//2),  T(1)))
            p_out  = patch_to_global(pd, ξ_in)
            ξ_back = global_to_patch(pd, p_out)
            @test ξ_back == ξ_in
        end
        # Non-trivial origin shift preserves exactness.
        m2 = make_uniform_hex(T, 3, T(-1//1), T(2//1))
        @test m2.vertex_coords[1, 1] == T(-1//1)
        @test m2.vertex_coords[1, end] == T(2//1)
    end

    @testset "make_inflated_cube_mesh: patch round-trip" begin
        for T in (Float32, Float64, BigFloat)
            m = make_inflated_cube_mesh(T, T(1)/10, T(3)/10, one(T), 2)
            @test m isa Mesh{3, T}
            @test npatches(m) == 13
            tol = 5 * default_tol(T)
            for patch_index in 1:13
                pd = m.patch_desc[patch_index]
                for trial in 1:5
                    ξ = SVector{3, T}(T(rand()), T(rand()), T(rand()))
                    p  = patch_to_global(pd, ξ)
                    ξ2 = global_to_patch(pd, p)
                    @test !isnan(ξ2[1])
                    @test abs(ξ2[1] - ξ[1]) ≤ tol
                    @test abs(ξ2[2] - ξ[2]) ≤ tol
                    @test abs(ξ2[3] - ξ[3]) ≤ tol
                end
            end
            # locate_patch picks the inner cube at the origin.
            @test locate_patch(m, SVector{3, T}(T(0), T(0), T(0))) == 1
        end
    end

end
