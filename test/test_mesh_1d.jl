# Tests for the 1D `LineMesh` family. Mirrors the existing 3D
# `test_mesh.jl` testset (where applicable — 1D has no curvilinear
# families, no D₄ orientation, etc.).

using HexMeshes
using HexMeshes: Mesh, make_uniform_line, nv, npatches,
                 element_vertices, locate_point, invert_element_map,
                 locate_patch, locate_element_in_patch,
                 patch_to_global, global_to_patch,
                 interpolate_field,
                 linear_shape, linear_dshape, linear_map, linear_jacobian
using StaticArrays
using Test

count_zero_neighbours_1d(m) = count(==(0), m.conn.neighbour)

@testset "mesh (1D)" begin

    @testset "make_uniform_line(T, …): shapes" begin
        m = make_uniform_line(Float64, 5, 0.0, 1.0)
        @test m isa Mesh{1, Float64}
        @test m.Ne == 5
        @test size(m.conn.neighbour)      == (2, 5)
        @test size(m.conn.neighbour_face) == (2, 5)
        @test size(m.conn.orientation)    == (2, 5)
        @test size(m.conn.bdry)           == (2, 5)
        @test size(m.vertex_coords)  == (1, 6)
        @test size(m.vertex_idx)     == (2, 5)
        @test nv(m) == 6
    end

    @testset "shared vertices: adjacent elements reuse the same vertex ID" begin
        m = make_uniform_line(Float64, 4, 0.0, 1.0)
        # Element e's +x vertex (vertex 2) == element (e+1)'s −x vertex
        # (vertex 1).
        for e in 1:(m.Ne - 1)
            @test m.vertex_idx[2, e] == m.vertex_idx[1, e+1]
        end
        # No duplication: Nv exactly Ne+1.
        @test nv(m) == m.Ne + 1
    end

    @testset "orientation is trivially 0 (1D has no orientation group)" begin
        m = make_uniform_line(Float64, 4, 0.0, 1.0)
        @test all(m.conn.orientation .== 0)
    end

    @testset "boundary tags only on outer faces" begin
        for M in (1, 2, 4, 8)
            m = make_uniform_line(Float64, M, 0.0, 1.0)
            for e in 1:m.Ne, f in 1:2
                @test (m.conn.bdry[f, e] ≠ 0) == (m.conn.neighbour[f, e] == 0)
            end
            # Tag convention: face 1 of element 1 → tag 1; face 2 of last
            # element → tag 2.
            @test m.conn.bdry[1, 1]    == 1
            @test m.conn.bdry[2, m.Ne] == 2
            # Every other face has tag 0.
            @test count_zero_neighbours_1d(m) == 2
        end
    end

    @testset "neighbour pointers are symmetric" begin
        m = make_uniform_line(Float64, 6, 0.0, 1.0)
        opposite = (2, 1)
        for e in 1:m.Ne, f in 1:2
            n = m.conn.neighbour[f, e]
            n == 0 && continue
            @test m.conn.neighbour[opposite[f], n] == e
            @test m.conn.neighbour_face[f, e]      == opposite[f]
        end
    end

    @testset "vertex coordinates: equispaced on [x0, x1]" begin
        m = make_uniform_line(Float64, 4, -1.0, 3.0)
        # h = (3 - (-1)) / 4 = 1
        @test m.vertex_coords[1, :] == [-1.0, 0.0, 1.0, 2.0, 3.0]
        # Element e spans (vertex_idx[1, e]) to (vertex_idx[2, e])
        for e in 1:m.Ne
            v1 = m.vertex_coords[1, m.vertex_idx[1, e]]
            v2 = m.vertex_coords[1, m.vertex_idx[2, e]]
            @test v2 - v1 ≈ 1.0
        end
    end

    @testset "linear shape functions: partition of unity & polynomial exactness" begin
        for T in (Float64, Float32)
            # Partition of unity at random ξ
            for ξ_raw in (0.0, 0.25, 0.5, 0.7, 1.0)
                ξ = T(ξ_raw)
                N1, N2 = linear_shape(ξ)
                @test N1 + N2 ≈ one(T)
                # Linear-exactness: image of (0, 1) at ξ equals ξ.
                verts = (SVector{1, T}(zero(T)), SVector{1, T}(one(T)))
                p = linear_map(verts, ξ)
                @test p[1] ≈ ξ
            end
        end
    end

    @testset "linear_jacobian: scalar, equal to v2 − v1" begin
        verts = (SVector{1, Float64}(0.5), SVector{1, Float64}(2.3))
        J = linear_jacobian(verts, 0.4)
        @test J isa SMatrix{1, 1, Float64, 1}
        @test J[1, 1] ≈ 2.3 - 0.5
    end

    @testset "invert_element_map (deprecated): inverts linear_map exactly" begin
        # The 1D `invert_element_map` is deprecated (`locate_point` now
        # uses the analytic patch path). Smoke-tested here to confirm the
        # function still behaves correctly while it's in the deprecation
        # window. The depwarn fires once per session.
        verts = (SVector{1, Float64}(0.5), SVector{1, Float64}(2.3))
        for ξ_raw in (0.0, 0.3, 0.6, 1.0)
            ξ_in = SVector{1, Float64}(ξ_raw)
            p = linear_map(verts, ξ_in[1])
            ξ_out, ok = invert_element_map(verts, p)
            @test ok
            @test ξ_out[1] ≈ ξ_in[1]
        end
        # Out-of-element point: ok = false.
        _, ok = invert_element_map(verts, SVector{1, Float64}(5.0))
        @test !ok
    end

    @testset "locate_point: returns the element containing the point" begin
        m = make_uniform_line(Float64, 4, 0.0, 1.0)
        # Element e covers [(e-1)/4, e/4].
        @test locate_point(m, SVector{1, Float64}(0.1))[1] == 1
        @test locate_point(m, SVector{1, Float64}(0.3))[1] == 2
        @test locate_point(m, SVector{1, Float64}(0.6))[1] == 3
        @test locate_point(m, SVector{1, Float64}(0.9))[1] == 4
        # Out of domain → 0.
        @test locate_point(m, SVector{1, Float64}(2.0))[1] == 0
    end

    @testset "make_uniform_line: patch_to_global ↔ global_to_patch round-trip" begin
        m = make_uniform_line(Float64, 5, -1.0, 2.0)
        @test npatches(m) == 1
        pd = m.patch_desc[1]
        # ξ → global → ξ
        for _ in 1:25
            ξ = SVector{1, Float64}(rand())
            p = patch_to_global(pd, ξ)
            ξ2 = global_to_patch(pd, p)
            @test !isnan(ξ2[1])
            @test ξ2[1] ≈ ξ[1] atol=1e-14
        end
        # global → ξ → global on sample points inside [-1, 2]
        for x in (-1.0, -0.7, 0.0, 0.5, 1.5, 2.0)
            p = SVector{1, Float64}(x)
            @test locate_patch(m, p) == 1
            ξ = global_to_patch(pd, p)
            @test !isnan(ξ[1])
            p2 = patch_to_global(pd, ξ)
            @test p2[1] ≈ p[1] atol=1e-14
        end
        # Outside domain → locate_patch returns 0.
        @test locate_patch(m, SVector{1, Float64}(5.0)) == 0
    end

    @testset "interpolate_field: linear field is recovered exactly" begin
        m = make_uniform_line(Float64, 4, 0.0, 1.0)
        N = 3
        # GLL-like nodes on [0, 1] for a 3-point reference element
        xs = [0.0, 0.5, 1.0]
        # Build u(x) = 2x + 1 on the mesh by sampling at GLL nodes per element.
        u = Matrix{Float64}(undef, N, m.Ne)
        for e in 1:m.Ne
            v1 = m.vertex_coords[1, m.vertex_idx[1, e]]
            v2 = m.vertex_coords[1, m.vertex_idx[2, e]]
            for i in 1:N
                x_phys = v1 + xs[i] * (v2 - v1)
                u[i, e] = 2 * x_phys + 1
            end
        end
        # Query at arbitrary points.
        for x_q in (0.05, 0.3, 0.55, 0.8, 0.999)
            val = interpolate_field(m, xs, u, SVector{1, Float64}(x_q))
            @test val ≈ 2 * x_q + 1 atol=1e-12
        end
    end

end
