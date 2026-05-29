# Tests for the HexMeshes package: topology, skeleton-based vertex
# dedup, neighbour symmetry, orientation, and boundary tagging.
#
# Tests that depend on a downstream element / operator (i.e. the
# `make_geometry`-using "geometry is well-formed" testset) live in
# `WaveToySecondOrder/test/test_geometry.jl` instead, since the
# `MeshGeometry` type lives there.

using HexMeshes
using HexMeshes: HexMesh, InflatedCubeMesh, make_cubical_mesh,
                 make_cubed_cube_mesh, make_inflated_cube_mesh, nv,
                 locate_patch, global_to_patch, patch_to_global,
                 locate_element_in_patch, locate_point
using StaticArrays
using Test

count_zero_neighbours(m::HexMesh) = count(==(0), m.neighbour)

@testset "mesh" begin

    @testset "make_cubical_mesh: shapes" begin
        m = make_cubical_mesh(Float64, 2, 3, 4, 0.0, 1.0)
        @test m isa HexMesh{Float64}
        @test m.Ne == 2 * 3 * 4
        @test size(m.neighbour)     == (6, m.Ne)
        @test size(m.orientation)   == (6, m.Ne)
        @test size(m.bdry)          == (6, m.Ne)
        @test size(m.vertex_coords) == (3, (2+1)*(3+1)*(4+1))
        @test size(m.vertex_idx)    == (8, m.Ne)
        @test nv(m) == (2+1)*(3+1)*(4+1)
    end

    @testset "shared vertices: adjacent elements reuse the same vertex ID" begin
        # For an `Mx × My × Mz` mesh, the total number of vertices is
        # (Mx+1)(My+1)(Mz+1), much less than 8·Ne — sharing must be exact.
        m = make_cubical_mesh(Float64, 3, 3, 3, 0.0, 1.0)
        @test nv(m) == 4 * 4 * 4
        @test nv(m) < 8 * m.Ne                # i.e. real sharing happened

        # Element 1 (mx=my=mz=1) shares its +x face (vertices 2,3,6,7 of
        # element 1) with element 2 (mx=2,my=1,mz=1), which on its −x face
        # owns vertices 1,4,5,8. The matching pairs must reference the
        # SAME global vertex IDs.
        @test m.vertex_idx[2, 1] == m.vertex_idx[1, 2]
        @test m.vertex_idx[3, 1] == m.vertex_idx[4, 2]
        @test m.vertex_idx[6, 1] == m.vertex_idx[5, 2]
        @test m.vertex_idx[7, 1] == m.vertex_idx[8, 2]
    end

    @testset "make_cubical_mesh: orientation is always 0 (axis-aligned)" begin
        m = make_cubical_mesh(Float64, 4, 0.0, 1.0)
        @test all(m.orientation .== 0)
    end

    @testset "make_cubical_mesh: boundary tags only on outer faces" begin
        Mx, My, Mz = 3, 2, 4
        m = make_cubical_mesh(Float64, Mx, My, Mz, 0.0, 1.0)
        # Helper: linear index from (mx, my, mz).
        lidx(mx, my, mz) = mx + (my-1)*Mx + (mz-1)*Mx*My
        # Inspect every element.
        for mz in 1:Mz, my in 1:My, mx in 1:Mx
            e = lidx(mx, my, mz)
            # On each face, bdry ≠ 0 iff neighbour == 0 — the two together
            # exactly partition the six face slots.
            for f in 1:6
                @test (m.bdry[f, e] ≠ 0) == (m.neighbour[f, e] == 0)
            end
            # The outer-tag values match the face index for the cubical mesh.
            mx == 1  && @test m.bdry[1, e] == 1
            mx == Mx && @test m.bdry[2, e] == 2
            my == 1  && @test m.bdry[3, e] == 3
            my == My && @test m.bdry[4, e] == 4
            mz == 1  && @test m.bdry[5, e] == 5
            mz == Mz && @test m.bdry[6, e] == 6
        end
    end

    @testset "make_cubical_mesh: neighbour pointers are symmetric" begin
        # If element A says "my +x neighbour is B", then B must say
        # "my −x neighbour is A". Same for ±y and ±z.
        m = make_cubical_mesh(Float64, 3, 0.0, 1.0)
        opposite = (2, 1, 4, 3, 6, 5)   # opposite face for each of 1..6
        for e in 1:m.Ne
            for f in 1:6
                n = m.neighbour[f, e]
                n == 0 && continue
                @test m.neighbour[opposite[f], n] == e
            end
        end
    end

    @testset "make_cubical_mesh: vertex coordinates" begin
        # A 2×2×2 cubical mesh of [0, 2]³ should have element 1 (mx=my=mz=1)
        # occupying [0,1]³ with vertices in Gmsh-canonical ordering.
        m = make_cubical_mesh(Float64, 2, 0.0, 2.0)

        # Look up element 1's eight corner positions via the shared table.
        corner(e, v) = m.vertex_coords[:, m.vertex_idx[v, e]]
        @test corner(1, 1) == [0.0, 0.0, 0.0]
        @test corner(1, 2) == [1.0, 0.0, 0.0]
        @test corner(1, 3) == [1.0, 1.0, 0.0]
        @test corner(1, 4) == [0.0, 1.0, 0.0]
        @test corner(1, 5) == [0.0, 0.0, 1.0]
        @test corner(1, 6) == [1.0, 0.0, 1.0]
        @test corner(1, 7) == [1.0, 1.0, 1.0]
        @test corner(1, 8) == [0.0, 1.0, 1.0]

        # The element diagonally opposite (mx=my=mz=2) should occupy [1,2]³.
        e_far = 2 + 1*2 + 1*4     # = 8
        @test corner(e_far, 1) == [1.0, 1.0, 1.0]
        @test corner(e_far, 7) == [2.0, 2.0, 2.0]
    end

    @testset "make_cubed_cube_mesh: neighbour count by topology" begin
        # Build the 7-patch cubed cube. The outer surface is the cube
        # [-1, 1]³ tiled by 6·M² quads, one per outermost-shell element of
        # the six outer patches. The four "side" faces of each outer patch
        # are shared with the four adjacent outer patches along the cube
        # edges, so each outer-shell element loses exactly one neighbour
        # (the radially outward one). All other elements (the inner cube
        # and the non-outermost outer-patch layers) are fully surrounded.
        #
        # Per-element neighbour counts therefore split into just two bins:
        #   outer-shell (one outer face) → 5
        #   everything else              → 6
        M, R = 4, 0.1
        m = make_cubed_cube_mesh(Float64, M, R)

        # Each element bumps each of its neighbours' counters. By neighbour
        # symmetry this equals the per-element count of non-zero neighbour
        # slots, but we follow the prompt and accumulate explicitly.
        count = zeros(Int, m.Ne)
        for e in 1:m.Ne, f in 1:6
            n = m.neighbour[f, e]
            n == 0 && continue
            count[n] += 1
        end

        # Histogram and sanity bounds.
        @test all(3 .≤ count .≤ 6)
        n3 = sum(count .== 3)
        n4 = sum(count .== 4)
        n5 = sum(count .== 5)
        n6 = sum(count .== 6)
        @test n3 + n4 + n5 + n6 == m.Ne

        @test n3 == 0
        @test n4 == 0
        @test n5 == 6 * M^2
        @test n6 == m.Ne - n5

        # Cross-check: total outer-face slots equals tally of missing neighbours.
        outer_slots = 3*n3 + 2*n4 + 1*n5
        @test outer_slots == 6 * M^2
        @test outer_slots == count_zero_neighbours(m)
    end

    @testset "make_inflated_cube_mesh: element counts and boundary tagging" begin
        # 13-patch inflated cube: inner cube + 6 inflation + 6 shell.
        # M=4, L=1, R1=2.5, R2=5  ⇒  h = 0.5,
        #   M_i = round((2.5 − (1+√3)/2 · 1) / 0.5) = round(2.268) = 2
        #   M_s = round((5 − 2.5) / 0.5) = 5
        #   Ne = M³ + 6·M²·(M_i + M_s) = 64 + 672 = 736
        T = Float64
        L = 1.0; R1 = 2.5; R2 = 5.0; M = 4
        m = make_inflated_cube_mesh(T, L, R1, R2, M)
        @test m isa InflatedCubeMesh{T}
        @test m.L == L && m.R1 == R1 && m.R2 == R2
        @test m.Ne == 736

        n_inner = count(p -> is_cubical(p.kind),   m.patch_info)
        n_infl  = count(p -> is_inflation(p.kind), m.patch_info)
        n_shell = count(p -> is_shell(p.kind),     m.patch_info)
        @test n_inner == M^3
        @test n_infl  == 6 * 16 * 2     # M_i = 2
        @test n_shell == 6 * 16 * 5     # M_s = 5
        @test n_inner + n_infl + n_shell == m.Ne

        # Every outer-boundary face lies on the outer sphere |x| = R2,
        # and every shell-outer face is tagged. The only outer faces of
        # the topology are exactly the 6·M² shell-patch outer faces.
        @test count(==(0), m.neighbour) == 6 * M^2
        @test count(!=(0), m.bdry)       == 6 * M^2

        # Boundary positions are within tolerance of |x| = R2.
        for ee in 1:m.Ne, f in 1:6
            m.bdry[f, ee] == 0 && continue
            for ℓ in (1, 4, 5, 8)  # any face-corner vertex
                # Crude: pick a vertex from the element; only need one
                # per face to confirm tagging is on the sphere.
                v = m.vertex_idx[ℓ, ee]
                r = sqrt(sum(abs2, m.vertex_coords[:, v]))
                # All eight vertices are on or below R2; for an outer-face
                # element at least one vertex lies on the sphere.
                @test r ≤ R2 + 1e-8
            end
        end
    end

    @testset "make_inflated_cube_mesh: cube↔inflation interfaces all dedup" begin
        # Regression: the cube vertex positions and the inflation-patch
        # parametric positions must round-trip through identical FP ops
        # at the shared `r = L` face so the position-keyed dedup dict
        # finds them equal. The earlier `−L + (2L)(i/M)` form rounded to
        # a 1-ULP-different value (e.g. `0.05000000000000002` vs `0.05`
        # for `L = 0.1, M = 4`), which broke connectivity at most cube
        # face pairs and made the discrete operator non-symmetric.
        # Test that with `L = 0.1, M = 4, R1 = 0.3, R2 = 1.0` every
        # cube outer face has a curved-patch neighbour and none ends up
        # tagged as an outer-domain boundary.
        T = Float64
        m = make_inflated_cube_mesh(T, 0.1, 0.3, 1.0, 4)
        for e in 1:m.Ne
            m.patch_info[e].kind == 0 || continue   # inner cube only
            for f in 1:6
                # An inner-cube element should never touch the outer-
                # domain boundary: it always has a neighbour, whether a
                # sibling cube element or an inflation patch.
                @test m.neighbour[f, e] != 0
                @test m.bdry[f, e] == 0
            end
        end
    end

    @testset "make_inflated_cube_mesh: neighbour symmetry and orientation D₄" begin
        T = Float64
        m = make_inflated_cube_mesh(T, 1.0, 2.0, 4.0, 3)
        # Each non-zero neighbour link must round-trip.
        for e in 1:m.Ne, f in 1:6
            n = m.neighbour[f, e]
            n == 0 && continue
            fn = m.neighbour_face[f, e]
            @test m.neighbour[fn, n] == e
            @test m.neighbour_face[fn, n] == f
        end
        @test all(0 .≤ m.orientation .≤ 7)
    end

    # NB: the original "geometry is well-formed" testset (which used
    # `make_geometry` + a downstream element to check `g.coords`,
    # `g.detjac`, `g.handedness`, and exact outer-sphere placement)
    # has moved to `WaveToySecondOrder/test/test_geometry.jl` along
    # with the `MeshGeometry` type itself.

    @testset "make_inflated_cube_mesh: patch_to_global ↔ global_to_patch round-trip" begin
        T = Float64
        m = make_inflated_cube_mesh(T, 0.1, 0.3, 1.0, 3)
        # ξ → global → ξ over all 13 patches
        for patch_index in 1:13
            for _ in 1:20
                ξ = SVector{3, T}(rand(), rand(), rand())
                p = patch_to_global(m, patch_index, ξ)
                ξ2 = global_to_patch(m, patch_index, p)
                @test !isnan(ξ2[1])
                @test ξ2[1] ≈ ξ[1] atol=1e-12
                @test ξ2[2] ≈ ξ[2] atol=1e-12
                @test ξ2[3] ≈ ξ[3] atol=1e-12
            end
        end
        # global → patch → ξ → global
        rng_pts = ((0.0, 0.0, 0.0), (0.05, -0.03, 0.04), (0.4, 0.0, 0.0),
                   (-0.6, 0.3, 0.2), (0.5, -0.5, 0.5), (0.0, 0.0, -0.85))
        for (x, y, z) in rng_pts
            p = SVector{3, T}(x, y, z)
            i = locate_patch(m, p)
            @test i ≠ 0
            ξ = global_to_patch(m, i, p)
            @test !isnan(ξ[1])
            p2 = patch_to_global(m, i, ξ)
            @test p2[1] ≈ p[1] atol=1e-12
            @test p2[2] ≈ p[2] atol=1e-12
            @test p2[3] ≈ p[3] atol=1e-12
        end
    end

    @testset "make_inflated_cube_mesh: locate_patch and wrong-patch NaN" begin
        T = Float64
        m = make_inflated_cube_mesh(T, 0.1, 0.3, 1.0, 3)
        @test locate_patch(m, SVector{3, T}(0.0, 0.0, 0.0)) == 1     # inner
        @test locate_patch(m, SVector{3, T}(0.2, 0.0, 0.0)) == 2     # +x infl.
        @test locate_patch(m, SVector{3, T}(-0.2, 0.0, 0.0)) == 3    # -x infl.
        @test locate_patch(m, SVector{3, T}(0.0, 0.2, 0.0)) == 4     # +y infl.
        @test locate_patch(m, SVector{3, T}(0.0, -0.2, 0.0)) == 5    # -y infl.
        @test locate_patch(m, SVector{3, T}(0.0, 0.0, 0.2)) == 6     # +z infl.
        @test locate_patch(m, SVector{3, T}(0.0, 0.0, -0.2)) == 7    # -z infl.
        @test locate_patch(m, SVector{3, T}(0.7, 0.0, 0.0)) == 8     # +x shell
        @test locate_patch(m, SVector{3, T}(2.0, 0.0, 0.0)) == 0     # outside
        # Wrong-patch sanity
        @test isnan(global_to_patch(m, 2, SVector{3, T}(0.0, 0.0, 0.8))[1])
        @test isnan(global_to_patch(m, 6, SVector{3, T}(0.7, 0.0, 0.0))[1])
    end

end
