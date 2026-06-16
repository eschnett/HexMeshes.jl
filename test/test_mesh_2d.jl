# Tests for the 2D `QuadMesh` family — uniform `make_uniform_quad` only.
# Curvilinear families (`make_cubed_square_mesh`,
# `make_inflated_square_mesh`) go in their own file once those builders
# land.

using HexMeshes
using HexMeshes: Mesh, PatchKind, Cubic, Wedge, Inflation, Shell, BilinearQuad,
                 PatchDesc, dims,
                 make_uniform_quad, make_cubed_square_mesh,
                 make_inflated_square_mesh, make_annulus_mesh, make_two_hole_mesh,
                 nv, npatches,
                 element_vertices, element_point_and_jac, locate_point, invert_element_map,
                 interpolate_field, patch_to_global, global_to_patch,
                 bilinear_shape, bilinear_dshape,
                 bilinear_map, bilinear_jacobian,
                 _neigh_p
using LinearAlgebra: det
using StaticArrays
using Test

@testset "mesh (2D)" begin

    @testset "make_uniform_quad(T, Mx, My, …): shapes" begin
        m = make_uniform_quad(Float64, 3, 4, 0.0, 1.0)
        @test m isa Mesh{2, Float64}
        @test m.Ne == 3 * 4
        @test size(m.conn.neighbour)      == (4, 12)
        @test size(m.conn.neighbour_face) == (4, 12)
        @test size(m.conn.orientation)    == (4, 12)
        @test size(m.conn.bdry)           == (4, 12)
        @test size(m.vertex_coords)  == (2, 4 * 5)
        @test size(m.vertex_idx)     == (4, 12)
        @test nv(m) == 20
    end

    @testset "Square shorthand: equal element count in both axes" begin
        m = make_uniform_quad(Float64, 3, 0.0, 1.0)
        @test m.Ne == 9
        @test nv(m) == 16
    end

    @testset "shared vertices: adjacent elements reuse the same vertex ID" begin
        Mx, My = 3, 3
        m = make_uniform_quad(Float64, Mx, My, 0.0, 1.0)
        @test nv(m) == (Mx + 1) * (My + 1)
        @test nv(m) < 4 * m.Ne                     # sharing happened

        # Element 1 (mx=my=1) shares its +x face (vertices 2, 3) with
        # element 2 (mx=2, my=1) which on its −x face owns vertices 1, 4.
        @test m.vertex_idx[2, 1] == m.vertex_idx[1, 2]
        @test m.vertex_idx[3, 1] == m.vertex_idx[4, 2]
        # Element 1 shares its +y face (vertices 4, 3) with element
        # 1 + Mx = 4 (mx=1, my=2) which on its −y face owns vertices 1, 2.
        @test m.vertex_idx[4, 1] == m.vertex_idx[1, 1 + Mx]
        @test m.vertex_idx[3, 1] == m.vertex_idx[2, 1 + Mx]
    end

    @testset "orientation is always 0 (axis-aligned)" begin
        m = make_uniform_quad(Float64, 4, 0.0, 1.0)
        @test all(m.conn.orientation .== 0)
    end

    @testset "boundary tags only on outer faces" begin
        Mx, My = 3, 2
        m = make_uniform_quad(Float64, Mx, My, 0.0, 1.0)
        lidx(mx, my) = mx + (my - 1) * Mx
        for my in 1:My, mx in 1:Mx
            e = lidx(mx, my)
            for f in 1:4
                @test (m.conn.bdry[f, e] ≠ 0) == (m.conn.neighbour[f, e] == 0)
            end
            mx == 1  && @test m.conn.bdry[1, e] == 1
            mx == Mx && @test m.conn.bdry[2, e] == 2
            my == 1  && @test m.conn.bdry[3, e] == 3
            my == My && @test m.conn.bdry[4, e] == 4
        end
    end

    @testset "neighbour pointers are symmetric" begin
        m = make_uniform_quad(Float64, 4, 0.0, 1.0)
        opposite = (2, 1, 4, 3)
        for e in 1:m.Ne, f in 1:4
            n = m.conn.neighbour[f, e]
            n == 0 && continue
            @test m.conn.neighbour[opposite[f], n] == e
            @test m.conn.neighbour_face[f, e]      == opposite[f]
        end
    end

    @testset "vertex coordinates: regular grid on [x0, x1]²" begin
        Mx, My = 2, 3
        m = make_uniform_quad(Float64, Mx, My, -1.0, 2.0)
        hx = (2.0 - -1.0) / Mx
        hy = (2.0 - -1.0) / My
        # Element (1, 1): vertex 1 is at (-1, -1); element (Mx, My):
        # vertex 3 is at (2, 2).
        @test m.vertex_coords[:, m.vertex_idx[1, 1]] == [-1.0, -1.0]
        @test m.vertex_coords[:, m.vertex_idx[3, Mx * My]] == [2.0, 2.0]
        # Element (mx, my)'s vertex 1 is at (x0 + (mx-1) hx, x0 + (my-1) hy).
        for my in 1:My, mx in 1:Mx
            e = mx + (my - 1) * Mx
            v1 = m.vertex_coords[:, m.vertex_idx[1, e]]
            @test v1[1] ≈ -1.0 + (mx - 1) * hx
            @test v1[2] ≈ -1.0 + (my - 1) * hy
        end
    end

    @testset "bilinear shape functions: partition of unity & polynomial exactness" begin
        for T in (Float64, Float32)
            for (ξ_raw, η_raw) in ((0.0, 0.0), (0.25, 0.75), (0.5, 0.5), (1.0, 1.0))
                ξ, η = T(ξ_raw), T(η_raw)
                N = bilinear_shape(ξ, η)
                @test sum(N) ≈ one(T)
                # Linear-exactness in physical coords for axis-aligned
                # unit-square verts.
                verts = (SVector{2, T}(zero(T), zero(T)),
                         SVector{2, T}(one(T),  zero(T)),
                         SVector{2, T}(one(T),  one(T)),
                         SVector{2, T}(zero(T), one(T)))
                p = bilinear_map(verts, ξ, η)
                @test p[1] ≈ ξ
                @test p[2] ≈ η
            end
        end
    end

    @testset "bilinear_jacobian: 2×2 identity on the unit square" begin
        verts = (SVector{2, Float64}(0.0, 0.0),
                 SVector{2, Float64}(1.0, 0.0),
                 SVector{2, Float64}(1.0, 1.0),
                 SVector{2, Float64}(0.0, 1.0))
        for (ξ, η) in ((0.3, 0.7), (0.1, 0.9))
            J = bilinear_jacobian(verts, ξ, η)
            @test J isa SMatrix{2, 2, Float64, 4}
            @test J ≈ SMatrix{2, 2, Float64}(1, 0, 0, 1)
        end
    end

    @testset "invert_element_map (deprecated): round-trips bilinear_map" begin
        # The 2D `invert_element_map` is deprecated. This testset still
        # exercises it as a smoke test; the depwarn fires once per
        # session (Base-level deduplication on the call id).
        verts = (SVector{2, Float64}(0.0, 0.0),
                 SVector{2, Float64}(2.0, 0.1),
                 SVector{2, Float64}(2.3, 1.4),
                 SVector{2, Float64}(0.2, 1.2))
        for (ξ_raw, η_raw) in ((0.3, 0.4), (0.1, 0.9), (0.6, 0.6))
            ξ_in = SVector{2, Float64}(ξ_raw, η_raw)
            p = bilinear_map(verts, ξ_in[1], ξ_in[2])
            ξ_out, ok = invert_element_map(verts, p)
            @test ok
            @test ξ_out ≈ ξ_in atol=1e-10
        end
    end

    @testset "locate_point: returns the element containing the point" begin
        m = make_uniform_quad(Float64, 3, 3, 0.0, 1.0)
        # Element (mx, my) covers ([(mx-1)/3, mx/3], [(my-1)/3, my/3]).
        @test locate_point(m, SVector{2, Float64}(0.1, 0.1))[1] == 1
        @test locate_point(m, SVector{2, Float64}(0.5, 0.5))[1] == 5  # (2,2) = 5
        @test locate_point(m, SVector{2, Float64}(0.9, 0.9))[1] == 9
        # Out of domain.
        @test locate_point(m, SVector{2, Float64}(2.0, 0.5))[1] == 0
    end

    @testset "interpolate_field: bilinear field is recovered exactly" begin
        m = make_uniform_quad(Float64, 3, 3, 0.0, 1.0)
        N = 3
        xs = [0.0, 0.5, 1.0]
        # u(x, y) = 1 + 2x + 3y + 4xy (bilinear, exactly representable).
        u = Array{Float64, 3}(undef, N, N, m.Ne)
        for e in 1:m.Ne
            verts = element_vertices(m, e)
            for j in 1:N, i in 1:N
                p = bilinear_map(verts, xs[i], xs[j])
                u[i, j, e] = 1 + 2 * p[1] + 3 * p[2] + 4 * p[1] * p[2]
            end
        end
        for (x, y) in ((0.13, 0.27), (0.51, 0.4), (0.78, 0.92))
            val = interpolate_field(m, xs, u, SVector{2, Float64}(x, y))
            @test val ≈ 1 + 2*x + 3*y + 4*x*y atol=1e-12
        end
    end

    @testset "_neigh_p: D₁ orientation transform" begin
        # Identity case: o = 0
        for N in (3, 5, 8), p in 1:N
            @test _neigh_p(Int8(0), p, N) == p
        end
        # Reversal: o = 1 → N + 1 − p
        for N in (3, 5, 8)
            @test _neigh_p(Int8(1), 1, N) == N
            @test _neigh_p(Int8(1), N, N) == 1
            # involutive: reversing twice is identity.
            for p in 1:N
                @test _neigh_p(Int8(1), _neigh_p(Int8(1), p, N), N) == p
            end
        end
    end

    # ── Cubed square ──────────────────────────────────────────────────

    @testset "make_cubed_square_mesh: shapes + element count" begin
        for (M, R) in ((3, 0.3), (4, 0.5), (5, 0.1))
            m = make_cubed_square_mesh(Float64, M, R)
            @test m isa Mesh{2, Float64}
            # `L = round(log(1/R) / log(1 + 2/M))`, clipped to ≥ 1.
            L = max(1, round(Int, log(1/R) / log(1 + 2/M)))
            expected_Ne = M^2 + 4 * L * M
            @test m.Ne == expected_Ne
            @test size(m.conn.neighbour) == (4, expected_Ne)
        end
    end

    @testset "make_cubed_square_mesh: outer boundary on [-1, 1]²" begin
        m = make_cubed_square_mesh(Float64, 4, 0.3)
        # Every vertex lies in [-1, 1]², with at least the outer ring
        # touching ±1 along one axis.
        @test maximum(m.vertex_coords) ≈ 1.0
        @test minimum(m.vertex_coords) ≈ -1.0
        @test all(abs.(m.vertex_coords) .≤ 1.0 + 1e-12)
    end

    @testset "make_cubed_square_mesh: orientation is always 0" begin
        m = make_cubed_square_mesh(Float64, 5, 0.2)
        @test all(m.conn.orientation .== 0)
    end

    @testset "make_cubed_square_mesh: boundary tags only on outer edges" begin
        m = make_cubed_square_mesh(Float64, 4, 0.3)
        # Each outer-edge wedge contributes M cells along the outer
        # square edge → M cells per tag, 4 tags.
        for e in 1:m.Ne, f in 1:4
            @test (m.conn.bdry[f, e] ≠ 0) == (m.conn.neighbour[f, e] == 0)
        end
        for t in 1:4
            @test count(==(t), m.conn.bdry) == 4   # = M = 4
        end
    end

    @testset "make_cubed_square_mesh: neighbour pointers are symmetric" begin
        # For a multi-patch mesh, "my +x neighbour" isn't necessarily
        # accessed via the neighbour's −x face — at cross-patch interfaces
        # the local axes can rotate. Use `neighbour_face` to follow the
        # symmetry: if `n = neighbour[f, e]` then `neighbour[neighbour_face[f, e], n] = e`.
        m = make_cubed_square_mesh(Float64, 5, 0.25)
        for e in 1:m.Ne, f in 1:4
            n = m.conn.neighbour[f, e]
            n == 0 && continue
            f_back = m.conn.neighbour_face[f, e]
            @test m.conn.neighbour[f_back, n] == e
        end
    end

    @testset "make_cubed_square_mesh: vertex dedup yields fewer than 4·Ne vertices" begin
        # If no sharing happened we'd have 4·Ne pre-dedup vertices; with
        # full sharing the count is much smaller. Tight bound (no slack):
        # interior cube + 4 wedge patches share their abutting edges
        # vertex-by-vertex, so `Nv` is exactly the count of distinct
        # node positions in the (i, j) lattice.
        for (M, R) in ((3, 0.3), (5, 0.2))
            m = make_cubed_square_mesh(Float64, M, R)
            @test nv(m) < 4 * m.Ne
        end
    end

    @testset "make_cubed_square_mesh: locate_point + interpolate_field" begin
        m = make_cubed_square_mesh(Float64, 4, 0.3)
        # Point at origin → inside the inner-cube patch.
        e_origin, _ = locate_point(m, SVector{2, Float64}(0.0, 0.0))
        @test e_origin ≠ 0
        # Point near outer corner.
        e_corner, _ = locate_point(m, SVector{2, Float64}(0.95, 0.95))
        @test e_corner ≠ 0
        # Out of domain.
        @test locate_point(m, SVector{2, Float64}(2.0, 0.0))[1] == 0
    end

    @testset "make_cubed_square_mesh: patch_to_global ↔ global_to_patch round-trip" begin
        # Covers all four Wedge directions; the dirs 2/3 inverse used
        # to flip the sign of the tangential coordinate b.
        m = make_cubed_square_mesh(Float64, 4, 0.3)
        for patch_index in 1:5
            for _ in 1:25
                ξ = SVector{2, Float64}(rand(), rand())
                p = patch_to_global(m.patch_desc[patch_index], ξ)
                ξ2 = global_to_patch(m.patch_desc[patch_index], p)
                @test !isnan(ξ2[1])
                @test ξ2[1] ≈ ξ[1] atol=1e-12
                @test ξ2[2] ≈ ξ[2] atol=1e-12
            end
        end
    end

    # ── Inflated square ──────────────────────────────────────────────

    @testset "make_inflated_square_mesh: types and shape" begin
        m = make_inflated_square_mesh(Float64, 0.1, 0.3, 1.0, 4)
        @test m isa Mesh{2, Float64}
        @test m.patch_desc[2].inflation.L == 0.1
        @test m.patch_desc[2].inflation.R1 == 0.3
        @test m.patch_desc[6].shell.R2    == 1.0
        @test m.Ne == length(m.patch_id)
        @test size(m.vertex_coords) == (2, nv(m))
        @test size(m.vertex_idx) == (4, m.Ne)
        @test npatches(m) == 9
    end

    @testset "make_inflated_square_mesh: patch kinds cover (Cubic, Inflation, Shell)" begin
        m = make_inflated_square_mesh(Float64, 0.1, 0.3, 1.0, 4)
        per_patch_kinds = [pd.kind for pd in m.patch_desc]
        @test per_patch_kinds == [Cubic;
                                   fill(Inflation, 4);
                                   fill(Shell, 4)]
    end

    @testset "make_inflated_square_mesh: outer boundary on the circle |x| = R2" begin
        m = make_inflated_square_mesh(Float64, 0.1, 0.3, 1.0, 4)
        R2 = m.patch_desc[6].shell.R2
        # Every vertex inside disk of radius R2.
        maxr = maximum(sqrt(m.vertex_coords[1, v]^2 + m.vertex_coords[2, v]^2)
                       for v in 1:nv(m))
        @test maxr ≤ R2 + 1e-12
        # All outer-boundary nodes on r = R2 exactly (the bilinear-patch
        # corners — analytic-Jacobian downstream geometry maps the
        # interior nodes too).
        for e in 1:m.Ne, f in 1:4
            m.conn.neighbour[f, e] == 0 || continue
            m.conn.bdry[f, e] == 1 || continue
            # Face f's vertex set on a quad: face f=2 (+x) → corners 2,3;
            # face f=4 (+y) → corners 4,3; etc.
            corner_pairs = ((1, 4), (2, 3), (1, 2), (4, 3))
            for v_local in corner_pairs[f]
                v_global = m.vertex_idx[v_local, e]
                x = m.vertex_coords[1, v_global]
                y = m.vertex_coords[2, v_global]
                @test sqrt(x^2 + y^2) ≈ R2 atol=1e-12
            end
        end
    end

    @testset "make_inflated_square_mesh: outer_bc = :sommerfeld" begin
        m = make_inflated_square_mesh(Float64, 0.1, 0.3, 1.0, 4;
                                       outer_bc = :sommerfeld)
        # Every outer face on the outer circle is tagged 7 (and only those).
        @test count(==(Int8(7)), m.conn.bdry) == 4 * 4   # M faces × 4 directions
        @test count(==(Int8(1)), m.conn.bdry) == 0       # no Dirichlet tag
    end

    @testset "make_inflated_square_mesh: neighbour pointers are symmetric" begin
        m = make_inflated_square_mesh(Float64, 0.1, 0.3, 1.0, 4)
        for e in 1:m.Ne, f in 1:4
            n = m.conn.neighbour[f, e]
            n == 0 && continue
            f_back = m.conn.neighbour_face[f, e]
            @test m.conn.neighbour[f_back, n] == e
        end
    end

    @testset "make_inflated_square_mesh: _patch_point_and_jac_2d well-formed" begin
        m = make_inflated_square_mesh(Float64, 0.1, 0.3, 1.0, 4)
        # Iterate over a few curved-patch elements and sample the
        # analytic Jacobian; det J should be positive everywhere
        # (right-handed local frames by design).
        sample_xi = (0.25, 0.5, 0.75)
        for e in 1:m.Ne
            pd = m.patch_desc[m.patch_id[e]]
            (pd.kind === Inflation || pd.kind === Shell) || continue
            idx = ntuple(d -> Int(m.patch_idx[d, e]), Val(2))
            for η in sample_xi, ξ in sample_xi
                P, J = HexMeshes._patch_point_and_jac_2d(pd, idx, ξ, η)
                @test all(isfinite, P)
                @test all(isfinite, J)
                @test det(J) > 0
            end
        end
    end

    @testset "make_inflated_square_mesh: locate_point analytic" begin
        m = make_inflated_square_mesh(Float64, 0.1, 0.3, 1.0, 4)
        # Origin → inside inner square.
        e0, _ = locate_point(m, SVector{2, Float64}(0.0, 0.0))
        @test e0 ≠ 0
        # Out of domain.
        @test locate_point(m, SVector{2, Float64}(2.0, 0.0))[1] == 0
    end

    @testset "make_inflated_square_mesh: patch_to_global ↔ global_to_patch round-trip" begin
        m = make_inflated_square_mesh(Float64, 0.1, 0.3, 1.0, 4)
        # ξ → global → ξ for each of the 9 patches
        for patch_index in 1:9
            for _ in 1:25
                ξ = SVector{2, Float64}(rand(), rand())
                p = patch_to_global(m.patch_desc[patch_index], ξ)
                ξ2 = global_to_patch(m.patch_desc[patch_index], p)
                @test !isnan(ξ2[1])
                @test ξ2[1] ≈ ξ[1] atol=1e-12
                @test ξ2[2] ≈ ξ[2] atol=1e-12
            end
        end
        # global → patch → ξ → global
        rng_pts = [(0.0, 0.0), (0.05, -0.07), (0.4, 0.0), (-0.6, 0.3),
                   (0.7, -0.7), (0.0, -0.85)]
        for (x, y) in rng_pts
            p = SVector{2, Float64}(x, y)
            i = locate_patch(m, p)
            @test i ≠ 0
            ξ = global_to_patch(m.patch_desc[i], p)
            @test !isnan(ξ[1])
            p2 = patch_to_global(m.patch_desc[i], ξ)
            @test p2[1] ≈ p[1] atol=1e-12
            @test p2[2] ≈ p[2] atol=1e-12
        end
    end

    @testset "make_inflated_square_mesh: global_to_patch returns NaN on wrong-patch query" begin
        m = make_inflated_square_mesh(Float64, 0.1, 0.3, 1.0, 4)
        # Point in -y inflation, asked as +x inflation → outside, NaN
        @test isnan(global_to_patch(m.patch_desc[2], SVector(0.0, -0.5))[1])
        # Point in +x inflation, asked as +y inflation → outside, NaN
        @test isnan(global_to_patch(m.patch_desc[4], SVector(0.5, 0.0))[1])
        # Inner-square point, asked as a shell patch → outside, NaN
        @test isnan(global_to_patch(m.patch_desc[6], SVector(0.0, 0.0))[1])
    end

    @testset "make_inflated_square_mesh: locate_patch returns just the index" begin
        m = make_inflated_square_mesh(Float64, 0.1, 0.3, 1.0, 4)
        @test locate_patch(m, SVector(0.0, 0.0)) == 1            # inner
        @test locate_patch(m, SVector(0.2, 0.0)) == 2            # +x inflation
        @test locate_patch(m, SVector(-0.2, 0.0)) == 3           # -x inflation
        @test locate_patch(m, SVector(0.0, 0.2)) == 4            # +y inflation
        @test locate_patch(m, SVector(0.0, -0.2)) == 5           # -y inflation
        @test locate_patch(m, SVector(0.7, 0.0)) == 6            # +x shell
        @test locate_patch(m, SVector(2.0, 0.0)) == 0            # outside
    end

    @testset "element_point_and_jac (2D): corners + analytic Jacobian" begin
        # Cubic (bilinear) path on the uniform quad.
        m = make_uniform_quad(Float64, 3, 3, 0.0, 1.0)
        for e in (1, 5, m.Ne)
            vs = element_vertices(m, e)
            for (c, s) in ((1, (0.0, 0.0)), (2, (1.0, 0.0)), (3, (1.0, 1.0)), (4, (0.0, 1.0)))
                P, J = element_point_and_jac(m, e, SVector(s...))
                @test P ≈ vs[c]
                @test J ≈ bilinear_jacobian(vs, s[1], s[2])
            end
        end
        # Curvilinear (Shell) path on the annulus: corners exact + analytic
        # Jacobian matches central finite differences.
        ma = make_annulus_mesh(Float64, 1.0, 2.0, 3)
        h = 1.0e-6
        for e in (1, 7, ma.Ne)
            vs = element_vertices(ma, e)
            for (c, s) in ((1, (0.0, 0.0)), (2, (1.0, 0.0)), (3, (1.0, 1.0)), (4, (0.0, 1.0)))
                P, _ = element_point_and_jac(ma, e, SVector(s...))
                @test maximum(abs.(P .- vs[c])) ≤ 1.0e-12
            end
            ξ = SVector(0.37, 0.61)
            _, J = element_point_and_jac(ma, e, ξ)
            @test det(J) > 0
            for a in 1:2
                eav = a == 1 ? SVector(h, 0.0) : SVector(0.0, h)
                Pp, _ = element_point_and_jac(ma, e, ξ + eav)
                Pm, _ = element_point_and_jac(ma, e, ξ - eav)
                @test maximum(abs.((Pp .- Pm) ./ (2h) .- J[:, a])) < 1.0e-5
            end
        end
    end

    @testset "make_two_hole_mesh: shape, kinds, conformity, det J" begin
        R1 = 1.0; R2 = 20.0; d = 10.0
        m = make_two_hole_mesh(Float64, R1, R2, d, 3;
                               A = 8.0, R_mid = 13.0,
                               M_h = 2, M_b = 2, M_i = 2, M_s = 2)
        @test m isa Mesh{2, Float64}
        @test npatches(m) == 28
        @test m.Ne == length(m.patch_id)
        @test size(m.vertex_coords) == (2, nv(m))
        @test size(m.vertex_idx) == (4, m.Ne)

        # Patch-kind layout: 8 hole inflation, 8 butterfly, 6 outer
        # inflation, 6 shell.
        kinds = [pd.kind for pd in m.patch_desc]
        @test kinds == [fill(Inflation, 8); fill(BilinearQuad, 8);
                        fill(Inflation, 6); fill(Shell, 6)]

        # Vertex dedup actually merged shared faces.
        @test nv(m) < 4 * m.Ne

        # Neighbour pointers symmetric.
        for e in 1:m.Ne, f in 1:4
            n = m.conn.neighbour[f, e]
            n == 0 && continue
            fb = m.conn.neighbour_face[f, e]
            @test m.conn.neighbour[fb, n] == e
        end

        # Conformity: each patch's analytic map reproduces the deduped
        # vertex coordinates exactly. This is the load-bearing check on
        # the auto-wired connectivity — `_skeleton_to_mesh` writes one
        # representative coordinate per vertex with no consistency check,
        # so a wrong interface orientation would silently disagree here.
        maxcoord = 0.0
        for e in 1:m.Ne
            pd = m.patch_desc[m.patch_id[e]]
            dd = dims(pd)
            cidx = (Int(m.patch_idx[1, e]), Int(m.patch_idx[2, e]))
            for (c, off) in enumerate(((0, 0), (1, 0), (1, 1), (0, 1)))
                ξ = SVector{2, Float64}((cidx[1] - 1 + off[1]) / dd[1],
                                        (cidx[2] - 1 + off[2]) / dd[2])
                P = patch_to_global(pd, ξ)
                vid = m.vertex_idx[c, e]
                maxcoord = max(maxcoord,
                               abs(P[1] - m.vertex_coords[1, vid]),
                               abs(P[2] - m.vertex_coords[2, vid]))
            end
        end
        @test maxcoord ≤ 1.0e-12

        # Every element positively oriented (det J > 0) — validates the
        # reversed-radial hole inflation and the CCW butterfly winding.
        for e in 1:m.Ne
            for ξ in (SVector(0.25, 0.25), SVector(0.75, 0.25),
                      SVector(0.25, 0.75), SVector(0.75, 0.75),
                      SVector(0.5, 0.5))
                _, J = element_point_and_jac(m, e, ξ)
                @test det(J) > 0
            end
        end
    end

    @testset "make_two_hole_mesh: hole circles, outer circle, boundary tags" begin
        R1 = 1.0; R2 = 20.0; d = 10.0; s = d / 2
        m = make_two_hole_mesh(Float64, R1, R2, d, 3;
                               A = 8.0, R_mid = 13.0,
                               M_h = 2, M_b = 2, M_i = 2, M_s = 2)
        corner_pairs = ((1, 4), (2, 3), (1, 2), (4, 3))  # local corners per face
        n_inner = 0; n_outer = 0; maxr = 0.0
        for e in 1:m.Ne, f in 1:4
            m.conn.neighbour[f, e] == 0 || continue
            tag = m.conn.bdry[f, e]
            for vl in corner_pairs[f]
                vid = m.vertex_idx[vl, e]
                x = m.vertex_coords[1, vid]; y = m.vertex_coords[2, vid]
                maxr = max(maxr, hypot(x, y))
                if tag == 8           # excision: on one of the two hole circles
                    @test min(hypot(x - s, y), hypot(x + s, y)) ≈ R1 atol = 1.0e-10
                elseif tag == 1       # outer: on |x| = R2
                    @test hypot(x, y) ≈ R2 atol = 1.0e-9
                end
            end
            tag == 8 && (n_inner += 1)
            tag == 1 && (n_outer += 1)
        end
        @test n_inner > 0
        @test n_outer > 0
        @test maxr ≤ R2 + 1.0e-9
        # Only the hole-circle and outer-circle faces carry a boundary tag.
        for e in 1:m.Ne, f in 1:4
            @test (m.conn.bdry[f, e] ≠ 0) == (m.conn.neighbour[f, e] == 0)
        end
    end

    @testset "make_two_hole_mesh: round-trip + bc variants + too-close error" begin
        m = make_two_hole_mesh(Float64, 1.0, 20.0, 10.0, 3;
                               A = 8.0, R_mid = 13.0,
                               M_h = 2, M_b = 2, M_i = 2, M_s = 2)
        # patch_to_global ↔ global_to_patch over all 28 patches (covers the
        # BilinearQuad Newton inverse and the off-centre inflation inverse).
        for p in 1:npatches(m)
            pd = m.patch_desc[p]
            for _ in 1:20
                ξ = SVector{2, Float64}(rand(), rand())
                x = patch_to_global(pd, ξ)
                ξ2 = global_to_patch(pd, x)
                @test !isnan(ξ2[1])
                @test ξ2[1] ≈ ξ[1] atol = 1.0e-10
                @test ξ2[2] ≈ ξ[2] atol = 1.0e-10
            end
        end
        # outer Sommerfeld → tag 7 on the outer circle; inner excision → 8.
        ms = make_two_hole_mesh(Float64, 1.0, 20.0, 10.0, 3;
                                A = 8.0, R_mid = 13.0, M_h = 2, M_b = 2,
                                M_i = 2, M_s = 2, outer_bc = :sommerfeld)
        @test any(==(Int8(7)), ms.conn.bdry)
        @test any(==(Int8(8)), ms.conn.bdry)
        # Holes too close for the butterfly (L ≥ d/2) must error.
        @test_throws AssertionError make_two_hole_mesh(Float64, 1.0, 20.0, 3.0, 3;
                                                       L = 2.0, A = 8.0, R_mid = 13.0)
    end

end
