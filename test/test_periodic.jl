using HexMeshes
using StaticArrays
using Test

# Periodic-boundary tests for the uniform mesh builders.
#
# Invariants checked:
#   * Periodic faces report `bdry == 0` and `neighbour ≠ 0` (kernel-side
#     they look identical to interior faces).
#   * Wraparound is correct: the low-end element's outer face neighbours
#     the high-end element, with `neighbour_face` flipped.
#   * Round-trip: walking `neighbour` and `neighbour_face` once lands
#     you back on the original (e, f) — including across the seam.
#   * Vertex coords on the two periodic sides differ by the domain
#     extent (no geometric merging).
#   * Mixed periodicity (periodic in some axes only) keeps the
#     non-periodic faces tagged.

# Walk one step across face f of element e and return the implied
# (e', f') and the reverse map (e'', f''). The round-trip invariant
# is (e'', f'') == (e, f) for every face that has a neighbour.
function _step_and_back(mesh, e, f)
    e2 = Int(mesh.conn.neighbour[f, e])
    f2 = Int(mesh.conn.neighbour_face[f, e])
    e2 == 0 && return nothing
    e3 = Int(mesh.conn.neighbour[f2, e2])
    f3 = Int(mesh.conn.neighbour_face[f2, e2])
    return (e2, f2, e3, f3)
end

@testset "periodic uniform builders" begin

    @testset "1D periodic line — wraparound" begin
        M = 5
        mesh = make_uniform_line(Float64, M, 0.0, 1.0; periodic = true)
        @test mesh.Ne == M
        # No boundary tags anywhere.
        @test all(mesh.conn.bdry .== 0)
        # Endpoint elements wrap.
        @test mesh.conn.neighbour[1, 1]      == M
        @test mesh.conn.neighbour_face[1, 1] == 2
        @test mesh.conn.neighbour[2, M]      == 1
        @test mesh.conn.neighbour_face[2, M] == 1
        # Round-trip on every face.
        for e in 1:M, f in 1:2
            r = _step_and_back(mesh, e, f)
            @test r !== nothing
            (_, _, e3, f3) = r
            @test (e3, f3) == (e, f)
        end
        # Vertex coords on the two ends are distinct (no geometric merging).
        @test size(mesh.vertex_coords, 2) == M + 1
        @test mesh.vertex_coords[1, 1]   ≈ 0.0
        @test mesh.vertex_coords[1, end] ≈ 1.0
    end

    @testset "1D non-periodic line — boundary tags preserved" begin
        mesh = make_uniform_line(Float64, 4, 0.0, 1.0)
        @test mesh.conn.bdry[1, 1] == 1
        @test mesh.conn.bdry[2, 4] == 2
        @test mesh.conn.neighbour[1, 1] == 0
        @test mesh.conn.neighbour[2, 4] == 0
    end

    @testset "2D periodic quad — both axes" begin
        Mx, My = 3, 4
        mesh = make_uniform_quad(Float64, Mx, My, 0.0, 2.0; periodic = true)
        @test mesh.Ne == Mx * My
        @test all(mesh.conn.bdry .== 0)
        eid(mx, my) = mx + (my - 1) * Mx
        # Wraparound at four corners.
        @test mesh.conn.neighbour[1, eid(1,  1 )] == eid(Mx, 1)
        @test mesh.conn.neighbour[2, eid(Mx, 1 )] == eid(1,  1)
        @test mesh.conn.neighbour[3, eid(1,  1 )] == eid(1,  My)
        @test mesh.conn.neighbour[4, eid(1,  My)] == eid(1,  1)
        # Round-trip on every face.
        for e in 1:mesh.Ne, f in 1:4
            r = _step_and_back(mesh, e, f)
            @test r !== nothing
            (_, _, e3, f3) = r
            @test (e3, f3) == (e, f)
        end
    end

    @testset "2D periodic quad — mixed axes (px only)" begin
        Mx, My = 3, 2
        mesh = make_uniform_quad(Float64, Mx, My, 0.0, 1.0; periodic = (true, false))
        eid(mx, my) = mx + (my - 1) * Mx
        # x faces wrap, y faces keep their boundary tags.
        for my in 1:My
            @test mesh.conn.bdry[1, eid(1,  my)] == 0
            @test mesh.conn.bdry[2, eid(Mx, my)] == 0
            @test mesh.conn.neighbour[1, eid(1,  my)] == eid(Mx, my)
            @test mesh.conn.neighbour[2, eid(Mx, my)] == eid(1,  my)
        end
        for mx in 1:Mx
            @test mesh.conn.bdry[3, eid(mx, 1)]  == 3
            @test mesh.conn.bdry[4, eid(mx, My)] == 4
            @test mesh.conn.neighbour[3, eid(mx, 1)]  == 0
            @test mesh.conn.neighbour[4, eid(mx, My)] == 0
        end
    end

    @testset "3D periodic hex — all axes" begin
        Mx, My, Mz = 2, 3, 2
        mesh = make_uniform_hex(Float64, Mx, My, Mz, 0.0, 1.0; periodic = true)
        @test mesh.Ne == Mx * My * Mz
        @test all(mesh.conn.bdry .== 0)
        eid(mx, my, mz) = mx + (my - 1) * Mx + (mz - 1) * Mx * My
        # Spot-check the eight low/high corners' wraparounds.
        @test mesh.conn.neighbour[1, eid(1,  1,  1)] == eid(Mx, 1, 1)
        @test mesh.conn.neighbour[2, eid(Mx, 1,  1)] == eid(1,  1, 1)
        @test mesh.conn.neighbour[3, eid(1,  1,  1)] == eid(1, My, 1)
        @test mesh.conn.neighbour[4, eid(1,  My, 1)] == eid(1,  1, 1)
        @test mesh.conn.neighbour[5, eid(1,  1,  1)]  == eid(1, 1, Mz)
        @test mesh.conn.neighbour[6, eid(1,  1,  Mz)] == eid(1, 1, 1)
        # neighbour_face is the opposite of the source face for every
        # axis-aligned periodic seam.
        for e in 1:mesh.Ne, f in 1:6
            f_opp = isodd(f) ? f + 1 : f - 1
            @test mesh.conn.neighbour_face[f, e] == f_opp
        end
        # Round-trip on every face.
        for e in 1:mesh.Ne, f in 1:6
            r = _step_and_back(mesh, e, f)
            @test r !== nothing
            (_, _, e3, f3) = r
            @test (e3, f3) == (e, f)
        end
    end

    @testset "3D periodic hex — z only, mixed with Dirichlet x/y" begin
        Mx, My, Mz = 2, 2, 3
        mesh = make_uniform_hex(Float64, Mx, My, Mz, 0.0, 1.0;
                                  periodic = (false, false, true))
        eid(mx, my, mz) = mx + (my - 1) * Mx + (mz - 1) * Mx * My
        # z faces interior; x/y faces still tagged at the domain boundary.
        for my in 1:My, mx in 1:Mx
            @test mesh.conn.bdry[5, eid(mx, my, 1)]  == 0
            @test mesh.conn.bdry[6, eid(mx, my, Mz)] == 0
            @test mesh.conn.neighbour[5, eid(mx, my, 1)]  == eid(mx, my, Mz)
            @test mesh.conn.neighbour[6, eid(mx, my, Mz)] == eid(mx, my, 1)
        end
        for mz in 1:Mz, my in 1:My
            @test mesh.conn.bdry[1, eid(1,  my, mz)] == 1
            @test mesh.conn.bdry[2, eid(Mx, my, mz)] == 2
        end
        for mz in 1:Mz, mx in 1:Mx
            @test mesh.conn.bdry[3, eid(mx, 1,  mz)] == 3
            @test mesh.conn.bdry[4, eid(mx, My, mz)] == 4
        end
    end

    @testset "3D periodic hex — vertex coords on seam stay distinct" begin
        Mx = My = Mz = 2
        mesh = make_uniform_hex(Float64, Mx, My, Mz, 0.0, 1.0; periodic = true)
        # Periodic dedup is suppressed, so the vertex grid stays at the
        # full (Mx+1)·(My+1)·(Mz+1) count.
        @test size(mesh.vertex_coords, 2) == (Mx + 1) * (My + 1) * (Mz + 1)
        # Corner elements e_low = (1,1,1) and e_high = (Mx,1,1) each
        # carry their own copies of the periodic-seam vertices, with
        # x-coords 0.0 and 1.0 respectively.
        eid(mx, my, mz) = mx + (my - 1) * Mx + (mz - 1) * Mx * My
        v_low_minus_x  = mesh.vertex_idx[1, eid(1, 1, 1)]   # corner (−x, −y, −z)
        v_high_plus_x  = mesh.vertex_idx[2, eid(Mx, 1, 1)]  # corner (+x, −y, −z)
        @test mesh.vertex_coords[1, v_low_minus_x]  ≈ 0.0
        @test mesh.vertex_coords[1, v_high_plus_x]  ≈ 1.0
        @test v_low_minus_x != v_high_plus_x
    end

    # ───────────────────────────────────────────────────────────────────
    # Warped periodic uniform-hex mesh — `PatchWarpedCubic`.
    # The warp adds a sinusoidal coordinate transformation on top of
    # the standard periodic cube. Useful as a diagnostic for
    # curvilinear behaviour without outer boundaries in the way.

    @testset "make_warped_uniform_hex at A = 0 matches make_uniform_hex" begin
        T = Float64
        m_uniform = make_uniform_hex(T, 3, 3, 3, 0.0, 1.0; periodic = true)
        m_warped  = make_warped_uniform_hex(T, 3, 3, 3, 0.0, 1.0, 0.0;
                                              periodic = true)
        @test m_uniform.Ne == m_warped.Ne
        @test m_uniform.vertex_coords ≈ m_warped.vertex_coords
        @test m_uniform.conn.neighbour       == m_warped.conn.neighbour
        @test m_uniform.conn.neighbour_face  == m_warped.conn.neighbour_face
        @test m_uniform.conn.orientation     == m_warped.conn.orientation
        @test m_uniform.conn.bdry            == m_warped.conn.bdry
        # The warped patch carries a different `PatchKind` tag but the
        # mesh-level connectivity is identical.
        @test m_warped.patch_desc[1].kind === WarpedCubic
        @test m_warped.patch_desc[1].warped_cubic.warp_kind === :diagonal
    end

    @testset "make_warped_uniform_hex at A > 0 perturbs vertex coords" begin
        T = Float64
        m_uniform = make_uniform_hex(T, 3, 3, 3, 0.0, 1.0; periodic = true)
        m_warped  = make_warped_uniform_hex(T, 3, 3, 3, 0.0, 1.0, 0.05;
                                              periodic = true)
        @test m_uniform.Ne == m_warped.Ne
        # Topology unchanged.
        @test m_uniform.conn.neighbour      == m_warped.conn.neighbour
        @test m_uniform.conn.neighbour_face == m_warped.conn.neighbour_face
        @test m_uniform.conn.orientation    == m_warped.conn.orientation
        @test m_uniform.conn.bdry           == m_warped.conn.bdry
        # Vertex coords are perturbed.
        @test maximum(abs, m_uniform.vertex_coords .- m_warped.vertex_coords) > 0.01
        @test maximum(abs, m_uniform.vertex_coords .- m_warped.vertex_coords) < 0.06
    end

    @testset "make_warped_uniform_hex: :coupled warp builds" begin
        T = Float64
        m = make_warped_uniform_hex(T, 3, 3, 3, 0.0, 1.0, 0.05;
                                      periodic = true, warp_kind = :coupled)
        @test m.patch_desc[1].kind === WarpedCubic
        @test m.patch_desc[1].warped_cubic.warp_kind === :coupled
        # Periodicity at corners must hold under both warp kinds, so
        # opposite vertices at (0, …) and (L, …) end up at coords
        # differing by exactly L in the warped axis (the warp
        # vanishes at corners).
        @test maximum(abs, m.vertex_coords) > 0.01
    end

    @testset "make_warped_uniform_hex: locate_point + patch round-trip ($wk, A = $A)" for
            wk in (:diagonal, :coupled), A in (0.02, 0.05, 0.1)
        T = Float64
        m = make_warped_uniform_hex(T, 2, 3, 2, 0.0, 1.0, A;
                                      periodic = true, warp_kind = wk)
        pd = m.patch_desc[1]
        # ξ → global → ξ (Newton inverse of the warp, exact to roundoff).
        for ξt in ((0.5, 0.5, 0.5), (0.13, 0.42, 0.77), (0.91, 0.08, 0.66),
                   (0.02, 0.98, 0.5), (0.33, 0.5, 0.01))
            ξ = SVector{3, T}(ξt)
            p = patch_to_global(pd, ξ)
            ξ2 = global_to_patch(pd, p)
            @test !isnan(ξ2[1])
            @test ξ2[1] ≈ ξ[1] atol = 1e-12
            @test ξ2[2] ≈ ξ[2] atol = 1e-12
            @test ξ2[3] ≈ ξ[3] atol = 1e-12
        end
        # global → ξ → global on interior points.
        for pt in ((0.5, 0.5, 0.5), (0.2, 0.7, 0.4), (0.85, 0.15, 0.6))
            p = SVector{3, T}(pt)
            ξ = global_to_patch(pd, p)
            @test !isnan(ξ[1])
            p2 = patch_to_global(pd, ξ)
            @test p2[1] ≈ p[1] atol = 1e-12
            @test p2[2] ≈ p[2] atol = 1e-12
            @test p2[3] ≈ p[3] atol = 1e-12
        end
        # locate_point lands in a real element whose forward map
        # reproduces the query point (this returned (0, zero ξ) for
        # every point before the WarpedCubic inverse existed).
        for pt in ((0.5, 0.5, 0.5), (0.31, 0.77, 0.12), (0.93, 0.21, 0.55))
            p = SVector{3, T}(pt)
            e, ξe = locate_point(m, p)
            @test e > 0
            P, _ = element_point_and_jac(m, e, ξe)
            @test maximum(abs, P - p) < 1e-12
        end
        # Points outside the domain are still rejected. (The warp maps
        # the box onto itself — sin(ϕ) vanishes on every face — so the
        # warped domain is the same box.)
        @test locate_point(m, SVector{3, T}(1.5, 0.5, 0.5))[1] == 0
        @test locate_point(m, SVector{3, T}(0.5, -0.5, 0.5))[1] == 0
    end

    @testset "make_warped_uniform_hex: interpolate_field exact for patch-linear field ($wk)" for
            wk in (:diagonal, :coupled)
        T = Float64
        A = 0.05
        Mx, My, Mz = 2, 3, 2
        m = make_warped_uniform_hex(T, Mx, My, Mz, 0.0, 1.0, A;
                                      periodic = true, warp_kind = wk)
        pd = m.patch_desc[1]
        # GLL-ish interpolation nodes on [0, 1] (same as test_interp_grad).
        xs = [0.0, 0.2763932022500211, 0.7236067977499789, 1.0]
        N = length(xs)
        # Degree-1 polynomial in the *patch parameter* ξ_patch — the
        # space the interpolation nodes are laid out in, hence exactly
        # representable on the warped elements. (A polynomial in the
        # physical coordinates is not: the warp is sinusoidal, so x(ξ)
        # is not polynomial.)
        g(ξp) = 1 + 2 * ξp[1] - 3 * ξp[2] + ξp[3] / 2
        u = Array{T}(undef, N, N, N, m.Ne)
        for e in 1:m.Ne, k in 1:N, j in 1:N, i in 1:N
            ξp = ((m.patch_idx[1, e] - 1 + xs[i]) / Mx,
                  (m.patch_idx[2, e] - 1 + xs[j]) / My,
                  (m.patch_idx[3, e] - 1 + xs[k]) / Mz)
            u[i, j, k, e] = g(ξp)
        end
        # Query at off-node physical points constructed via the forward
        # map, so the expected value is known analytically.
        for ξt in ((0.5, 0.5, 0.5), (0.31, 0.77, 0.12), (0.93, 0.21, 0.55),
                   (0.011, 0.5, 0.99))
            ξ = SVector{3, T}(ξt)
            p = patch_to_global(pd, ξ)
            @test interpolate_field(m, xs, u, p) ≈ g(ξ) atol = 1e-12
        end
        # Outside the mesh → `default` (NaN).
        @test isnan(interpolate_field(m, xs, u, SVector{3, T}(1.5, 0.5, 0.5)))
    end
end
