# Render PNG figures of the three 2D mesh families for the README.
#
# Run from the repository root:
#
#     julia --project=@. --eval 'include("bin/render_2d_meshes.jl"); render()'
#
# Outputs `docs/src/figures/mesh_*.png` (one per mesh family).
#
# Each element is drawn as a filled quad with its outline; the fill
# colour identifies the patch the element belongs to, so the
# 5-patch `cubed_square` and 9-patch `inflated_square` decompositions
# are visible at a glance. The trivial `uniform_quad` has one patch
# and so renders in a single colour.

using CairoMakie
using HexMeshes

# Discrete colormap with enough distinct entries for the 9-patch
# inflated_square mesh. `:tab10` (matplotlib's default categorical
# palette) gives 10 high-contrast colours.
const PATCH_COLORS = Makie.to_colormap(:tab10)

# Draw `mesh` into `ax`. Each element is filled with a colour keyed by
# its `patch_id`, and outlined with a thin black stroke so the
# element-level subdivision stays visible inside each patch.
function plot_mesh!(ax, mesh::HexMeshes.Mesh{2, T}) where {T}
    vc = mesh.vertex_coords
    vi = mesh.vertex_idx
    pid = mesh.patch_id
    for e in 1:mesh.Ne
        # Gmsh-canonical winding: (−x,−y), (+x,−y), (+x,+y), (−x,+y).
        pts = [Point2f(vc[1, vi[k, e]], vc[2, vi[k, e]]) for k in 1:4]
        c   = PATCH_COLORS[mod1(pid[e], length(PATCH_COLORS))]
        poly!(ax, pts; color = c, strokecolor = (:black, 0.7),
                       strokewidth = 0.6)
    end
    return ax
end

# Draw the z = 0 cross-section of a 3D mesh: every element face whose four
# vertices lie on the plane z = 0 (|z| < tol), projected to (x, y) and coloured
# by `patch_id`. A z = 0 face layer exists when the z-resolution is even (pass
# an even `M`). Faces shared across z = 0 are drawn from both sides (harmless).
function plot_slice_z0!(ax, mesh::HexMeshes.Mesh{3, T}; tol = 1.0e-9) where {T}
    vc = mesh.vertex_coords
    vi = mesh.vertex_idx
    pid = mesh.patch_id
    # Gmsh hex faces → 4 local corner indices (±x, ±y, ±z).
    faces = ((1, 4, 8, 5), (2, 3, 7, 6), (1, 2, 6, 5),
             (4, 3, 7, 8), (1, 2, 3, 4), (5, 6, 7, 8))
    for e in 1:mesh.Ne, fc in faces
        vs = (vi[fc[1], e], vi[fc[2], e], vi[fc[3], e], vi[fc[4], e])
        all(v -> abs(vc[3, v]) < tol, vs) || continue
        pts = [Point2f(vc[1, v], vc[2, v]) for v in vs]
        c = PATCH_COLORS[mod1(pid[e], length(PATCH_COLORS))]
        poly!(ax, pts; color = c, strokecolor = (:black, 0.7), strokewidth = 0.6)
    end
    return ax
end

function render(out_dir::AbstractString = joinpath(@__DIR__, "..", "docs", "src", "figures");
                  fig_size = (480, 480))
    mkpath(out_dir)
    T = Float64

    # ── uniform_quad ──────────────────────────────────────────────
    # Single Cubic patch on the unit square. Default 8×8 grid is dense
    # enough to read the regular tensor-product structure.
    mesh_uq = make_uniform_quad(T, 8, T(0), T(1))
    fig_uq  = Figure(; size = fig_size)
    ax_uq   = Axis(fig_uq[1, 1];
                   title  = "make_uniform_quad (M = 8)",
                   xlabel = "x", ylabel = "y",
                   aspect = DataAspect())
    plot_mesh!(ax_uq, mesh_uq)
    save(joinpath(out_dir, "mesh_uniform_quad.png"), fig_uq)

    # ── cubed_square ──────────────────────────────────────────────
    # 5 patches: 1 inner Cubic + 4 inflations interpolating the square's
    # edges out to (±1, ±1). M = 6 gives a moderately dense view.
    mesh_cs = make_cubed_square_mesh(T, 6, T(0.3))
    fig_cs  = Figure(; size = fig_size)
    ax_cs   = Axis(fig_cs[1, 1];
                   title  = "make_cubed_square_mesh (M = 6, R = 0.3)",
                   xlabel = "x", ylabel = "y",
                   aspect = DataAspect())
    plot_mesh!(ax_cs, mesh_cs)
    save(joinpath(out_dir, "mesh_cubed_square.png"), fig_cs)

    # ── inflated_square ───────────────────────────────────────────
    # 9 patches: 1 inner Cubic + 4 Inflation + 4 Shell. The shell ring
    # is annular, so the outer disk |x| ≤ R2 is fully resolved.
    mesh_is = make_inflated_square_mesh(T, T(0.1), T(0.3), T(1.0), 6)
    fig_is  = Figure(; size = fig_size)
    ax_is   = Axis(fig_is[1, 1];
                   title  = "make_inflated_square_mesh (L = 0.1, R1 = 0.3, R2 = 1.0, M = 6)",
                   xlabel = "x", ylabel = "y",
                   aspect = DataAspect())
    plot_mesh!(ax_is, mesh_is)
    save(joinpath(out_dir, "mesh_inflated_square.png"), fig_is)

    # ── two_hole ──────────────────────────────────────────────────
    # 28 patches: a disk |x| ≤ R2 with two circular holes (radius R1)
    # centred at (±d/2, 0). 8 hole-inflation + 8 "butterfly" BilinearQuad
    # + 6 outer-inflation + 6 shell patches. Holes are far apart relative
    # to the hole-square half-side so the seam blocks stay well-shaped.
    mesh_th = make_two_hole_mesh(T, T(1.0), T(16.0), T(8.0), 5;
                                 A = T(6.0), R_mid = T(9.0),
                                 M_h = 2, M_b = 2, M_i = 3, M_s = 5)
    fig_th  = Figure(; size = fig_size)
    ax_th   = Axis(fig_th[1, 1];
                   title  = "make_two_hole_mesh (R1 = 1, R2 = 16, d = 8)",
                   xlabel = "x", ylabel = "y",
                   aspect = DataAspect())
    plot_mesh!(ax_th, mesh_th)
    save(joinpath(out_dir, "mesh_two_hole.png"), fig_th)

    # ── two_hole, :touching mode ──────────────────────────────────
    # Close holes (R1 = 1, d = 4 ⇒ the hole squares meet at x = 0). The
    # two seam blocks are dropped → 26 patches, with valence-6 vertices
    # where the squares and the top/bottom spokes meet on the axis.
    mesh_tt = make_two_hole_mesh(T, T(1.0), T(16.0), T(4.0), 5;
                                 A = T(6.0), R_mid = T(10.0),
                                 M_h = 2, M_b = 2, M_i = 3, M_s = 5,
                                 mode = :touching)
    fig_tt  = Figure(; size = fig_size)
    ax_tt   = Axis(fig_tt[1, 1];
                   title  = "make_two_hole_mesh (R1 = 1, R2 = 16, d = 4, :touching)",
                   xlabel = "x", ylabel = "y",
                   aspect = DataAspect())
    plot_mesh!(ax_tt, mesh_tt)
    save(joinpath(out_dir, "mesh_two_hole_touching.png"), fig_tt)

    # ── two_ball z = 0 slices (3D) ────────────────────────────────
    # Cross-section of the 3D ball-with-two-spherical-holes at z = 0 (even M
    # so a face layer sits on the plane). Shows the two excised disks and the
    # round outer boundary — the 3D analog of the two_hole figures.
    for (mode, dd, tag) in ((:separated, 8.0, "separated"), (:touching, 4.0, "touching"))
        mb = make_two_ball_mesh(T, T(1.0), T(16.0), T(dd), 4;
                                A = T(6.0), R_mid = T(12.0),
                                M_h = 2, M_b = 2, M_i = 2, M_s = 4, mode = mode)
        fig = Figure(; size = fig_size)
        ax  = Axis(fig[1, 1];
                   title  = "make_two_ball_mesh z=0 slice (R1=1, R2=16, d=$(dd), :$(mode))",
                   xlabel = "x", ylabel = "y", aspect = DataAspect())
        plot_slice_z0!(ax, mb)
        save(joinpath(out_dir, "mesh_two_ball_slice_$(tag).png"), fig)
    end

    println("wrote 7 PNGs to ", out_dir)
    return out_dir
end
