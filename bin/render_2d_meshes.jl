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

    println("wrote 3 PNGs to ", out_dir)
    return out_dir
end
