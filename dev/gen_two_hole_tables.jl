#!/usr/bin/env julia
#
# Regenerate `src/two_hole_tables.jl`, the frozen interior connectivity for
# `make_two_hole_mesh` (2D disk with two circular holes).
#
#     julia --project=. dev/gen_two_hole_tables.jl
#
# Run this after changing the patch build order or topology in
# `_two_hole_patches`. The connectivity is pure topology — independent of the
# geometry, resolution, precision, and outer radius — so this drives the
# build-time `_two_hole_autowire` scaffold (which matches the two physical
# endpoints of each face) over a finite-R2 reference geometry, asserts the
# result is geometry-independent, and writes the integer tables. The shipped
# builder then uses the tables with no floating-point comparison.
#
# D₁ orientations are self-inverse, so the table holds ONE entry per interior
# face pair and the builder sets both half-links with the same `o`.

using HexMeshes
const H = HexMeshes

const OUT = normpath(joinpath(@__DIR__, "..", "src", "two_hole_tables.jl"))

genlinks(mode; R1, R2, d, M, kw...) =
    H._two_hole_autowire(H._two_hole_patches(Float64, R1, R2, d, M; mode, kw...)[1:2]...,
                         1.0e-9 * R2)

function fmt(links)
    io = IOBuffer()
    for (i, t) in enumerate(links)
        print(io, "($(t[1]),$(t[2]),$(t[3]),$(t[4]),$(t[5]))")
        i < length(links) && print(io, ",")
        print(io, i % 8 == 0 ? "\n    " : " ")
    end
    return rstrip(String(take!(io)))
end

sep  = genlinks(:separated; R1 = 1.0, R2 = 100.0, d = 10.0, M = 3)
tou  = genlinks(:touching;  R1 = 1.0, R2 = 100.0, d = 4.0,  M = 3)
sep2 = genlinks(:separated; R1 = 0.3, R2 = 60.0, d = 18.0, M = 5, A = 11.0, R_mid = 20.0)
tou2 = genlinks(:touching;  R1 = 0.5, R2 = 80.0, d = 6.0,  M = 5, A = 10.0, R_mid = 16.0)
@assert Set(sep) == Set(sep2) "separated connectivity is geometry-dependent!"
@assert Set(tou) == Set(tou2) "touching connectivity is geometry-dependent!"

open(OUT, "w") do io
    print(io, """
# Frozen interior connectivity for `make_two_hole_mesh` — GENERATED, do not edit
# by hand. Regenerate with `julia --project=. dev/gen_two_hole_tables.jl`. Each
# entry `(p, f, q, g, o)`: face `f` of patch `p` links to face `g` of patch `q`
# with D₁ orientation `o`; the builder sets both half-links (D₁ is self-inverse).
# One entry per interior face pair. Tied to the build order in `_two_hole_patches`.

const _TWO_HOLE_LINKS_SEPARATED = NTuple{5, Int}[
    $(fmt(sep))]

const _TWO_HOLE_LINKS_TOUCHING = NTuple{5, Int}[
    $(fmt(tou))]
""")
end
println("wrote $OUT: separated = $(length(sep)), touching = $(length(tou))")
