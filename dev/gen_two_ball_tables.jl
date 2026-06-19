#!/usr/bin/env julia
#
# Regenerate `src/two_ball_tables.jl`, the frozen interior connectivity for
# `make_two_ball_mesh` (3D ball with two spherical holes).
#
#     julia --project=. dev/gen_two_ball_tables.jl
#
# Run this after changing the patch build order or topology in
# `_two_ball_patches`. The connectivity is pure topology — independent of the
# geometry, resolution, precision, and outer radius — so this drives the
# build-time `_two_ball_autowire` scaffold (which matches the four physical
# corners of each quad face up to the eight D₄ transforms) over a finite-R2
# reference geometry, asserts the result is geometry-independent, and writes
# the integer tables. The shipped builder then uses the tables with no
# floating-point comparison.

using HexMeshes
const H = HexMeshes

const OUT = normpath(joinpath(@__DIR__, "..", "src", "two_ball_tables.jl"))

genlinks(mode; R1, R2, d, M, kw...) =
    H._two_ball_autowire(H._two_ball_patches(Float64, R1, R2, d, M; mode, kw...)[1:2]...,
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

# Reference geometry (finite R2; the tables are R2-independent) and a very
# different one used only to assert geometry-independence.
sep  = genlinks(:separated; R1 = 1.0, R2 = 100.0, d = 10.0, M = 3)
tou  = genlinks(:touching;  R1 = 1.0, R2 = 100.0, d = 4.0,  M = 3)
sep2 = genlinks(:separated; R1 = 0.3, R2 = 60.0, d = 16.0, M = 5, A = 12.0, R_mid = 24.0)
tou2 = genlinks(:touching;  R1 = 0.5, R2 = 80.0, d = 7.0,  M = 5, A = 16.0, R_mid = 30.0)
@assert Set(sep) == Set(sep2) "separated connectivity is geometry-dependent!"
@assert Set(tou) == Set(tou2) "touching connectivity is geometry-dependent!"

open(OUT, "w") do io
    print(io, """
# Frozen interior connectivity for `make_two_ball_mesh` — GENERATED, do not edit
# by hand. Regenerate with `julia --project=. dev/gen_two_ball_tables.jl`. Each
# entry `(p, f, q, g, o)`: face `f` of patch `p` links to face `g` of patch `q`
# with D₄ orientation `o`. Two directed entries per pair (D₄ is not self-
# inverse). Tied to the patch build order in `_two_ball_patches`. Stored as
# Vectors (not giant tuples) to keep load/compile times sane.

const _TWO_BALL_LINKS_SEPARATED = NTuple{5, Int}[
    $(fmt(sep))]

const _TWO_BALL_LINKS_TOUCHING = NTuple{5, Int}[
    $(fmt(tou))]
""")
end
println("wrote $OUT: separated = $(length(sep)), touching = $(length(tou))")
