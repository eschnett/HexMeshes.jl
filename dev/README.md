# dev/ — developer tooling

Scripts here are **not** part of the package; they generate checked-in source.

## Connectivity-table generators

`make_two_hole_mesh` (2D) and `make_two_ball_mesh` (3D) wire their interior
patch faces from **frozen integer tables** (`src/two_hole_tables.jl`,
`src/two_ball_tables.jl`) rather than by comparing floating-point coordinates at
build time. This keeps the builders purely combinatorial, so they assemble
correctly at any precision (`Float32`, `MultiFloat`, …), under arbitrary
distortion, and with a compactified (`R2 = Inf`) outer boundary.

Those tables are **generated**, not hand-written. The generator logic lives in
each builder as a `_*_autowire` scaffold that discovers the connectivity by
matching the physical corners of each shared face (up to the D₁ / D₄
face-orientation group) — the *only* floating-point comparison anywhere in the
construction, and it runs *here*, never in the shipped builder.

Regenerate after changing a builder's patch build order or topology:

```sh
julia --project=. dev/gen_two_hole_tables.jl   # rewrites src/two_hole_tables.jl
julia --project=. dev/gen_two_ball_tables.jl   # rewrites src/two_ball_tables.jl
```

Each script builds the patches over a reference geometry, runs the auto-wire,
**asserts the connectivity is identical on a very different geometry** (proving
it is pure topology), and writes the table file. The generated tables are then
pinned by the package tests — the coordinate-consistency check (every patch's
analytic map must reproduce the deduped vertices) and the `det J > 0` check fail
loudly if a table is wrong or stale. After regenerating, run `Pkg.test()`.
