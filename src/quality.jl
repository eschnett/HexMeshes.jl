# ----------------------------------------------------------------------
# Host-side mesh-quality diagnostics.
#
# Jacobian-based distortion metrics for curvilinear quad/hex meshes,
# computed from the analytic element Jacobian `J = ∂x/∂ξ`
# (`element_point_and_jac`) sampled at a tensor grid of reference points.
# All three metrics below are standard (Knupp's algebraic metrics / the
# Verdict library); the scaled Jacobian is the headline "distortedness".

"""
    mesh_quality(mesh::Mesh{D, T}; nodes = range(0, 1; length = 3)) → NamedTuple

Jacobian-based distortion metrics for a curvilinear quad/hex mesh, evaluated
at the tensor product of `nodes` (1D reference coordinates in `[0, 1]`) inside
every element via [`element_point_and_jac`](@ref). Pass the solver's GLL/Gauss
nodes for operator-relevant numbers; the default `[0, ½, 1]` per axis samples
corners, edge midpoints, and the centre (curvilinear `det J` is non-monotonic,
so corner-only sampling can miss an interior dip).

From the Jacobian `J` at each sample:

* **scaled Jacobian** `det(J) / ∏ᵢ‖J[:,i]‖ ∈ [-1, 1]` — orthogonality and
  non-degeneracy (`1` orthogonal coordinate lines, `0` collapsed, `< 0`
  inverted). Insensitive to anisotropy, so it flags shear/degeneracy without
  penalising intentional radial grading. This is the headline metric.
* **condition number** `κ = ‖J‖_F·‖J⁻¹‖_F / D ∈ [1, ∞)` (Knupp) — shape
  distortion (shear + anisotropy); `1` for an ideal square/cube.
* **aspect ratio** `σ_max/σ_min` (singular values of `J`) — pure anisotropy.

Returns `(; min_scaled_jacobian, max_condition_number, max_aspect_ratio,
n_inverted, per_patch, element_scaled_jacobian)`, where `per_patch[p]` is the
worst case over patch `p` and `element_scaled_jacobian[e]` is the per-element
minimum scaled Jacobian (for histograms / quantiles). Non-finite samples — e.g.
the i⁰ face of a compactified (`R2 = Inf`) shell, whose outer vertices are at
infinity — are skipped.
"""
function mesh_quality(mesh::Mesh{D, T};
                      nodes = range(zero(T), one(T); length = 3)) where {D, T}
    refs    = collect(T, nodes)
    samples = vec(collect(Iterators.product(ntuple(_ -> refs, Val(D))...)))
    np = npatches(mesh)
    elem_sJ = fill(T(Inf), mesh.Ne)        # per-element min scaled Jacobian
    pp_sJ   = fill(T(Inf), np)             # per-patch worst (min) scaled Jacobian
    pp_cnd  = ones(T, np)                  # per-patch worst (max) condition number
    pp_asp  = ones(T, np)                  # per-patch worst (max) aspect ratio
    n_inverted = 0

    for e in 1:mesh.Ne
        p = Int(mesh.patch_id[e])
        inverted = false
        for s in samples
            ξ = SVector{D, T}(s)
            _, J = element_point_and_jac(mesh, e, ξ)
            all(isfinite, J) || continue          # skip i⁰ (compactified) samples
            dJ = det(J)
            cn = one(T)
            for i in 1:D
                cn *= norm(J[:, i])
            end
            sJ = cn > zero(T) ? dJ / cn : zero(T)
            isfinite(sJ) || continue
            sJ < zero(T) && (inverted = true)
            elem_sJ[e] = min(elem_sJ[e], sJ)
            if dJ > zero(T)
                κ = norm(J) * norm(inv(J)) / D
                isfinite(κ) && (pp_cnd[p] = max(pp_cnd[p], κ))
                σ = svdvals(J)
                asp = σ[end] > zero(T) ? σ[1] / σ[end] : T(Inf)
                isfinite(asp) && (pp_asp[p] = max(pp_asp[p], asp))
            end
        end
        inverted && (n_inverted += 1)
        pp_sJ[p] = min(pp_sJ[p], elem_sJ[e])
    end

    per_patch = [(min_scaled_jacobian  = pp_sJ[p],
                  max_condition_number = pp_cnd[p],
                  max_aspect_ratio     = pp_asp[p]) for p in 1:np]
    return (min_scaled_jacobian     = minimum(elem_sJ),
            max_condition_number    = maximum(pp_cnd),
            max_aspect_ratio        = maximum(pp_asp),
            n_inverted              = n_inverted,
            per_patch               = per_patch,
            element_scaled_jacobian = elem_sJ)
end
