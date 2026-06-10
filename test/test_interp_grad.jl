# The interpolation-derivative queries: lagrange_basis_deriv exactness
# on polynomials, tensor_interp_grad consistency, and
# element_point_and_jac against central differences of the forward map.

using HexMeshes
using HexMeshes: patch_to_global, locate_point
using StaticArrays
using Test

@testset "interp_grad queries" begin
    T = Float64
    # GLL-ish nodes on [0,1] (exactness only needs distinct nodes).
    xs = [0.0, 0.2763932022500211, 0.7236067977499789, 1.0]
    N = length(xs)

    @testset "lagrange_basis_deriv: exact for degree < N" begin
        for (f, df) in ((x -> 1.0, x -> 0.0),
                        (x -> x, x -> 1.0),
                        (x -> x^2 - 0.3x, x -> 2x - 0.3),
                        (x -> x^3, x -> 3x^2))
            vals = f.(xs)
            for ξ in (0.0, 0.31, 0.5, 0.97, 1.0)
                ℓ = HexMeshes.lagrange_basis(xs, ξ)
                dℓ = lagrange_basis_deriv(xs, ξ)
                @test sum(vals .* collect(ℓ)) ≈ f(ξ) atol = 1e-13
                @test sum(vals .* collect(dℓ)) ≈ df(ξ) atol = 1e-12
            end
        end
    end

    @testset "tensor_interp_grad ≡ tensor_interp + exact gradient" begin
        # Trilinear-in-each-direction polynomial field on the element.
        fxyz(ξ, η, ζ) = (1 + 2ξ + ξ^2) * (2 - η) * (1 + ζ^3)
        ue = [fxyz(xs[i], xs[j], xs[k]) for i in 1:N, j in 1:N, k in 1:N]
        ξ, η, ζ = 0.37, 0.81, 0.13
        v, g = tensor_interp_grad(ue, ξ, η, ζ, xs)
        @test v ≈ tensor_interp(ue, ξ, η, ζ, xs) atol = 1e-13
        @test g[1] ≈ (2 + 2ξ) * (2 - η) * (1 + ζ^3) atol = 1e-11
        @test g[2] ≈ -(1 + 2ξ + ξ^2) * (1 + ζ^3) atol = 1e-11
        @test g[3] ≈ (1 + 2ξ + ξ^2) * (2 - η) * 3ζ^2 atol = 1e-11
    end

    @testset "element_point_and_jac vs finite differences ($name)" for
            (name, mesh) in
            (("uniform", make_uniform_hex(T, 2, 2, 2, T(0), T(1))),
             ("radial shell", make_radial_shell_mesh(T, T(1), T(2), 2;
                                                     M_r = 2)),
             ("cubed cube", make_cubed_cube_mesh(T, 2, T(0.4))))
        for e in (1, mesh.Ne)
            ξ0 = SVector(0.3, 0.6, 0.45)
            P, J = element_point_and_jac(mesh, e, ξ0)
            h = 1e-6
            for a in 1:3
                δ = SVector{3,T}(ntuple(i -> i == a ? h : 0.0, 3))
                Pp, _ = element_point_and_jac(mesh, e, ξ0 + δ)
                Pm, _ = element_point_and_jac(mesh, e, ξ0 - δ)
                fd = (Pp - Pm) / (2h)
                @test maximum(abs, J[:, a] - fd) < 1e-6
            end
            # Consistency with the locate_point round-trip (covers the
            # Wedge dirs 2/3/6, whose global_to_patch inverse used to
            # disagree with the _ppj_wedge_3d node layout).
            eloc, ξloc = locate_point(mesh, P)
            @test eloc == e
            @test maximum(abs, ξloc - ξ0) < 1e-10
        end
    end
end
