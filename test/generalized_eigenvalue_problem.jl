using ArnoldiMethod
using ArnoldiMethodTransformations
using LinearAlgebra
using SparseArrays
using Statistics
using Test

N = 20
# construct fixed eval matrix in random basis

@testset "generalized eigenvalue problem: " begin
    for i ∈ 1:500

        A = rand(N,N)
        B = N*I + rand(ComplexF64,N,N)
        # find eigenpairs closest to 5.001 (cannot be 5 as algorithm is unstable if σ is exactly an eval)
        decomp, hist = partialschur(A,B,5.001)
        # get evecs
        λ, v = partialeigen(decomp,5.001)
        @test norm(A*v-B*v*diagm(0=>λ))/prod(size(v)) ≤ sqrt(eps(float(N)))N

        As = sparse(A)
        Bs = sparse(B)
        # find eigenpairs closest to 5.001 (cannot be 5 as algorithm is unstable if σ is exactly an eval)
        decomp, hist = partialschur(As,Bs,5.001)
        # get evecs
        λ, v = partialeigen(decomp,5.001)
        @test norm(A*v-Bs*v*diagm(0=>λ))/prod(size(v)) ≤ sqrt(eps(float(N)))N
    end
end
