using ArnoldiMethod
using ArnoldiMethodTransformations
using LinearAlgebra
using SparseArrays
using Statistics
using Test

N = 20
# construct fixed eval matrix in random basis
D = diagm(0=>0:(N-1))

@testset "ordinary eigenvalue problem: " begin
    for i ∈ 1:500
        S = I + .1*randn(N,N) # random basis

        A = S\D*S
        # find eigenpairs closest to 5.001 (cannot be 5 as algorithm is unstable if σ is exactly an eval)
        decomp, hist = partialschur(A,5.001)
        # get evecs
        λ, v = partialeigen(decomp,5.001)
        @test norm(A*v-v*diagm(0=>λ))/prod(size(v)) ≤ sqrt(eps(float(N)))N

        As = sparse(A)
        # find eigenpairs closest to 5.001 (cannot be 5 as algorithm is unstable if σ is exactly an eval)
        decomp, hist = partialschur(As,5.001)
        # get evecs
        λ, v = partialeigen(decomp,5.001)
        @test norm(A*v-v*diagm(0=>λ))/prod(size(v)) ≤ sqrt(eps(float(N)))N
    end
end
