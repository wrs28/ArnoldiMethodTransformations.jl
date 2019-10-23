using ArnoldiMethod
using ArnoldiMethodTransformations
using LinearAlgebra
using SparseArrays
using Test

# construct fixed eval matrix in random basis
D = diagm(0=>[0,1,2,3,4,5,6,7,8,9])

@testset "functionlity: " begin
    S = I + .1*randn(10,10) # random basis
    A = S\D*S
    S = I + .1*randn(10,10) # random basis
    B = S\D*S

    si = ArnoldiMethodTransformations.ShiftAndInvert(A;σ=5.001)
    decomp, hist = partialschur(si)
    λ, v = partialeigen(decomp,5.001)
    @test norm(A*v-v*diagm(0=>λ))/prod(size(v)) ≤ 5*sqrt(eps())
    decomp, hist = partialschur(A,5.001)
    λ, v = partialeigen(decomp,5.001)
    @test norm(A*v-v*diagm(0=>λ))/prod(size(v)) ≤ 5*sqrt(eps())

    As = sparse(A)
    si = ArnoldiMethodTransformations.ShiftAndInvert(As;σ=5.001)
    decomp, hist = partialschur(si)
    λ, v = partialeigen(decomp,5.001)
    @test norm(A*v-v*diagm(0=>λ))/prod(size(v)) ≤ 5*sqrt(eps())
    decomp, hist = partialschur(A,5.001)
    λ, v = partialeigen(decomp,5.001)
    @test norm(A*v-v*diagm(0=>λ))/prod(size(v)) ≤ 5*sqrt(eps())

    si = ArnoldiMethodTransformations.ShiftAndInvert(A,B;σ=5.001)
    decomp, hist = partialschur(si)
    λ, v = partialeigen(decomp,5.001)
    @test norm(A*v-B*v*diagm(0=>λ))/prod(size(v)) ≤ 5*sqrt(eps())
    decomp, hist = partialschur(A,B,5.001)
    λ, v = partialeigen(decomp,5.001)
    @test norm(A*v-B*v*diagm(0=>λ))/prod(size(v)) ≤ 5*sqrt(eps())

    si = ArnoldiMethodTransformations.ShiftAndInvert(A,B;σ=5.001)
    Bs = sparse(B)
    decomp, hist = partialschur(si)
    λ, v = partialeigen(decomp,5.001)
    @test norm(As*v-Bs*v*diagm(0=>λ))/prod(size(v)) ≤ 5*sqrt(eps())
    decomp, hist = partialschur(As,Bs,5.001)
    λ, v = partialeigen(decomp,5.001)
    @test norm(As*v-Bs*v*diagm(0=>λ))/prod(size(v)) ≤ 5*sqrt(eps())
end

# @code_warntype ArnoldiMethodTransformations.ShiftAndInvert(A;σ=5.001)
# @time si = ArnoldiMethodTransformations.ShiftAndInvert(A;σ=5.001)
#
# @code_warntype ArnoldiMethodTransformations.ShiftAndInvert(As;σ=5.001)
# @time si = ArnoldiMethodTransformations.ShiftAndInvert(As;σ=5.001)
#
# @code_warntype ArnoldiMethodTransformations.ShiftAndInvert(A,B;σ=5.001)
# @time si = ArnoldiMethodTransformations.ShiftAndInvert(A,B;σ=5.001)
#
# @code_warntype ArnoldiMethodTransformations.ShiftAndInvert(As,Bs;σ=5.001)
# @time si = ArnoldiMethodTransformations.ShiftAndInvert(As,Bs;σ=5.001)
