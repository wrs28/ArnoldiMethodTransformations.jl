"""
    module ArnoldiMethodTransformations

Provides convenience wrapper for interfacing with the package ArnoldiMethod.

Implements the shift-and-invert transformation detailed [here](https://haampie.github.io/ArnoldiMethod.jl/stable/).

The main functions are `partialschur(A,[B],σ; kwargs...)` and `partialeigen(A,σ; kwargs...)`

The constants `USOLVER`, `PSOLVER`, and `MSOLVER` are exported.
"""
module ArnoldiMethodTransformations

export PSOLVER
export PSolver
export MSOLVER
export MSolver
export USOLVER
export USolver

using ArnoldiMethod
using LinearAlgebra
using LinearMaps
using Requires
using SparseArrays

abstract type AbstractSolver end
struct PSolver <: AbstractSolver end
struct MSolver <: AbstractSolver end
struct USolver <: AbstractSolver end

# solver constants (exported)
const PSOLVER = PSolver()
const MSOLVER = MSolver()
const USOLVER = USolver()

# colors for pretty printing
const DEFAULT_SOLVER = USOLVER
const SOLVER_COLOR = :cyan
const PRINTED_TYPE_COLOR = 171

# load MUMPS or PARDISO as approprite
function __init__()
    @require MUMPS3="da04e1cc-30fd-572f-bb4f-1f8673147195" @eval using MUMPS3, MPI
    @require Pardiso="46dd5b70-b6fb-5a00-ae2d-e8fea33afaf2" @eval using Pardiso
end

# main structure used internally
struct ShiftAndInvert{M,TLU,TB,V,Σ,TZ}
    LU::TLU
    B::TB
    temp1::V
    temp2::V
    σ::Σ
    diag_inv_B::Bool
    issymmetric::Bool
    Z::TZ
end

# for ordinary EVP with Dense Matrix
function ShiftAndInvert(
            A::Matrix{TA};
            σ::Σ=0,
            lupack::AbstractSolver = DEFAULT_SOLVER
            ) where {TA,TB,Σ<:Number}

    diag_inv_B=true
    T = promote_type(TA,Σ)
    α = A-σ*I
    β = Matrix(one(T)I,size(A))

    temp1 = Vector{T}(undef,size(α,1))
    temp2 = Vector{T}(undef,size(α,1))
    issym = issymmetric(α) & issymmetric(β)
    Z = spzeros(T,size(β)...)

    M, a_lu = initialize_according_to_package(lupack,α,issym,T,temp1,temp2)
    return ShiftAndInvert{typeof(M),typeof(a_lu),typeof(β),typeof(temp1),typeof(σ),typeof(Z)}(a_lu,β,temp1,temp2,σ,diag_inv_B,issym,Z)
end
# for generalized EVP with Dense Matrices
function ShiftAndInvert(
            A::Matrix{TA},
            B::Matrix{TB};
            σ::Σ=0,
            diag_inv_B::Bool=false,
            lupack::AbstractSolver = DEFAULT_SOLVER
            ) where {TA,TB,Σ<:Number}

    T = promote_type(TA,TB,Σ)
    if diag_inv_B
        for i ∈ diagind(B) B[i]=1/B[i] end
        α = B*A-σ*I
        for i ∈ diagind(B) B[i]=1/B[i] end
        β = Matrix(one(T)I,size(B))
    else
        α = A-σ*B
        β = convert(Matrix{T},B)
    end
    temp1 = Vector{T}(undef,size(α,1))
    temp2 = Vector{T}(undef,size(α,1))
    issym = issymmetric(α) & issymmetric(β)
    Z = spzeros(T,size(β)...)

    M, a_lu = initialize_according_to_package(lupack,α,issym,T,temp1,temp2)
    return ShiftAndInvert{typeof(M),typeof(a_lu),typeof(β),typeof(temp1),typeof(σ),typeof(Z)}(a_lu,β,temp1,temp2,σ,diag_inv_B,issym,Z)
end
# for ordinary EVP with Sparse Matrix
function ShiftAndInvert(
            A::SparseMatrixCSC{TA};
            σ::Σ=0,
            lupack::AbstractSolver = DEFAULT_SOLVER
            ) where {TA,TB,Σ}

    diag_inv_B=true
    T = promote_type(TA,Σ)
    α = A-σ*I
    β = sparse(one(T)I,size(A))
    temp1 = Vector{T}(undef,size(α,1))
    temp2 = Vector{T}(undef,size(α,1))
    issym = issymmetric(α) & issymmetric(β)
    Z = spzeros(T,size(β)...)

    # initialize according to package used
    M, a_lu = initialize_according_to_package(lupack,α,issym,T,temp1,temp2)
    return ShiftAndInvert{typeof(M),typeof(a_lu),typeof(β),typeof(temp1),typeof(σ),typeof(Z)}(a_lu,β,temp1,temp2,σ,diag_inv_B,issym,Z)
end
# for generalized EVP with Sparse Matrices
function ShiftAndInvert(
            A::SparseMatrixCSC{TA},
            B::SparseMatrixCSC{TB};
            σ::Σ=0,
            diag_inv_B::Bool=false,
            lupack::AbstractSolver = DEFAULT_SOLVER
            ) where {TA,TB,Σ}

    T = promote_type(TA,TB,Σ)
    if diag_inv_B
        for i ∈ diagind(B) B[i]=1/B[i] end
        α = B*A-σ*I
        for i ∈ diagind(B) B[i]=1/B[i] end
        β = sparse(one(T)I,size(B))
    else
        α = A-σ*B
        β = convert(SparseMatrixCSC{T,Int},B)
    end
    temp1 = Vector{T}(undef,size(α,1))
    temp2 = Vector{T}(undef,size(α,1))
    issym = issymmetric(α) & issymmetric(β)
    Z = spzeros(T,size(β)...)

    # initialize according to package used
    M, a_lu = initialize_according_to_package(lupack,α,issym,T,temp1,temp2)
    return ShiftAndInvert{typeof(M),typeof(a_lu),typeof(β),typeof(temp1),typeof(σ),typeof(Z)}(a_lu,β,temp1,temp2,σ,diag_inv_B,issym,Z)
end
# if not both dense or both sparse
ShiftAndInvert(A,B;kwargs...) = throw("typeof(A)=$(typeof(A)), typeof(B)=$(typeof(B)). Either both sparse or both dense")

# define action of ShiftAndInvert
function (SI::ShiftAndInvert{M})(y,x) where M<:Union{USolver,MSolver}
    #first multiply Bv
    if SI.diag_inv_B
        for i ∈ eachindex(SI.temp1) SI.temp1[i] = x[i] end
    else
        mul!(SI.temp1, SI.B, x)
    end
    # then left-divide (A-σB)⁻¹*(Bv)
    ldiv!(y, SI.LU, SI.temp1)
    return nothing
end
function (SI::ShiftAndInvert{PSolver})(y,x)
    #first multiply Bv
    if SI.diag_inv_B
        for i ∈ eachindex(SI.temp1) SI.temp1[i] = x[i] end
    else
        mul!(SI.temp1, SI.B, x)
    end
    # then left-divide (A-σB)⁻¹*(Bv)
    pardiso(SI.LU,SI.temp2,SI.Z,SI.temp1)
    for i ∈ eachindex(y) y[i] = SI.temp2[i] end
    return nothing
end

# generate a solver-dependent LU object
# UMFPACK
initialize_according_to_package(::USolver,α,args...) = USOLVER, lu(α)
# MUMPS
function initialize_according_to_package(::MSolver,α,args...)
    try MPI catch; throw(ErrorException("MUMPS3 not loaded. Try again after `using MUMPS3`")) end
    MPI.Initialized() ? nothing : MPI.Init()
    return MSOLVER, mumps_factorize(α)
end
# PARDISO
function initialize_according_to_package(::PSolver,α,issym,type,x,y)
    try PardisoSolver catch; throw(ErrorException("Pardiso not loaded. Try again after `using Pardiso`")) end
    A_lu = PardisoSolver()
    set_iparm!(A_lu,1,1) # don't revert to defaults
    set_iparm!(A_lu,12,1) # transpose b/c of CSR vs SCS
    # x = Vector{type}(undef,size(α,1))
    # y = Vector{type}(undef,size(α,1))
    if issym & (type<:Real)
        set_matrixtype!(A_lu,Pardiso.REAL_SYM)
        pardiso(A_lu,x,sparse(tril(α)),y)
    elseif issym & (type<:Complex)
        set_matrixtype!(A_lu,Pardiso.COMPLEX_SYM)
        pardiso(A_lu,x,sparse(tril(α)),y)
    elseif !issym & (type<:Real)
        set_matrixtype!(A_lu,Pardiso.REAL_NONSYM)
        pardiso(A_lu,x,sparse(α),y)
    else # if !issym & type<:Complex
        set_matrixtype!(A_lu,Pardiso.COMPLEX_NONSYM)
        pardiso(A_lu,x,sparse(α),y)
    end
    set_phase!(A_lu,12) # analyze and factorize
    set_phase!(A_lu,33) # set to solve for future calls
    return PSOLVER, A_lu
end

# pretty printing
function Base.show(io::IO,si::ShiftAndInvert{M}) where M
    printstyled(io,"ShiftAndInvert ",color=PRINTED_TYPE_COLOR)
    print(io,"(using ")
    if M==MSolver
        printstyled(io,"MUMPS",color=SOLVER_COLOR)
    elseif M==PSolver
        printstyled(io,"Pardiso",color=SOLVER_COLOR)
    elseif M==USolver
        printstyled(io,"UMFPACK",color=SOLVER_COLOR)
    end
    print(io,")")
end


################################################################################

# add method to partialschur(::ShiftAndInvert)
function ArnoldiMethod.partialschur(si::ShiftAndInvert; kwargs...)
    a = LinearMap{eltype(si.B)}(si, size(si.B,1); ismutating=true, issymmetric=si.issymmetric)
    return partialschur(a; kwargs...)
end
# add method to partialschur(A,σ)
function ArnoldiMethod.partialschur(
            A::AbstractArray,
            σ::Number;
            lupack::AbstractSolver=DEFAULT_SOLVER,
            kwargs...)
    return partialschur(ShiftAndInvert(A; σ=σ, lupack=lupack); kwargs...)
end
# add method to partialschur(A,B,σ)
function ArnoldiMethod.partialschur(
            A::AbstractArray,
            B::AbstractArray,
            σ::Number;
            diag_inv_B::Bool = isdiag(B) && !any(iszero.(diag(B))),
            lupack::AbstractSolver=DEFAULT_SOLVER,
            kwargs...)
    return partialschur(ShiftAndInvert(A, B; σ=σ, diag_inv_B=diag_inv_B, lupack=lupack); kwargs...)
end

# add method to partialeigen(decomp,σ)
function ArnoldiMethod.partialeigen(
            decomp::ArnoldiMethod.PartialSchur,
            σ::Number)
    λ, v = partialeigen(decomp)
    foreach(i->λ[i]=σ+1/λ[i],eachindex(λ))
    return λ, v
end


################################################################################
"""
    struct ShiftAndInvert

Container for arrays in `ArnoldiMethod.partialschur`

-------------

    function ShiftAndInvert(A, [B]; σ=0, diag_inv_B=false, lupack=USOLVER) -> si

create a LinearMap object to feed to ArnoldiMethod.partialschur which transforms `Ax=λBx` into `(A-σB)⁻¹Bx=x/(λ-σ)`.

Set `diag_inv_B=true` if `B` is both diagonal and invertible, so that it is easy to compute `B⁻¹`. In this case instead return a linear map `(B⁻¹A-σI)⁻¹`, which has same evals as above.

`A` and `B` must both be sparse or both dense. `A`, `B`, `σ` need not have common element type.

Keyword `lupack` determines what linear algebra library to use. Options are `PSOLVER`, `MSOLVER`, `USOLVER` (the default).
For all solvers besides UMFPACK, the appropriate library must be loaded at the top level.
----------

    function ::ShiftAndInvert(y,x)

`A\\B*x = y`
"""
ShiftAndInvert


"""
    partialschur(A, [B], σ; [diag_inv_B=false, lupack=USOLVER, kwargs...]) -> decomp, history

Partial Schur decomposition of `A`, with shift `σ` and mass matrix `B`, solving `Av=σBv`

Keyword `diag_inv_B` defaults to `true` if `B` is both diagonal and invertible. This enables
a simplified shift-and-invert scheme.

Keyword `lupack` determines what linear algebra library to use. Options are `PSOLVER`, `MSOLVER`, `USOLVER` (the default).
For all solvers besides UMFPACK, the appropriate library must be loaded at the top level.

For other keywords, see ArnoldiMethod.partialschur

see also: [`partialeigen`](@ref) in ArnoldiMethod
"""
partialschur


"""
    partialeigen(decomp, σ)

Transforms a partial Schur decomposition into an eigendecomposition, but undoes the
shift-and-invert of the eigenvalues by σ.
"""
partialeigen

end # module
