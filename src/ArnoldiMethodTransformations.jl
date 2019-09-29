"""
    module ArnoldiMethodTransformations

Provides convenience wrapper for interfacing with the package ArnoldiMethod.

Implements the shift-and-invert transformation detailed [here](https://haampie.github.io/ArnoldiMethod.jl/stable/).

The main functions are `partialschur(A,[B],σ; kwargs...)` and `partialeigen(A,[B],σ; kwargs...)`

The constants `USOLVER`, `PSOLVER`, and `MSOLVER` are exported.
"""
module ArnoldiMethodTransformations

export PSOLVER
export MSOLVER
export USOLVER

using ArnoldiMethod
using LinearAlgebra
using LinearMaps
using Requires
using SparseArrays

abstract type AbstractSolver end
struct PSolver <: AbstractSolver end
const PSOLVER = PSolver()
struct MSolver <: AbstractSolver end
const MSOLVER = MSolver()
struct USolver <: AbstractSolver end
const USOLVER = USolver()

const DEFAULT_SOLVER = USOLVER

function __init__()
    @require MUMPS3="da04e1cc-30fd-572f-bb4f-1f8673147195" @eval using MUMPS3, MPI
    @require Pardiso="46dd5b70-b6fb-5a00-ae2d-e8fea33afaf2" @eval using Pardiso
end

struct ShiftAndInvert{M,T,U,V,Σ}
    A_lu::T
    B::U
    temp::V
    temp2::V
    σ::Σ
    issymmetric::Bool

    function ShiftAndInvert(
                A::S,
                B::T,
                σ::U;
                diag_inv_B::Bool,
                lupack::AbstractSolver = DEFAULT_SOLVER
                ) where {S<:AbstractArray,T<:AbstractArray,U<:Number}

        type = promote_type(eltype(S),eltype(T),U)
        A, B = convert.(type,A), convert.(type,B)
        supertype(typeof(A))==supertype(typeof(B)) || throw("typeof(A)=$(typeof(A)), typeof(B)=$(typeof(B)). Either both sparse or both dense")
        if diag_inv_B
            if T<:AbstractSparseArray
                α = spdiagm(0=>map(x->1/x,diag(B)))*A-σ*I
                β = sparse(one(type)*I,size(B))
            else
                α = Diagonal(map(x->1/x,diag(B)))*A-σ*I
                β = Matrix(one(type)*I,size(B))
            end
        else
            α = A-σ*B
            β = convert.(type,B)
        end
        temp = Vector{type}(undef,size(α,1))
        temp2 = Vector{type}(undef,size(α,1))
        issym = issymmetric(α) & issymmetric(β)

        # initialize according to package used
        M, A_lu = initialize_according_to_package(lupack,issym,type,α,temp,temp2)
        return new{typeof(M),typeof(A_lu),typeof(β),typeof(temp),typeof(σ)}(A_lu,β,temp,temp2,σ,issym)
    end
end

ShiftAndInvert(A::S,σ::Q; kwargs...) where {S<:AbstractSparseArray,Q<:Number} =
    ShiftAndInvert(A, sparse(one(promote_type(eltype(S),Q))*I,size(A)...), σ; diag_inv_B=true, kwargs...)
ShiftAndInvert(A::S,σ::Q; kwargs...) where {S<:AbstractArray,Q<:Number} =
    ShiftAndInvert(A, Matrix(one(promote_type(eltype(S),Q))*I,size(A)...), σ; diag_inv_B=true, kwargs...)

# define action of ShiftAndInvert
function (SI::ShiftAndInvert{PSolver})(y,x)
    mul!(SI.temp, SI.B, x)
    pardiso(SI.A_lu,SI.temp2,spzeros(eltype(SI.B),size(SI.B)...),SI.temp)
    for i ∈ eachindex(y) y[i] = SI.temp2[i] end
    return nothing
end
function (SI::ShiftAndInvert{M})(y,x) where M<:Union{USolver,MSolver}
    mul!(SI.temp, SI.B, x)
    ldiv!(y, SI.A_lu, SI.temp)
    return nothing
end


function initialize_according_to_package(
            lupack::S,
            issym,
            type,
            α,
            temp1,
            temp2
            ) where S<:AbstractSolver

    if S <: PSolver
        M = PSOLVER
        try PardisoSolver catch; throw(ErrorException("Pardiso not loaded. Try again after `using Pardiso`")) end
        A_lu = PardisoSolver()
        set_iparm!(A_lu,1,1) # don't revert to defaults
        set_iparm!(A_lu,12,1) # transpose b/c of CSR vs SCS
        x = Vector{type}(undef,size(α,1))
        y = Vector{type}(undef,size(α,1))
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
    elseif S <: MSolver
        M = MSOLVER
        try MPI catch; throw(ErrorException("MUMPS3 not loaded. Try again after `using MUMPS3`")) end
        MPI.Initialized() ? nothing : MPI.Init()
        A_lu = mumps_factorize(α)
    else
        M = USOLVER
        A_lu = lu(α)
    end
    return M, A_lu
end


function ArnoldiMethod.partialschur(
            si::ShiftAndInvert;
            kwargs...)
    a = LinearMap{eltype(si.B)}(si, size(si.B,1); ismutating=true, issymmetric=si.issymmetric)
    return partialschur(a; kwargs...)
end
function ArnoldiMethod.partialschur(
            A::AbstractArray,
            σ::Number;
            lupack::AbstractSolver=DEFAULT_SOLVER,
            kwargs...)
    return partialschur(ShiftAndInvert(A, σ; lupack=lupack); kwargs...)
end
function ArnoldiMethod.partialschur(
            A::AbstractArray,
            B::AbstractArray,
            σ::Number;
            diag_inv_B::Bool = isdiag(B) &&! any(iszero.(diag(B))),
            lupack::AbstractSolver = DEFAULT_SOLVER,
            kwargs...)
    return partialschur(ShiftAndInvert(A, B, σ; diag_inv_B=diag_inv_B, lupack=lupack); kwargs...)
end


function ArnoldiMethod.partialeigen(
            decomp::ArnoldiMethod.PartialSchur,
            σ::Number)
    λ, v = partialeigen(decomp)
    return σ.+1 ./λ, v
end


"""
    struct ShiftAndInvert{M,T,U,V,Σ}

Container for arrays in `ArnoldiMethod.partialschur`

-------------

    function ShiftAndInvert(A, [B], σ; diag_inv_B=isdiag(B), lupack=:auto) -> si

create a LinearMap object to feed to ArnoldiMethod.partialschur which transforms `Ax=λBx` into `(A-σB)⁻¹Bx=x/(λ-σ)`.

Set `diag_inv_B=true` if `B` is both diagonal and invertible, so that it is easy to compute `B⁻¹`. In this case instead return a linear map `(B⁻¹A-σI)⁻¹`, which has same evals as above.

`A` and `B` must both be sparse or both dense. `A`, `B`, `σ` need not have common element type.

Keyword `lupack` determines what linear algebra library to use. Options are `:pardiso`, `:mumps`, `:umfpack`,
and the default `:auto`, which chooses based on availability at the top level, in this order:
PARDISO > MUMPS > UMFPACK. For example, if at the top level there is `using ArnoldiMethod, ArnoldiMethodTransformations`,
will default to UMFPACK, while the additional `using MUMPS3` will default to `:mumps`, and so on.

----------

    function ::ShiftAndInvert(y,x)

`A\\B*x = y`, where `A=ShiftAndInvert.A_lu` (factorized), `B=ShiftAndInvert.B`, and `x=ShiftAndInvert.x`
"""
ShiftAndInvert


"""
    partialschur(A, [B], σ; [diag_inv_B, lupack=:auto, kwargs...]) -> decomp, history

Partial Schur decomposition of `A`, with shift `σ` and mass matrix `B`, solving `A*v=σB*v`

Keyword `diag_inv_B` defaults to `true` if `B` is both diagonal and invertible. This enables
a simplified shift-and-invert scheme.

Keyword `lupack` determines what linear algebra library to use. Options are `:pardiso`, `:mumps`, `:umfpack`,
and the default `:auto`, which chooses based on availability at the top level, in this order:
PARDISO > MUMPS > UMFPACK. For example, if at the top level there is `using ArnoldiMethod, ArnoldiMethodTransformations`,
will default to UMFPACK, while the additional `using MUMPS3` will default to `:mumps`, and so on.

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


end # module ArnoldiMethodTransformations
