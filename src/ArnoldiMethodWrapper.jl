"""
    module ArnoldiMethodWrapper

Provides convenience wrapper for accessing the package ArnoldiMethod.

Implements the shift-and-invert transformation indicated [here](https://haampie.github.io/ArnoldiMethod.jl/stable/).

Main export is `partialschur(A,[B],σ;kwargs...)` and `partialeigen(A,[B],σ;kwargs...)`
"""
module ArnoldiMethodWrapper

export partialeigen

using ArnoldiMethod,
LinearAlgebra,
LinearMaps,
SparseArrays


"""
    type ShiftAndInvert{T,U,V}

Fieds:
`A::T`, `B::U`, `temp::V`.

Used to contain arrays for function `shift_and_invert` for use in `ArnoldiMethod.partialschur`
"""
struct ShiftAndInvert{T,U,V,Σ}
    A_lu::T
    B::U
    temp::V
    σ::Σ
    issymmetric::Bool
end


"""
    ShiftAndInvert(y,x)

A\\B*x = y, where `A=ShiftAndInvert.M.A_lu`, `B=ShiftAndInvert.M.B`, `x=ShiftAndInvert.M.x`
"""
function (SI::ShiftAndInvert)(y,x)
    mul!(SI.temp, SI.B, x)
    ldiv!(y, SI.A_lu, SI.temp)
    return nothing
end


"""
a = ShiftAndInvert(A, B, σ; diag_inv_B=false)

create a LinearMap object to feed to ArnoldiMethod.partialschur which transforms `Ax=λBx` into `(A-σB)⁻¹Bx=x/(λ-σ)`.

Set `diag_inv_B`=true if `B` is both diagonal and invertible, then it is easy to compute `B⁻¹`, so instead returns linear map `(B⁻¹A-σI)⁻¹`, which has same evals as above.

`A` and `B` must both be sparse or both dense. `A`, `B`, `σ` need not have common element type.
"""
function ShiftAndInvert(A::S, B::T, σ::Number; diag_inv_B::Bool=true) where {S,T}

    onetype = one(eltype(S))*one(eltype(T))*one(σ) ; type = typeof(onetype)
    A, B = onetype*A, onetype*B
    @assert supertype(typeof(A))==supertype(typeof(B)) "typeof(A)=$(typeof(A)), typeof(B)=$(typeof(B)). Either both sparse or both dense"
    if diag_inv_B
        if T<:AbstractSparseArray
            α = spdiagm(0=>map(x->1/x,diag(B)))*A-σ*I
            matrix_fun = sparse
        else
            α = Diagonal(map(x->1/x,diag(B)))*A-σ*I
            display(α)
            matrix_fun = Matrix
        end
        β = matrix_fun(onetype*I,size(B))
    else
        α = A-σ*B
        β = onetype*B
    end
    return ShiftAndInvert(lu(α), β, Vector{type}(undef,size(α,1)), σ, issymmetric(α)*issymmetric(β))
end


"""
    a = shift_and_invert(A, σ)
"""
function ShiftAndInvert(A::S, σ::Number; kwargs...) where S
    onetype = one(eltype(S))*one(σ)
    if S<:AbstractSparseArray
        return ShiftAndInvert(A, sparse(onetype*I,size(A)...), σ; diag_inv_B=true)
    else
        return ShiftAndInvert(A, Matrix(onetype*I,size(A)...), σ; diag_inv_B=true)
    end
end


"""
    partialschur(A,[B],σ; diag_inv_B=false, kwargs...) -> decomp, history

Partial Schur decomposition of `A`, with shift `σ` with mass matrix `B`, solving `Av=σBv`

For other keywords, see ArnoldiMethod.partialschur

see also: [`partialeigen`](@ref) in ArnoldiMethod
"""
function ArnoldiMethod.partialschur(si::ShiftAndInvert; kwargs...)
    a = LinearMap{eltype(si.B)}(si, size(si.B,1); ismutating=true, issymmetric=si.issymmetric)
    return partialschur(a; kwargs...)
end
ArnoldiMethod.partialschur(A, σ::Number; kwargs...) = partialschur(ShiftAndInvert(A, σ); kwargs...)
ArnoldiMethod.partialschur(A, B, σ::Number; diag_inv_B::Bool=false, kwargs...) = partialschur(ShiftAndInvert(A, B, σ; diag_inv_B=diag_inv_B); kwargs...)


"""
    partialeigen(A,[B],σ; diag_inv_B=false, untransform=true, kwargs...) -> (λ::Vector, v::Matrix)

Partial eigendecomposition of `A`, with mass matrix `B` and shift `σ` , solving `Av=λBv` for the eigenvalues closests to `σ`

If keyword `untransform=true`, the shift-invert transformation of the eigenvalues is inverted before returning

For other keywords, see ArnoldiMethod.partialschur

see also: [`partialschur`](@ref), [`partialeigen`](@ref) in ArnoldiMethod
"""
function partialeigen(si::ShiftAndInvert; kwargs...)
    decomp, history = partialschur(si; kwargs...)
    λ, v = partialeigen(decomp)
    get(kwargs,:untransform,true) ? λ = si.σ .+ 1 ./λ : nothing
    return λ, v
end
partialeigen(A, σ::Number; kwargs...) = partialeigen(ShiftAndInvert(A, σ); kwargs...)
partialeigen(A, B, σ::Number; diag_inv_B::Bool=false, kwargs...) = partialeigen(ShiftAndInvert(A, B, σ; diag_inv_B=diag_inv_B); kwargs...)


end # module ArnoldiMethodWrapper
