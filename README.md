# ArnoldiMethodTransformations

A package for easily interfacing with [ArnoldiMethod](https://github.com/haampie/ArnoldiMethod.jl), using the suggested [transformations](https://haampie.github.io/ArnoldiMethod.jl/stable/usage/02_spectral_transformations.html) suggested in the [documentation](https://haampie.github.io/ArnoldiMethod.jl/stable/index.html).


## Installation

In REPL, type either `] add git@github.com:wrs28/ArnoldiMethodTransformations.git` or
````JULIA
using Pkg
Pkg.add("git@github.com:wrs28/ArnoldiMethodTransformations.git")
````

This package does not export any new methods, it only extends some methods of [ArnoldiMethod](https://github.com/haampie/ArnoldiMethod.jl), which needs to be separately installed.

## Example
Ordinary eigenvalue problem `Ax=λx`
````JULIA
using LinearAlgebra, ArnoldiMethod, ArnoldiMethodTransformations

# construct fixed eval matrix in random basis
D = diagm(0=>[0,1,2,3,4,5,6,7,8,9])
S = randn(10,10)
A = S\D*S

# find eigenpairs closest to 5.001 (cannot be 5 as algorithm is unstable if σ is exactly an eval)
decomp, hist = partialschur(A,5.001)

# get evecs
λ, v = partialeigen(decomp)

display(decomp.eigenvalues)
norm(A*v-v*diagm(0=>decomp.eigenvalues))
# should be ~1e-11 or smaller
````

Generalized eigenvalue problem `Ax=λBx`
````JULIA
using LinearAlgebra, ArnoldiMethod, ArnoldiMethodTransformations

# construct fixed eval matrix in random basis
A = rand(ComplexF64,10,10)
B = rand(ComplexF64,10,10)

# find eigenpairs closest to .5
decomp, hist = partialschur(A,B,.5)

# get evecs
λ, v = partialeigen(decomp)

display(decomp.eigenvalues)
norm(A*v-B*v*diagm(0=>decomp.eigenvalues))
# should be ~1e-14 or smaller
````

Note that in both cases, `ArnoldiMethod` needed to be explicitly brought into scope with `using`.

## Methods
This package exports no methods, but extends `partialschur`  and `partialeigen` from [ArnoldiMethod](https://github.com/haampie/ArnoldiMethod.jl).

The new methods are:


`partialschur(A,σ; kwargs...) -> decomp, hist` which shift-and-inverts `A` by `σ`. The eigenvalues returned are those closest to `σ`.


`partialschur(A,B,σ; diag_inv_B=false, kwargs...) -> decomp, hist` which shift-and-inverts the generalized eigenvalue problem `Ax=σBx`, as described [here](https://haampie.github.io/ArnoldiMethod.jl/stable/theory.html#Spectral-transformations-1). `diag_inv_B=true` means that `B` is diagonal and invertible, which makes for an especially efficient transformation.

For both, `kwargs` are the keyword arguments from [`ArnoldiMethod.partialschur`](https://haampie.github.io/ArnoldiMethod.jl/stable/usage/01_getting_started.html#ArnoldiMethod.partialschur)

Note that the shifting to an exact eigenvalue poses a problem, see note on [purification](https://haampie.github.io/ArnoldiMethod.jl/stable/theory.html#Purification-1).


## Linear Solvers
There are three solvers currently available for use in this package: UMFPACK (via `Base.LinAlg`), [MUMPS](http://mumps.enseeiht.fr) (via [`MUMPS3`](https://github.com/wrs28/MUMPS3.jl)), and [Pardiso](https://pardiso-project.org) (via [`Pardiso`](https://github.com/JuliaSparse/Pardiso.jl)).

MUMPS and Pardiso are often faster, and use significantly less memory, but require separate installation, which not all users will want to do. This optional dependency is implemented with [Requires.jl](https://github.com/MikeInnes/Requires.jl), and works like so: MUMPS is used for linear solve if `MUMPS3` is loaded at the top level, Pardiso is used if `Pardiso` is loaded, else UMFPACK is used.
