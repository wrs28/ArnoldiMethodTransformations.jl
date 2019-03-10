# ArnoldiMethodWrapper

A package for easily interfacing with [ArnoldiMethod](https://github.com/haampie/ArnoldiMethod.jl), using the suggested [transformations](https://haampie.github.io/ArnoldiMethod.jl/stable/usage/02_spectral_transformations.html) suggested in the [documentation](https://haampie.github.io/ArnoldiMethod.jl/stable/index.html).


## Installation

In REPL, type either `] add git@github.com:wrs28/ArnoldiMethodWrapper.git` or
````JULIA
using Pkg
Pkg.add("git@github.com:wrs28/ArnoldiMethodWrapper.git")
````

## Methods
This package exports no methods, but extends `partialschur` from [ArnoldiMethod](https://github.com/haampie/ArnoldiMethod.jl).

The new methods are:


`partialschur(A,σ; kwargs...) -> decomp, hist` which shift-and-inverts `A` by `σ`. The eigenvalues returned are those closest to `σ`.


`partialschur(A,B,σ; diag_inv_B=true, kwargs...) -> decomp, hist` which shift-and-inverts the generalized eigenvalue problem `Ax=σBx`, as described [here](https://haampie.github.io/ArnoldiMethod.jl/stable/theory.html#Spectral-transformations-1). `diag_inv_B=true` means that `B` is diagonal and invertible, which makes for an especially efficient transformation.

For both, `kwargs` are the keyword arguments from [`ArnoldiMethod.partialschur`](https://haampie.github.io/ArnoldiMethod.jl/stable/usage/01_getting_started.html#ArnoldiMethod.partialschur)
