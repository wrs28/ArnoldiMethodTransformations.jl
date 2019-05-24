using LinearAlgebra, ArnoldiMethod, ArnoldiMethodTransformations

# construct fixed eval matrix in random basis
A = rand(ComplexF64,10,10)
B = rand(ComplexF64,10,10)

# find eigenpairs closest to .5
decomp, hist = partialschur(A,B,.5)

# get evecs
λ, v = partialeigen(decomp,.5)

norm(A*v-B*v*diagm(0=>λ))
