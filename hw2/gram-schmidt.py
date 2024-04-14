import numpy as np

# first create vec b_k[n] = n^k
b_k = np.zeros((5, 13))
index = np.arange(-6, 7)

for n in range(5):
    b_k[n] = np.power(index, n)

def gram_schmidt(A):
    
    basis = np.zeros_like(A)
    for i in range(A.shape[0]):
        
        for j in range(i):
            
            A[i] = A[i] - basis[j] * np.einsum('i,i->', basis[j], A[i])
        
        basis[i] = A[i] / np.sqrt(np.einsum('i,i->', A[i], A[i]))
    
    return basis

basis = gram_schmidt(b_k)

print("Gram-Schmidt-Basis:")
for i in range(basis.shape[0]):
    print(basis[i])