import numpy as np

# first create vec b_k[n] = n^k
b_k = np.zeros((5, 13))
index = np.arange(-6, 7)
basis = np.zeros_like(b_k)
weight = 1 - np.abs(index) / 7

for n in range(5):
    b_k[n] = np.power(index, n)

# start doing gram-schmidt process
def wgram_schmidt(A,w):
    
    basis = np.zeros_like(A)
    for i in range(A.shape[0]):
        
        for j in range(i):
            
            A[i] = A[i] - basis[j] * np.einsum('i,i,i->', basis[j], A[i], w)
        
        basis[i] = A[i] / np.sqrt(np.einsum('i,i,i->', A[i], A[i], w))
    
    return np.array(basis)
  

basis = wgram_schmidt(b_k, weight)

print("Weighted-Gram-Schmidt-Basis:")
for i in range(basis.shape[0]):
    print(basis[i])