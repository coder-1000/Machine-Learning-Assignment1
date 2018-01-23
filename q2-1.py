def responsibilities(k, mat):

    index = mat.argsort().argsort()
    mat[index >= k] = 0
    mat[index < k] = 1/k
    return mat



k = 2
a = 1000*np.random.random((4,4))
res = responsibilities(k, a)
print(res)
