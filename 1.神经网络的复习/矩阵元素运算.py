import numpy as np

W = np.array([[1, 2, 3], [4, 5, 6]])
X = np.array([[0, 1, 2], [3, 4, 5]])
plus_result = W + X
multi_result = W * X

print('plus:\n', plus_result)
print('multi:\n', multi_result)

# 广播机制，形状不同的数组之间进行运算
A = np.array([[1, 2], [3, 4]])
print('广播机制:\n', A * 10)
A = np.array([[1, 2], [3, 4]])
b = np.array([10, 20])
print(A * b)
