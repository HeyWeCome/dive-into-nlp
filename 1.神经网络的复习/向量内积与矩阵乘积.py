"""
向量内积直观的表示了"两个向量在多大程度上指向同一方向"
如果内积之和为1，完全指向同一方向
如果内积之和为-1，向量方向相反
"""
import numpy as np

# 向量内积®
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
dot_res = np.dot(a, b)
print("内积为:\n", dot_res)

# 矩阵乘积
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
arr_dot_res = np.dot(A, B)
print("矩阵乘积为:\n", arr_dot_res)
