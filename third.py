import numpy as np



def seidel(A, b, x0, tol=0.001, max_iter=1000):
    n = len(x0)
    x = x0.copy() # начальное приближение
    iter_num = 0 # счётчик
    while iter_num < max_iter:
        iter_num += 1
        for i in range(n):
            x[i] = (b[i] - np.dot(A[i, :i], x[:i]) - np.dot(A[i, i+1:], x0[i+1:])) / A[i, i]   
            # сумма по элементам до диагонального элемента в текущем уравнении и после
        if np.linalg.norm(x - x0) < tol:
            return x, iter_num
        x0 = x.copy()
        print(x)


A = np.array([[5.54, -0.29, 0.26], [0.83, -0.92, 3.11], [-1.51, 12.85, 2.5]])
b = np.array([-0.56, 4.42, -0.54])
# задаем начальное приближение
x0 = np.zeros_like(b)
seidel(A, b, x0, tol=0.001, max_iter=1000)
