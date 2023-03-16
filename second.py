import numpy as np

# задаем коэффициенты СЛАУ
# 1 пример
A = np.array([[5.54, -0.29, 0.26], [0.83, -0.92, 3.11], [-1.51, 12.85, 2.5]])
b = np.array([-0.56, 4.42, -0.54])

# задаем начальное приближение
x0 = np.zeros_like(b)

# задаем максимальное число итераций
max_iter = 1000

# задаем точность
tolerance = 0.001

# выполняем итерации
x = x0
for i in range(max_iter):
    x_new = np.zeros_like(x)
    for j in range(len(b)):
        s = np.dot(A[j, :], x) - A[j, j] * x[j]
        x_new[j] = x[j] + (b[j] - s) / A[j, j]
    if np.allclose(x, x_new, rtol=tolerance):
        break
    x = x_new
    solution = "Solution: " + str(x) if x is not None else "No solution found" # чтоб отбросить хоть чуть None
    print(solution)
