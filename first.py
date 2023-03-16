import numpy as np

def gauss(A, B): # 1
    n = len(A)
    for i in range(n):
        max_row = i
        for j in range(i+1, n):
            # проверка на наибольший по модулю элемент
            if abs(A[j][i]) > abs(A[max_row][i]):
                max_row = j
        A[i], A[max_row] = A[max_row], A[i]
        B[i], B[max_row] = B[max_row], B[i]
        for j in range(i+1, n):
            q = A[j][i] / A[i][i]
            for k in range(i+1, n):
                A[j][k] -= q * A[i][k]
            B[j] -= q * B[i]
    X = [0] * n
    # обратный ход
    for i in range(n-1, -1, -1):
        X[i] = B[i] / A[i][i]
        for j in range(i):
            B[j] -= A[j][i] * X[i]
    return X


def lu_decomposition(A): # 2
    n = len(A)
    L = [[0.0] * n for i in range(n)]
    U = [[0.0] * n for i in range(n)]

    for j in range(n): 
        L[j][j] = 1.0  # Задаем диагональные элементы матрицы L равными 1
        for i in range(j+1): 
            s1 = sum(U[k][j] * L[i][k] for k in range(i))  # Считаем сумму элементов
            U[i][j] = A[i][j] - s1  # Вычисляем элементы матрицы U
        for i in range(j, n):
            s2 = sum(U[k][j] * L[i][k] for k in range(j))  # Считаем сумму элементов
            L[i][j] = (A[i][j] - s2) / U[j][j]  # Вычисляем элементы матрицы L
    
    return L, U

def lu_solve(A, b): # 2
    n = len(A) 
    L, U = lu_decomposition(A)  # Вычисляем LU разложение матрицы A
    y = [0.0] * n
    x = [0.0] * n

    # Решение Ly = b методом прогонки
    for i in range(n):
        s = sum(L[i][j] * y[j] for j in range(i))  # Считаем сумму элементов
        y[i] = b[i] - s

    # Решение Ux = y методом прогонки
    for i in range(n-1, -1, -1):
        s = sum(U[i][j] * x[j] for j in range(i+1, n))  # Считаем сумму элементов
        x[i] = (y[i] - s) / U[i][i]

    return x


def cholesky(A): #3
    n = len(A)
    L = np.zeros_like(A)

    for i in range(n):
        for j in range(i+1):
            s = sum(L[i][k] * L[j][k] for k in range(j))
            if (i == j):
                L[i][j] = np.sqrt(A[i][i] - s)
            else:
                L[i][j] = (1.0 / L[j][j] * (A[i][j] - s))
    return L


def llt_solve(A, b): #3
    L = cholesky(A)
    y = np.zeros_like(b)
    x = np.zeros_like(b)
    # Решение Ly = b
    for i in range(len(b)):
        s = sum(L[i][j] * y[j] for j in range(i))
        y[i] = (b[i] - s) / L[i][i]
    # Решение L.T x = y
    for i in range(len(b)-1, -1, -1):
        s = sum(L[j][i] * x[j] for j in range(i+1, len(b)))
        x[i] = (y[i] - s) / L[i][i]

    return x


def sweep(A, b): #4
    n = len(b)
    alpha = np.zeros(n-1)
    beta = np.zeros(n)

    alpha[0] = -A[0][1] / A[0][0]
    beta[0] = b[0] / A[0][0]

    for i in range(1, n-1):
        alpha[i] = -A[i][i+1] / (A[i][i-1]*alpha[i-1] + A[i][i])
        beta[i] = (b[i] - A[i][i-1]*beta[i-1]) / (A[i][i-1]*alpha[i-1] + A[i][i])

    beta[n-1] = (b[n-1] - A[n-1][n-2]*beta[n-2]) / (A[n-1][n-1] + A[n-1][n-2]*alpha[n-2])

    # Вычисляем решение СЛАУ
    # Находим решение методом обратной прогонки
    x = np.zeros(n)
    x[n-1] = beta[n-1]
    for i in range(n-2, -1, -1):
        x[i] = alpha[i]*x[i+1] + beta[i]

    return x


print("Добро пожаловать в Лабораторную работу №2.")
print("Данную работу выполнил Морозов Александр из группы АИ-21-1.")
print("Выберите каким методом хотить решить СЛАУ.\
    \n1 - методом Гауса\n2 - методом LU - разложения\
    \n3 - методом LL(T) разложения\n4 - методом прогонки")
choice = int(input('>> '))

if choice == 1:
    print('Вы выбрали метод Гауса. Теперь нужно выбрать систему. Доступно 3 системы.\n Какую хотите выбрать?')
    table = int(input('>> '))
    if table == 1:
        A = [[5.54, -0.29, 0.26], [0.83, -0.92, 3.11], [-1.51, 12.85, 2.5]]
        B = [-0.56, 4.42, -0.54]
    elif table == 2:
        A = [[9, 3, -15, 9], [3, 5, 5, 5], [-15, 5, 75, 10], [9, 5, 10, 27]]
        B = [54, -68, -595, -228]
    elif table == 3:
        A = [[1, 4, 0, 0], [3, -6, 1, 0], [0, -5, -3, 6], [0, 0, 4, 1]]
        B = [15, -1, -63, 28]
    else:
        print("Такой системы нет.")
        exit(0)
    x = gauss(A, B)
    print(x)
elif choice == 2:
    A = np.array([[9, 3, -15, 9], [3, 5, 5, 5], [-15, 5, 75, 10], [9, 5, 10, 27]])
    B = np.array([54, -68, -595, -228])
    x = lu_solve(A, B)
    print(x)
elif choice == 3:
    A = np.array([[9, 3, -15, 9], [3, 5, 5, 5], [-15, 5, 75, 10], [9, 5, 10, 27]])
    B = np.array([54, -68, -595, -228])
    x = llt_solve(A, B)
    print(x)
elif choice == 4:
    A = np.array([[1, 4, 0, 0], [3, -6, 1, 0], [0, -5, -3, 6], [0, 0, 4, 1]])
    B = np.array([15, -1, -63, 28])
    x = sweep(A, B)
    print(x)
else:
    print("Ошибка ввода, метода под таким числом не существует.")
    exit(0)
