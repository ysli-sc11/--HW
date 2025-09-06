import numpy as np
import matplotlib.pyplot as plt

# 目標函數
def f(x):
    return np.sin(x)

# ----------------------
# Newton 差商插值
# ----------------------
def newton_divided_diff(x, y):
    n = len(y)
    coef = np.copy(y).astype(float)
    for j in range(1, n):
        coef[j:n] = (coef[j:n] - coef[j-1:n-1]) / (x[j:n] - x[0:n-j])
    return coef

def newton_poly(coef, x_data, x):
    n = len(coef)
    p = coef[-1]
    for k in range(n-2, -1, -1):
        p = p * (x - x_data[k]) + coef[k]
    return p

# ----------------------
# Lagrange 插值
# ----------------------
def lagrange_poly(x_data, y_data, x):
    total = 0
    n = len(x_data)
    for i in range(n):
        xi, yi = x_data[i], y_data[i]
        Li = 1
        for j in range(n):
            if i != j:
                Li *= (x - x_data[j]) / (xi - x_data[j])
        total += yi * Li
    return total

# ----------------------
# Modified Lagrange (重寫形式)
# ----------------------
def modified_lagrange(x_data, y_data, x):
    n = len(x_data)
    w = np.ones(n)
    for j in range(n):
        for k in range(n):
            if j != k:
                w[j] *= (x_data[j] - x_data[k])
    # Horner-like evaluation
    numerator = 0
    denominator = 0
    for j in range(n):
        term = w[j] / (x - x_data[j])
        numerator += y_data[j] * term
        denominator += term
    return numerator / denominator

# ----------------------
# 主程式
# ----------------------
def interpolation_demo(n_points=10):
    # 隨機節點
    x_nodes = np.sort(np.random.rand(n_points))
    y_nodes = f(x_nodes)

    # Newton
    coef = newton_divided_diff(x_nodes, y_nodes)

    # 繪圖
    x = np.linspace(0, 1, 500)
    y_true = f(x)
    y_newton = newton_poly(coef, x_nodes, x)
    y_lagrange = [lagrange_poly(x_nodes, y_nodes, xi) for xi in x]
    y_modified = [modified_lagrange(x_nodes, y_nodes, xi) for xi in x]

    plt.figure(figsize=(10, 6))
    plt.plot(x, y_true, 'k-', label="True function sin(x)")
    plt.plot(x, y_newton, 'r--', label="Newton")
    plt.plot(x, y_lagrange, 'b-.', label="Lagrange")
    plt.plot(x, y_modified, 'g:', label="Modified Lagrange")
    plt.plot(x_nodes, y_nodes, 'mo', label="Interpolation nodes")
    plt.title(f"Interpolation of sin(x) with {n_points} random points")
    plt.legend()
    plt.grid(True)
    plt.show()

# 執行
interpolation_demo(n_points=10)
