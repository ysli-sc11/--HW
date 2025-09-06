import numpy as np
import matplotlib.pyplot as plt

# 原始函數
def f(x):
    return 1 / (1 + x**2)

# Runge 現象實驗
def runge_demo(n_points=11):
    # 等距節點
    x_nodes = np.linspace(-5, 5, n_points)
    y_nodes = f(x_nodes)

    # Lagrange 多項式插值 (使用 numpy.polyfit)
    coeffs = np.polyfit(x_nodes, y_nodes, n_points-1)
    poly = np.poly1d(coeffs)

    # 繪圖範圍
    x = np.linspace(-5, 5, 500)
    y_true = f(x)
    y_interp = poly(x)

    plt.figure(figsize=(8, 5))
    plt.plot(x, y_true, 'k-', label="True function $1/(1+x^2)$")
    plt.plot(x, y_interp, 'r--', label=f"Interpolated (n={n_points})")
    plt.plot(x_nodes, y_nodes, 'bo', label="Interpolation nodes")
    plt.title("Runge Phenomenon Demonstration")
    plt.legend()
    plt.grid(True)
    plt.show()

# 執行
runge_demo(n_points=11)
