import numpy as np
import matplotlib.pyplot as plt

def logistic_euler(h, y0=0.2):
    t = np.arange(0, 10 + h, h)
    y = np.zeros_like(t)
    y[0] = y0
    for n in range(len(t)-1):
        y[n+1] = y[n] + h * y[n] * (1 - y[n])
    return t, y

hs = [0.2, 1.0, 2.1]
y0 = 0.2
t_exact = np.linspace(0, 10, 500)
y_exact = 1 / (1 + ((1/y0) - 1)*np.exp(-t_exact))

plt.figure(figsize=(7,5))
for h in hs:
    t, y = logistic_euler(h, y0)
    plt.plot(t, y, 'o-', label=f'h={h}')
plt.plot(t_exact, y_exact, 'k--', label='Exact')
plt.title("Forward Euler: y' = y(1 - y)")
plt.xlabel('t')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()

'''
y' = f(y) = y(1-y), y(0) = y_{0}, 0 < y_{0} < 1
f'(y) = 1 - 2y
/ y = 0: f'(0) = 1 > 0  
\ y = 1: f'(1) = -1 < 0
Forward Euler: y_{n+1} = y_{n} + h*f(t_{n},y_{n}) = y_{n} + h*y_{n}*(1-y_{n})
|1 + h*f'(1)| < 1 implies |1-h| < 1 implies 0 < h < 2
'''
