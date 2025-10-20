import numpy as np
import matplotlib.pyplot as plt

# ====== Forward Euler: y' = -5y ======
'''
y(t) = exp(-5t)
y_{n+1} = y_{n} + hf(t_{n},y_{n}) = y-{n}-5y_{n}h = (1-5h)y_{n}
|1-5h| < 1 implies 0 < h < 0.4
'''
def forward_euler(h):
    t = np.arange(0, 10 + h, h)
    y = np.zeros_like(t)
    y[0] = 1 #y(0) = 1
    for n in range(len(t)-1):
        y[n+1] = y[n] + h * (-5 * y[n]) 
    return t, y

# ====== Backward Euler: y' = 5y ======
'''
y(t) = exp(5t)
y_{n+1} = y_{n} + hf(t_{n+1},y_{n+1})
y_{n+1} = \frac{y_{n}}{1-5h}
'''
def backward_euler(h):
    t = np.arange(0, 10 + h, h)
    y = np.zeros_like(t)
    y[0] = 1 #y(0) = 1
    for n in range(len(t)-1):
        y[n+1] = y[n] / (1 - 5*h) 
    return t, y

# ====== Exact solutions ======
t_exact = np.linspace(0, 10, 500)
y_exact_a = np.exp(-5*t_exact) 
y_exact_b = np.exp(5*t_exact) 

# ====== Simulations ======
hs = [0.1, 0.4, 0.41]

plt.figure(figsize=(12,10))

# ---- Forward Euler results ----
plt.subplot(2,2,1)
for h in hs:
    t, y = forward_euler(h)
    plt.plot(t, y, 'o-', label=f'h={h}')
plt.plot(t_exact, y_exact_a, 'k--', label='Exact')
plt.title("Forward Euler: y' = -5y")
plt.xlabel('t')
plt.ylabel('y')
plt.legend()
plt.grid(True)

# ---- Forward Euler errors ----
plt.subplot(2,2,2)
for h in hs:
    t, y = forward_euler(h)
    y_true = np.exp(-5*t)
    error = np.abs(y - y_true)
    plt.plot(t, error, 'o-', label=f'h={h}')
plt.title("Forward Euler Error: |y_n - y(t)|")
plt.xlabel('t')
plt.ylabel('Error')
plt.legend()
plt.grid(True)

# ---- Backward Euler results ----
plt.subplot(2,2,3)
for h in hs:
    t, y = backward_euler(h)
    plt.plot(t, y, 'o-', label=f'h={h}')
plt.plot(t_exact, y_exact_b, 'k--', label='Exact')
plt.title("Backward Euler: y' = 5y")
plt.xlabel('t')
plt.ylabel('y')
plt.legend()
plt.grid(True)

# ---- Backward Euler errors ----
plt.subplot(2,2,4)
for h in hs:
    t, y = backward_euler(h)
    y_true = np.exp(5*t)
    error = np.abs(y - y_true)
    plt.plot(t, error, 'o-', label=f'h={h}')
plt.title("Backward Euler Error: |y_n - y(t)|")
plt.xlabel('t')
plt.ylabel('Error')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
