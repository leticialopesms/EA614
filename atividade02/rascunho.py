import numpy as np
import matplotlib.pyplot as plt

T = 7
omega_0 = 2 * np.pi / T
a_0 = 0
t = np.linspace(-(T+T/2), (T+T/2), 50_000)

def x_original(t, T):
    t_mod = (t +T/2) % T - T/2
    x = np.zeros_like(t_mod)
    x[(t_mod >= -2) & (t_mod < -1)] = -1
    x[(t_mod >= -1) & (t_mod < 0)] = t_mod[(t_mod >= -1) & (t_mod < 0)] + 1
    x[(t_mod >= 0) & (t_mod < 1)] = t_mod[(t_mod >= 0) & (t_mod < 1)] - 1
    x[(t_mod >= 1) & (t_mod < 2)] = 1
    return x

x_t = x_original(t, T)

def x_fourier(t, N):
    x = a_0
    for k in range(1, N+1):
        a_k = (1/T) * (-2j *(np.sin(k*omega_0) + k*omega_0*np.cos(k*omega_0)*(1-2*np.cos(k*omega_0)))) / (k**2 * omega_0**2)
        x += a_k * np.exp(1j * k * omega_0 * t)
    return x

x1 = x_fourier(t, 1)
x10 = x_fourier(t, 10)
x20 = x_fourier(t, 20)
x50 = x_fourier(t, 50)

config = {
    'axes.spines.right': False,
    'axes.spines.top': False,
    'axes.edgecolor': '.4',
    'axes.labelcolor': '.0',
    'axes.titlesize': 'medium', # ou 'medium', ou 'large'
    'axes.labelsize': 'medium', # ou 'small' ou 'medium'
    'figure.autolayout': True,
    'figure.figsize': (9, 2.5),
    'font.family': ['serif'],
    'font.size': 10.0,
    'grid.linestyle': '--',
    'legend.facecolor': '.9',
    'legend.frameon': True,
    'savefig.transparent': True,
    'text.color': '.0',
    'xtick.labelsize': 'small',
    'ytick.labelsize': 'small',
}

### PROCEDIMENTOS PADRÃO
plt.style.use([config])

# N = 1
plt.plot(t, x_t, color='tab:red', label='$x(t)$')
plt.scatter(x=t, y=x1.real, s=2, color='tab:blue', label='$\widetilde{x}_{1} (t)$ ')
plt.xlabel('$t$')
plt.ylabel('$x(t)$')
plt.grid(True)
plt.legend()
plt.show()

# N = 10
plt.plot(t, x_t, color='tab:red', label='$x(t)$')
plt.scatter(x=t, y=x10.real, s=2, color='tab:blue', label='$\widetilde{x}_{10} (t)$')
plt.xlabel('$t$')
plt.ylabel('$x(t)$')
plt.grid(True)
plt.legend()
plt.show()

# N = 20
plt.plot(t, x_t, color='tab:red', label='$x(t)$')
plt.scatter(x=t, y=x20.real, s=2, color='tab:blue', label='$\widetilde{x}_{20} (t)$')
plt.xlabel('$t$')
plt.ylabel('$x(t)$')
plt.grid(True)
plt.legend()
plt.show()

# N = 50
plt.plot(t, x_t, color='tab:red', label='$x(t)$')
plt.scatter(x=t, y=x50.real, s=2, color='tab:blue', label='$\widetilde{x}_{50} (t)$')
plt.xlabel('$t$')
plt.ylabel('$x(t)$')
plt.grid(True)
plt.legend()
plt.show()

def coef(N):
    a = np.zeros(N+1, dtype=complex)
    a[0] = a_0
    for k in range(1, N+1):
        a[k] = (1/T) * (-2j *(np.sin(k*omega_0) + k*omega_0*np.cos(k*omega_0)*(1-2*np.cos(k*omega_0)))) / (k*omega_0)**2
    return a

N = 50
omega = np.linspace(-(N+1), (N+1), 2*N) * omega_0
a_k = ((1/T) * (-2j *(np.sin(omega) + omega*np.cos(omega)*(1-2*np.cos(omega)))) / (omega)**2)
a_k = np.abs(a_k)

plt.stem(omega, a_k.real, basefmt=" ")
plt.title(r'Módulo dos coeficientes $|a_k|$ em função de $\omega$')
plt.xlabel(r'$\omega = k \cdot \omega_0$')
plt.ylabel(r'$|a_k|$')
plt.grid(True)
plt.show()