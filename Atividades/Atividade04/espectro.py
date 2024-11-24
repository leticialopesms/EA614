import math
import numpy as np
import matplotlib.pyplot as plt

def espectro(y):
    '''
    Rotina que exibe o espectro de magnitude (X(ejw)) de um sinal discreto.
    Args:
        y: Sinal discreto
    Returns:
        Y: Modulo da transformada de Fourier
        w: Frequencias avaliadas
    '''
    # Modulo da transformada de Fourier
    Y = np.abs(np.fft.fft(y))
    # Frequencias avaliadas
    w = np.linspace(0,2*math.pi,Y.size)

    # Exibe o grafico do espectro
    plt.figure() 
    plt.plot(w,Y)
    plt.xlabel('$\Omega$ [rad]', fontsize=14)
    plt.ylabel('|$Y(e^{j\Omega})$|', fontsize=14)
    plt.grid(True)
    plt.xlim((0,2*math.pi))
    plt.show()

    return Y,w