import numpy as np

def calcula_coeficientes(w,wc,n):
    """ Rotina que calcula os coeficientes do polinômio de Chebyshev de maneira não-recursiva 

    Parâmetros: w - vetor de frequências (sugestão: usar um vetor com amostras de 0 a 20 rad/s)
                wc - freq. de corte do filtro (em rad/s)
                n - ordem do filtro de Chebyshev
    Saída:      Tn - vetor com os coeficientes calculados do polinômio de Chebyshev (possui o mesmo tamanho que w)

    """
    Tn = np.zeros((w.size,))
    #determina os valores dos coeficientes segundo as expressões padronizadas
    Tn[abs(w) < wc] = np.cos(n*np.arccos(w[abs(w) < wc] / wc))
    Tn[abs(w) >= wc] = np.cosh(n*np.arccosh(w[abs(w) >= wc] / wc))
    return Tn