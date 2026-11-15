"""Construcción del multiplex estructura-función y sus métricas.

Replica el procedimiento de Breedt y colaboradores: cada capa se reduce a su árbol
de expansión de peso máximo, las capas se acoplan en una matriz de supra-adyacencia
con peso intercapa omega, y la métrica central es la versatilidad de autovector.
"""

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree


def arbol_maximo(w, binario=True):
    """Árbol de expansión de peso máximo.

    En conectómica se conserva la red de enlaces más fuertes que mantiene todo
    conectado, así que se calcula el árbol mínimo sobre los pesos negados. El
    resultado tiene exactamente N-1 enlaces, sin importar la densidad original.
    """
    w = np.asarray(w, dtype=float)
    n = w.shape[0]
    coste = np.where(w > 0, -w, 0.0)
    t = minimum_spanning_tree(csr_matrix(coste)).toarray()
    a = np.zeros((n, n))
    filas, cols = np.nonzero(t)
    for i, j in zip(filas, cols):
        valor = 1.0 if binario else w[i, j]
        a[i, j] = a[j, i] = valor
    return a


def supra_adyacencia(capas, omega):
    """Matriz de supra-adyacencia de tamaño LN x LN.

    Los bloques de la diagonal son las capas. Los bloques exteriores llevan omega
    en su diagonal, que es el enlace de cada región consigo misma entre capas.
    """
    ele = len(capas)
    n = capas[0].shape[0]
    s = np.zeros((ele * n, ele * n))
    for a in range(ele):
        s[a * n:(a + 1) * n, a * n:(a + 1) * n] = capas[a]
    if omega != 0:
        ident = np.eye(n) * float(omega)
        for a in range(ele):
            for b in range(ele):
                if a != b:
                    s[a * n:(a + 1) * n, b * n:(b + 1) * n] = ident
    return s


def supra_permutada(capas, omega, perm):
    """Supra-adyacencia donde la correspondencia entre capas se rompe.

    Es el modelo nulo: la región i de la capa 1 se enlaza con la región perm[i] de
    la capa 2 en vez de consigo misma. Todo lo demás (densidad, pesos, estructura
    de cada capa) queda igual, de modo que lo único que cambia es si la identidad
    de la región a través de las capas significa algo.
    """
    n = capas[0].shape[0]
    s = supra_adyacencia(capas, 0.0)
    p = np.zeros((n, n))
    p[np.arange(n), perm] = float(omega)
    s[:n, n:] = p
    s[n:, :n] = p.T
    return s


def _autovector_principal(m):
    """Autovector del mayor autovalor, con signo fijado y norma unitaria."""
    valores, vectores = np.linalg.eigh(m)
    v = np.abs(vectores[:, int(np.argmax(valores))])
    s = v.sum()
    return v / s if s > 0 else v


def versatilidad(capas, omega):
    """Versatilidad de autovector: centralidad sobre la supra-adyacencia, agregada
    por región a través de las capas."""
    ele = len(capas)
    n = capas[0].shape[0]
    v = _autovector_principal(supra_adyacencia(capas, omega))
    return v.reshape(ele, n).sum(axis=0)


def versatilidad_nula(capas, omega, perm):
    """Versatilidad con la correspondencia entre capas permutada.

    Para que la comparación sea justa, el valor de cada región se reconstruye
    deshaciendo la permutación: la región i recibe su parte de la capa 1 y la
    parte que le toco en la capa 2.
    """
    n = capas[0].shape[0]
    v = _autovector_principal(supra_permutada(capas, omega, perm))
    v1, v2 = v[:n], v[n:]
    inversa = np.empty(n, dtype=int)
    inversa[perm] = np.arange(n)
    return v1 + v2[perm]


def centralidad_capa(a):
    """Centralidad de autovector de una sola capa."""
    return _autovector_principal(np.asarray(a, dtype=float))


def grado_solapado(capas):
    """Suma del grado de cada región a través de las capas."""
    return np.sum([np.asarray(a).sum(axis=1) for a in capas], axis=0)


def ruido_simetrico(n, sigma, rng):
    """Matriz simétrica con un valor gaussiano independiente N(0, sigma²) por arista no dirigida.

    Se sortea un valor por par (i, j) con i < j y se copia en (j, i). Promediar una matriz de ruido
    con su transpuesta daría una desviación estándar efectiva de sigma / raíz de 2.
    """
    i, j = np.triu_indices(n, 1)
    z = np.zeros((n, n))
    z[i, j] = rng.normal(0.0, sigma, len(i))
    return z + z.T


def perturbar(w, sigma, rng, modo="multiplicativo"):
    """Perturba los pesos de una red.

    El modo multiplicativo escala el ruido al peso de cada enlace, que es el modelo
    realista del error de medición en tractografía: un enlace fuerte se mide con un
    error absoluto mayor que uno débil. Es también un test más exigente que escalar
    el ruido a un valor único de la red, porque perturba de verdad los enlaces que
    forman el árbol.

    El modo aditivo escala el ruido al percentil 90 de los pesos no nulos, no a la
    mediana, por la misma razón: en una distribución de cola pesada la mediana deja
    casi intactos los enlaces que importan.

    Cada arista no dirigida recibe un solo valor de ruido con desviación estándar sigma
    (relativa al peso en el modo multiplicativo). Los pesos negativos no se perturban y,
    como los resultantes, se recortan a cero.
    """
    w = np.asarray(w, dtype=float)
    nz = w[w > 0]
    if nz.size == 0:
        return w.copy()
    escala = 1.0 if modo == "multiplicativo" else float(np.percentile(nz, 90))
    ruido = ruido_simetrico(w.shape[0], sigma * escala, rng)
    if modo == "multiplicativo":
        p = w * (1.0 + np.where(w > 0, ruido, 0.0))
    else:
        p = w + np.where(w > 0, ruido, 0.0)
    p = np.clip(p, 0.0, None)
    np.fill_diagonal(p, 0.0)
    return p


def enlaces(a):
    """Conjunto de enlaces (i, j) con i < j de una matriz de adyacencia."""
    i, j = np.nonzero(np.triu(a, 1))
    return set(zip(i.tolist(), j.tolist()))


def jaccard(a, b):
    """Solapamiento de Jaccard entre los enlaces de dos árboles."""
    ea, eb = enlaces(a), enlaces(b)
    if not ea and not eb:
        return 1.0
    return len(ea & eb) / float(len(ea | eb))
