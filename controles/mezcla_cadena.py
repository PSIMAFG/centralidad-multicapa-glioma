"""Diagnóstico de mezcla de la cadena de Markov del nulo de grados y distancia.

Para cada sujeto y capa se vuelve a correr la cadena con el beta ya calibrado (fijo, guardado en
nulo_grados_distancia.json): calentamiento y luego muestreo con el mismo número de pasos que el
análisis principal. Se registra cada 10 pasos el estadístico que importa para el resultado: cuántas
aristas del árbol nulo coinciden con el árbol real de la otra capa. Con esa serie se estima el tiempo
de autocorrelación integrado (ventana automática de Sokal) y el tamaño efectivo de muestra.

Además se mide, sin volver a correr las cadenas, el tamaño efectivo de muestra del cruce y de P en
los 100 pares nulos que usa el análisis principal.

También se comprueba la simetría de la propuesta: la cadena elige dos aristas al azar y orientaciones
al azar antes de descartar las propuestas inválidas, y la propuesta inversa tiene la misma
probabilidad, así que no hace falta corrección de Hastings. La prueba con enumeración exacta está en
pruebas/test_metodos.py.
"""

import json
import os
import random
import sys
from multiprocessing import Pool

import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(AQUI))
import rutas
sys.path.insert(0, rutas.MODULOS)
sys.path.insert(0, AQUI)
sys.path.insert(0, os.path.join(AQUI, "..", "regional"))
import cargar as C
import multiplex as M
from lectura import leer_centros
from nulo_grados_distancia import CADA, CALENTAMIENTO, MUESTREO, SEMILLA, Cadena, aristas_de

RAIZ = rutas.RAIZ


def tiempo_autocorrelacion(x, c=5.0):
    """Tiempo de autocorrelación integrado con la ventana automática de Sokal."""
    x = np.asarray(x, dtype=float) - np.mean(x)
    n = len(x)
    if np.allclose(x, 0):
        return 1.0
    f = np.fft.rfft(x, 2 * n)
    acf = np.fft.irfft(f * np.conjugate(f))[:n]
    acf = acf / acf[0]
    tau = 1.0
    for m in range(1, n):
        tau = 1.0 + 2.0 * acf[1:m + 1].sum()
        if m >= c * tau:
            break
    return max(tau, 1.0)


def serie(arbol, otro, dist, beta, semilla):
    otro_set = set(aristas_de(otro))
    cad = Cadena(aristas_de(arbol), arbol.shape[0], dist, [], random.Random(semilla))
    for _ in range(CALENTAMIENTO):
        cad.paso(beta)
    valores = []
    for t in range(MUESTREO):
        cad.paso(beta)
        if t % CADA == 0:
            valores.append(sum(1 for e in cad.e if e in otro_set))
    return np.array(valores)


def por_sujeto(args):
    k, s, sc, fc, beta = args
    _, xyz = leer_centros(s)
    dist = np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=2).tolist()
    a, b = M.arbol_maximo(sc), M.arbol_maximo(fc)
    sa = serie(a, b, dist, beta["dmri"], SEMILLA * 1000 + k * 10 + 1)
    sb = serie(b, a, dist, beta["fmri"], SEMILLA * 1000 + k * 10 + 2)
    ta, tb = tiempo_autocorrelacion(sa), tiempo_autocorrelacion(sb)
    return {"sujeto": s, "muestras": len(sa),
            "tau_dmri": ta, "ess_dmri": len(sa) / ta, "tau_fmri": tb, "ess_fmri": len(sb) / tb,
            "media_dmri_vs_fmri_real": float(sa.mean()), "media_fmri_vs_dmri_real": float(sb.mean())}


def ess_entre_pares():
    """Tamaño efectivo de muestra del cruce y de P en los 100 pares nulos de cada sujeto.

    Los pares se toman de las cadenas cada MUESTREO / 100 pasos y se guardan en ese orden en
    nulo_grados_distancia.npz, así que la autocorrelación entre pares consecutivos mide si los
    pares usados para la mediana del cruce y de P son casi independientes.
    """
    g = np.load(os.path.join(AQUI, "nulo_grados_distancia.npz"))
    salida = {}
    for clave, nombre in (("cruces_nulos", "cruce"), ("P_nulos", "P")):
        x = np.log(g[clave])
        ess = np.array([len(f) / tiempo_autocorrelacion(f) for f in x])
        r1 = np.array([np.corrcoef(f[:-1], f[1:])[0, 1] for f in x])
        salida[nombre] = {"pares_por_sujeto": int(x.shape[1]), "ess_mediana": float(np.median(ess)),
                          "ess_minimo": float(ess.min()), "autocorrelacion_lag1_mediana": float(np.median(r1)),
                          "autocorrelacion_lag1_maxima": float(r1.max())}
    return salida


def main():
    gd = json.load(open(os.path.join(AQUI, "nulo_grados_distancia.json"), encoding="utf-8"))
    betas = {f["sujeto"]: f["beta"] for f in gd["por_sujeto"]}
    tareas = []
    for k, s in enumerate(C.sujetos(RAIZ)):
        par = C.cargar_sujeto(RAIZ, s)
        if par is not None:
            tareas.append((k, s, par[0], par[1], betas[s]))
    with Pool(int(os.environ.get("PROCESOS", "2"))) as pool:
        filas = pool.map(por_sujeto, tareas)
    ess_a = np.array([f["ess_dmri"] for f in filas])
    ess_b = np.array([f["ess_fmri"] for f in filas])
    res = {
        "muestras_por_cadena": filas[0]["muestras"], "pasos_entre_muestras": CADA,
        "ess_dmri": {"mediana": float(np.median(ess_a)), "minimo": float(ess_a.min())},
        "ess_fmri": {"mediana": float(np.median(ess_b)), "minimo": float(ess_b.min())},
        "tau_pasos_dmri_mediana": float(np.median([f["tau_dmri"] for f in filas]) * CADA),
        "tau_pasos_fmri_mediana": float(np.median([f["tau_fmri"] for f in filas]) * CADA),
        "propuesta_simetrica": True,
        "beta_fijo_durante_el_muestreo": True,
        "ess_entre_pares_nulos": ess_entre_pares(),
        "por_sujeto": filas,
    }
    with open(os.path.join(AQUI, "mezcla_cadena.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "por_sujeto"}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
