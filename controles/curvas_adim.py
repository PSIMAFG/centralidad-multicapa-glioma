"""Curvas de semejanza en función del acoplamiento adimensional para cada construcción.

Para cada sujeto y construcción: rho de Spearman entre la versatilidad de autovector y la
centralidad de la fMRI sola, de la dMRI sola y de la suma, en una grilla común de
omega / radio espectral medio. Guarda también a qué valor adimensional corresponde omega = 1.
"""

import json
import os
import sys

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import rutas
sys.path.insert(0, rutas.MODULOS)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cargar as C
import multiplex as M
from controles import radio, rescalar

RAIZ = rutas.RAIZ
SALIDA = os.path.dirname(os.path.abspath(__file__))
GRILLA = np.logspace(-3, 2, 51)
CONSTRUCCIONES = {
    "cruda": lambda sc, fc: (sc, fc),
    "reescalada": lambda sc, fc: (rescalar(sc), rescalar(fc)),
    "arbol_ponderado_reescalado": lambda sc, fc: (rescalar(M.arbol_maximo(sc, binario=False)),
                                                  rescalar(M.arbol_maximo(fc, binario=False))),
    "arbol_binarizado": lambda sc, fc: (M.arbol_maximo(sc), M.arbol_maximo(fc)),
}


def main():
    datos = [(s,) + C.cargar_sujeto(RAIZ, s) for s in C.sujetos(RAIZ) if C.cargar_sujeto(RAIZ, s) is not None]
    curvas, omega1 = {}, {}
    for nombre, construir in CONSTRUCCIONES.items():
        fm, dm, su, w1 = [], [], [], []
        for _, sc, fc in datos:
            a, b = construir(sc, fc)
            ref = (radio(a) + radio(b)) / 2.0
            w1.append(1.0 / ref)
            ca, cb, cs = M.centralidad_capa(a), M.centralidad_capa(b), M.centralidad_capa(a + b)
            fila_f, fila_d, fila_s = [], [], []
            for w in GRILLA:
                v = M.versatilidad([a, b], w * ref)
                fila_f.append(spearmanr(v, cb).statistic)
                fila_d.append(spearmanr(v, ca).statistic)
                fila_s.append(spearmanr(v, cs).statistic)
            fm.append(fila_f); dm.append(fila_d); su.append(fila_s)
        curvas[f"{nombre}_fmri"] = np.array(fm)
        curvas[f"{nombre}_dmri"] = np.array(dm)
        curvas[f"{nombre}_suma"] = np.array(su)
        curvas[f"{nombre}_omega1"] = np.array(w1)
        omega1[nombre] = {"mediana": float(np.median(w1)), "min": float(np.min(w1)), "max": float(np.max(w1))}
        print(nombre, "omega = 1 equivale a omega adimensional", omega1[nombre])
    np.savez(os.path.join(SALIDA, "curvas_todas.npz"), grilla=GRILLA, **curvas)
    with open(os.path.join(SALIDA, "curvas_todas.json"), "w", encoding="utf-8") as fh:
        json.dump({"omega1_adimensional": omega1}, fh, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
