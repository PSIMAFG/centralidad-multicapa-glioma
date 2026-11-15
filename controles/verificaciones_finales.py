"""Dos verificaciones pedidas al cerrar la discusión.

1. Desglose de las aristas compartidas por categoría anatómica, y composición de cada árbol.
2. Definición del primer cruce: barrido denso (400 puntos) para comprobar que la grilla gruesa
   con bisección no omite un cruce anterior seguido de un descenso, en los 36 reales y en una
   muestra de nulos.
"""

import json
import os
import sys

import numpy as np
from scipy.stats import rankdata

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import rutas
sys.path.insert(0, rutas.MODULOS)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cargar as C
import multiplex as M
from controles import arbol_a_prufer, prufer_a_arbol, radio
from explicacion_espectral import centralidad, cruce, rho
from validacion_P import categorias, regiones

RAIZ = rutas.RAIZ
SALIDA = os.path.dirname(os.path.abspath(__file__))
DENSA = np.logspace(-2.5, 0.7, 400)


def barrido(a, b):
    ref = (radio(a) + radio(b)) / 2.0
    rs = rankdata(centralidad(a + b))
    f = np.array([rho(a, b, g * ref, rs) for g in DENSA])
    sobre = np.where(f >= 0.95)[0]
    if len(sobre) == 0:
        return np.inf, False
    k = sobre[0]
    vuelve_a_bajar = bool(np.any(f[k:] < 0.95))
    if k == 0:
        return float(DENSA[0]), vuelve_a_bajar
    t = (0.95 - f[k - 1]) / (f[k] - f[k - 1])
    return float(10 ** (np.log10(DENSA[k - 1]) + t * (np.log10(DENSA[k]) - np.log10(DENSA[k - 1])))), vuelve_a_bajar


def main():
    rng = np.random.default_rng(20261119)
    cat = categorias(regiones())
    iu = np.triu_indices(68, 1)
    c = cat[iu]
    claves = ["mismo_lobulo", "distinto_lobulo", "homotopica", "otra_interhemisferica"]
    comp, cap_a, cap_b = {k: [] for k in claves}, {k: [] for k in claves}, {k: [] for k in claves}
    filas = []
    for s in C.sujetos(RAIZ):
        par = C.cargar_sujeto(RAIZ, s)
        if par is None:
            continue
        a, b = M.arbol_maximo(par[0]), M.arbol_maximo(par[1])
        ea, eb = a[iu] > 0, b[iu] > 0
        for k in claves:
            m = c == k
            comp[k].append(int((ea & eb & m).sum()))
            cap_a[k].append(int((ea & m).sum()))
            cap_b[k].append(int((eb & m).sum()))
        real_b = cruce(a, b)
        real_d, baja = barrido(a, b)
        fila = {"sujeto": s, "real_biseccion": real_b, "real_densa": real_d, "real_vuelve_a_bajar": baja, "nulos": []}
        (pa, nda), (pb, ndb) = arbol_a_prufer(a), arbol_a_prufer(b)
        for _ in range(4):
            na, nb = prufer_a_arbol(rng.permutation(pa), nda, 68), prufer_a_arbol(rng.permutation(pb), ndb, 68)
            p = rng.permutation(68)
            for x, y in ((na, nb), (a, b[np.ix_(p, p)])):
                d, v = barrido(x, y)
                fila["nulos"].append((cruce(x, y), d, v))
        filas.append(fila)
    rb = np.array([f["real_biseccion"] for f in filas])
    rd = np.array([f["real_densa"] for f in filas])
    nul = np.array([n for f in filas for n in f["nulos"]], dtype=object)
    nb_, nd_ = nul[:, 0].astype(float), nul[:, 1].astype(float)
    res = {
        "aristas_compartidas_por_categoria_media": {k: float(np.mean(v)) for k, v in comp.items()},
        "aristas_por_categoria_dmri_media": {k: float(np.mean(v)) for k, v in cap_a.items()},
        "aristas_por_categoria_fmri_media": {k: float(np.mean(v)) for k, v in cap_b.items()},
        "candidatas_por_categoria": {k: int((c == k).sum()) for k in claves},
        "cruce_reales": {"diferencia_relativa_max": float(np.max(np.abs(rb / rd - 1))),
                         "diferencia_relativa_mediana": float(np.median(np.abs(rb / rd - 1))),
                         "sujetos_con_rho_que_vuelve_bajo_0,95": int(sum(f["real_vuelve_a_bajar"] for f in filas))},
        "cruce_nulos_muestra": {"n": int(len(nul)),
                                "diferencia_relativa_mediana": float(np.median(np.abs(nb_ / nd_ - 1))),
                                "fraccion_con_diferencia_mayor_10_pct": float(np.mean(np.abs(nb_ / nd_ - 1) > 0.10)),
                                "fraccion_que_vuelve_bajo_0,95": float(np.mean(nul[:, 2].astype(bool)))},
    }
    with open(os.path.join(SALIDA, "verificaciones_finales.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1)
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
