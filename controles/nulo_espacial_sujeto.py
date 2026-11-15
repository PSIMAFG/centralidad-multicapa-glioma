"""Nulo espacial con los centroides de cada sujeto (centres.txt de SC.zip, OpenNeuro ds001226).

Reemplaza la versión con centroides de plantilla (nulo_espacial_plantilla.py). Para cada
sujeto y capa se generan árboles aleatorios sesgados hacia aristas cortas: árbol de expansión
máxima sobre E_ij · exp(-d_ij / lambda), con E_ij exponencial y d_ij la distancia euclídea
entre los centroides de ese sujeto. Lambda se calibra por capa para que la longitud media de
las aristas del árbol nulo iguale la del árbol real. Luego se cuentan las aristas compartidas
entre árboles nulos independientes de las dos capas y se comparan con las reales.

SC.zip no trae longitudes de tractos (tract_lengths.txt viene vacío), así que la distancia es
euclídea. El nulo no conserva la secuencia de grados. Como el nulo de distancia casi no genera
aristas homotópicas (están lejos) y el árbol fMRI real tiene unas 20, calibrar con la longitud
media de todas sus aristas alarga las demás aristas del nulo fMRI. Por eso también se informa
una variante que fija las aristas homotópicas del árbol fMRI real y sortea el resto, con lambda
calibrado para que esas aristas libres tengan la longitud media de las no homotópicas reales.
"""

import json
import os
import sys

import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.stats import wilcoxon

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import rutas
sys.path.insert(0, rutas.MODULOS)
sys.path.insert(0, AQUI)
sys.path.insert(0, os.path.join(AQUI, "..", "regional"))
import cargar as C
import multiplex as M
from lectura import leer_centros
from nulo_espacial_plantilla import arbol_geometrico, calibrar, longitud_media
from validacion_P import categorias, regiones

RAIZ = rutas.RAIZ
K = 200
SEMILLA = 20261121


def compartidas(a, b):
    return int(((np.triu(a, 1) > 0) & (np.triu(b, 1) > 0)).sum())


def arbol_con_fijas(d, lam, fijas, rng):
    """Árbol sesgado por distancia que contiene las aristas fijas (sin ciclos entre ellas)."""
    n = d.shape[0]
    e = rng.exponential(1.0, (n, n))
    w = np.triu(e * np.exp(-d / lam), 1)
    w = w + w.T
    w[fijas > 0] = w.max() * 10.0 + 1.0
    t = minimum_spanning_tree(-np.triu(w, 1)).toarray()
    return ((t + t.T) != 0).astype(float)


def longitud_libres(a, fijas, d):
    i, j = np.nonzero(np.triu(a * (fijas == 0), 1))
    return float(d[i, j].mean())


def calibrar_fijas(objetivo, d, fijas, rng, n_arboles=12):
    """Lambda para que las aristas no fijas del nulo tengan la longitud media de las reales."""
    lo, hi = 1.0, 400.0
    for _ in range(18):
        mid = np.sqrt(lo * hi)
        lm = np.mean([longitud_libres(arbol_con_fijas(d, mid, fijas, rng), fijas, d) for _ in range(n_arboles)])
        if lm > objetivo:
            hi = mid
        else:
            lo = mid
    return np.sqrt(lo * hi)


def main():
    rng = np.random.default_rng(SEMILLA)
    nombres = regiones()
    cat = categorias(nombres)
    homo = (cat == "homotopica").astype(float)
    filas = []
    for s in C.sujetos(RAIZ):
        par = C.cargar_sujeto(RAIZ, s)
        if par is None:
            continue
        _, xyz = leer_centros(s)
        d = np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=2)
        a, b = M.arbol_maximo(par[0]), M.arbol_maximo(par[1])
        la, lb = calibrar(longitud_media(a, d), d, rng), calibrar(longitud_media(b, d), d, rng)
        nulos = [compartidas(arbol_geometrico(d, la, rng), arbol_geometrico(d, lb, rng)) for _ in range(K)]
        fijas = b * homo
        lh = calibrar_fijas(longitud_libres(b, fijas, d), d, fijas, rng)
        nulos_h = [compartidas(arbol_geometrico(d, la, rng), arbol_con_fijas(d, lh, fijas, rng)) for _ in range(K)]
        nb = arbol_geometrico(d, lb, rng)
        i, j = np.nonzero(np.triu(a * b, 1))
        filas.append({"sujeto": s, "reales": compartidas(a, b), "nulo": float(np.mean(nulos)),
                      "nulo_p95": float(np.percentile(nulos, 95)), "nulo_homotopicas_fijas": float(np.mean(nulos_h)),
                      "longitud_dmri": longitud_media(a, d), "longitud_fmri": longitud_media(b, d),
                      "longitud_compartidas": float(d[i, j].mean()) if len(i) else float("nan"),
                      "homotopicas_fmri_real": int(np.triu(fijas, 1).sum()),
                      "homotopicas_fmri_nulo": int(np.triu(nb * homo, 1).sum())})
    reales = np.array([f["reales"] for f in filas], dtype=float)
    nulo = np.array([f["nulo"] for f in filas])
    nulo_h = np.array([f["nulo_homotopicas_fijas"] for f in filas])
    p95 = np.array([f["nulo_p95"] for f in filas])
    plantilla = json.load(open(os.path.join(AQUI, "nulo_espacial_plantilla.json"), encoding="utf-8"))
    res = {
        "sujetos": len(filas), "K": K,
        "aristas_compartidas_reales_media": float(reales.mean()),
        "nulo_distancia_sujeto_media": float(nulo.mean()),
        "razon_real_sobre_nulo_mediana": float(np.median(reales / nulo)),
        "sujetos_real_mayor": int(np.sum(reales > nulo)),
        "sujetos_real_sobre_p95_del_nulo": int(np.sum(reales > p95)),
        "wilcoxon_p": float(wilcoxon(reales - nulo).pvalue),
        "nulo_homotopicas_fijas_media": float(nulo_h.mean()),
        "razon_real_sobre_nulo_homotopicas_fijas_mediana": float(np.median(reales / nulo_h)),
        "sujetos_real_mayor_homotopicas_fijas": int(np.sum(reales > nulo_h)),
        "wilcoxon_homotopicas_fijas_p": float(wilcoxon(reales - nulo_h).pvalue),
        "longitud_media_mm": {"dmri": float(np.mean([f["longitud_dmri"] for f in filas])),
                              "fmri": float(np.mean([f["longitud_fmri"] for f in filas])),
                              "aristas_compartidas": float(np.nanmean([f["longitud_compartidas"] for f in filas]))},
        "homotopicas_fmri": {"real": float(np.mean([f["homotopicas_fmri_real"] for f in filas])),
                             "nulo_distancia": float(np.mean([f["homotopicas_fmri_nulo"] for f in filas]))},
        "comparacion_plantilla": {"nulo_plantilla_media": plantilla["aristas_compartidas_nulo_espacial_media"],
                                  "razon_plantilla_mediana": plantilla["razon_real_sobre_nulo_espacial_mediana"]},
        "por_sujeto": filas,
    }
    with open(os.path.join(AQUI, "nulo_espacial_sujeto.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "por_sujeto"}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
