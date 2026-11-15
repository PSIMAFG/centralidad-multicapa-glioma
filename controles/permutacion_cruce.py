"""Control de correspondencia bien planteado: se baraja la correspondencia entre capas y se
compara cada multiplex barajado con la suma de SUS propias capas. Si el cruce se mueve hacia el
de árboles aleatorios, la semejanza estructura-función de las capas reales es lo que adelanta la
convergencia a la suma.

También resume, sujeto a sujeto, el cruce real frente a la mediana de sus árboles nulos
(controles.json), con Wilcoxon pareado. Requiere haber corrido controles.py antes."""

import json
import os
import sys

import numpy as np
from scipy.stats import spearmanr, wilcoxon

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import rutas
sys.path.insert(0, rutas.MODULOS)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cargar as C
import multiplex as M
from controles import OMT, cruce, curva_adim

RAIZ = rutas.RAIZ
K = 8


def aristas(a):
    i, j = np.nonzero(np.triu(a))
    return set(zip(i.tolist(), j.tolist()))


def jaccard(a, b):
    ea, eb = aristas(a), aristas(b)
    return len(ea & eb) / len(ea | eb)


def main():
    rng = np.random.default_rng(20261115)
    datos = [(s,) + C.cargar_sujeto(RAIZ, s) for s in C.sujetos(RAIZ) if C.cargar_sujeto(RAIZ, s) is not None]
    real_c, perm_c, jac_real, jac_perm = [], [], [], []
    for s, sc, fc in datos:
        ab, bb = M.arbol_maximo(sc), M.arbol_maximo(fc)
        f, _ = curva_adim(ab, bb)
        real_c.append(cruce(f, OMT))
        jac_real.append(jaccard(ab, bb))
        cs_sujeto, js = [], []
        for _ in range(K):
            p = rng.permutation(ab.shape[0])
            bp = bb[np.ix_(p, p)]
            f, _ = curva_adim(ab, bp)
            cs_sujeto.append(cruce(f, OMT))
            js.append(jaccard(ab, bp))
        perm_c.append(float(np.median(cs_sujeto)))
        jac_perm.append(float(np.mean(js)))
    real_c, perm_c = np.array(real_c), np.array(perm_c)
    w = wilcoxon(real_c, perm_c)
    res = {"cruce_real_mediana": float(np.median(real_c)), "cruce_barajado_mediana": float(np.median(perm_c)),
           "sujetos_real_antes_que_barajado": int(np.sum(real_c < perm_c)), "wilcoxon_W": float(w.statistic),
           "wilcoxon_p": float(w.pvalue), "jaccard_real_media": float(np.mean(jac_real)),
           "jaccard_barajado_media": float(np.mean(jac_perm)),
           "cruce_real_por_sujeto": real_c.tolist(), "cruce_barajado_por_sujeto": perm_c.tolist(),
           "sujetos": [d[0] for d in datos]}
    ctl = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "controles.json"), encoding="utf-8"))
    for clave in ("nulo_mismos_grados", "nulo_uniforme"):
        nul = np.median(np.array(ctl["cruces_por_sujeto"][clave]).reshape(len(real_c), -1), axis=1)
        wn = wilcoxon(real_c, nul)
        res[f"pareado_{clave}"] = {"mediana_de_medianas_nulas": float(np.median(nul)),
                                   "sujetos_real_antes": int(np.sum(real_c < nul)),
                                   "wilcoxon_W": float(wn.statistic), "wilcoxon_p": float(wn.pvalue)}
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "permutacion_cruce.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1)
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
