"""La heterogeneidad individual del cruce, ¿la explica la semejanza estructura-función de cada sujeto?"""
import json
import os
import sys

import numpy as np
from scipy.stats import spearmanr, kruskal

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import rutas
sys.path.insert(0, rutas.MODULOS)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cargar as C
import multiplex as M
from controles import OMT, cruce, curva_adim
from permutacion_cruce import jaccard  # noqa: E402  (solo la función)

RAIZ = rutas.RAIZ
grupos = C.grupos(RAIZ)
filas = []
for s in C.sujetos(RAIZ):
    par = C.cargar_sujeto(RAIZ, s)
    if par is None:
        continue
    sc, fc = par
    ab, bb = M.arbol_maximo(sc), M.arbol_maximo(fc)
    f, ref = curva_adim(ab, bb)
    i, j = np.triu_indices(68, 1)
    filas.append({"sujeto": s, "grupo": grupos.get(s), "cruce": cruce(f, OMT), "jaccard": jaccard(ab, bb),
                  "sfc": float(spearmanr(sc[i, j], fc[i, j]).statistic)})
cr = np.array([f["cruce"] for f in filas]); ja = np.array([f["jaccard"] for f in filas]); sf = np.array([f["sfc"] for f in filas])
r1 = spearmanr(cr, ja); r2 = spearmanr(cr, sf)
gs = {g: cr[[f["grupo"] == g for f in filas]] for g in ("glioma", "meningioma", "control")}
kw = kruskal(*gs.values())
res = {"rho_cruce_jaccard": float(r1.statistic), "p_cruce_jaccard": float(r1.pvalue),
       "rho_cruce_acoplamiento": float(r2.statistic), "p_cruce_acoplamiento": float(r2.pvalue),
       "cruce_por_grupo_mediana": {g: float(np.median(v)) for g, v in gs.items()}, "kruskal_p": float(kw.pvalue)}
json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "heterogeneidad.json"), "w"), indent=1)
print(json.dumps(res, indent=1))
