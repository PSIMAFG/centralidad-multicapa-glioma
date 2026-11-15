"""Sensibilidad del cruce al umbral de semejanza y localización del autovector.

Para cada sujeto se calcula la curva completa de ρ (Spearman entre la versatilidad y la centralidad
de la suma) en una grilla fina de acoplamiento adimensional, para el par real de árboles binarizados
y para K pares nulos de cada tipo topológico (mismo grado por región, uniformes y correspondencia
barajada). De la misma curva se leen los cruces con umbrales 0,90, 0,95 y 0,99, y se compara el real
con la mediana de sus nulos. Como medida continua, sin umbral, se usa el área bajo 1 − ρ en la grilla
(en décadas de acoplamiento): cuanto menor, antes se parece la centralidad a la de la suma. También se
mide la localización del autovector principal de cada capa y de la suma (razón de participación
inversa), porque en árboles la centralidad de autovector puede concentrarse en pocos nodos.
"""

import json
import os
import sys
from multiprocessing import Pool

import numpy as np
import scipy.linalg as sl
from scipy.stats import rankdata, spearmanr, wilcoxon

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(AQUI))
import rutas
sys.path.insert(0, rutas.MODULOS)
sys.path.insert(0, AQUI)
import cargar as C
import multiplex as M
from controles import arbol_a_prufer, prufer_a_arbol, radio

RAIZ = rutas.RAIZ
K = 100
SEMILLA = 20261114
GRILLA = np.logspace(-2.5, 1.5, 81)
UMBRALES = (0.90, 0.95, 0.99)
TIPOS = ("mismo_grado", "uniforme", "barajada")


def autovector(m):
    v = np.abs(sl.eigh(m, subset_by_index=[m.shape[0] - 1, m.shape[0] - 1], driver="evr")[1][:, 0])
    return v / np.linalg.norm(v)


def curva(a, b):
    n = a.shape[0]
    ref = (radio(a) + radio(b)) / 2.0
    rs = rankdata(autovector(a + b))
    salida = np.empty(len(GRILLA))
    for k, g in enumerate(GRILLA):
        w = g * ref
        x = autovector(np.block([[a, w * np.eye(n)], [w * np.eye(n), b]]))
        salida[k] = np.corrcoef(rankdata(x[:n] + x[n:]), rs)[0, 1]
    return salida


def primer_cruce(rho, umbral):
    """Primer acoplamiento de la grilla en que ρ alcanza el umbral, interpolado en escala logarítmica."""
    k = np.flatnonzero(rho >= umbral)
    if len(k) == 0:
        return float(GRILLA[-1] * 10)
    k = k[0]
    if k == 0:
        return float(GRILLA[0])
    lo, hi = np.log10(GRILLA[k - 1]), np.log10(GRILLA[k])
    t = (umbral - rho[k - 1]) / (rho[k] - rho[k - 1])
    return float(10 ** (lo + t * (hi - lo)))


def area(rho):
    """Área bajo 1 − ρ a lo largo de la grilla, en décadas de acoplamiento adimensional."""
    return float(np.trapezoid(1.0 - rho, np.log10(GRILLA)))


def ipr(v):
    return float(np.sum(v ** 4))


def por_sujeto(args):
    indice, s, sc, fc = args
    rng = np.random.default_rng([SEMILLA, indice])
    n = sc.shape[0]
    ab, bb = M.arbol_maximo(sc), M.arbol_maximo(fc)
    cr = curva(ab, bb)
    fila = {"sujeto": s, "real": {str(u): primer_cruce(cr, u) for u in UMBRALES}, "area_real": area(cr),
            "ipr_real": {"dmri": ipr(autovector(ab)), "fmri": ipr(autovector(bb)), "suma": ipr(autovector(ab + bb))}}
    (pa, nda), (pb, ndb) = arbol_a_prufer(ab), arbol_a_prufer(bb)
    nulos = {t: {str(u): [] for u in UMBRALES} for t in TIPOS}
    areas = {t: [] for t in TIPOS}
    ipr_nulos = {t: [] for t in TIPOS}
    for _ in range(K):
        pares = {
            "mismo_grado": (prufer_a_arbol(rng.permutation(pa), nda, n), prufer_a_arbol(rng.permutation(pb), ndb, n)),
            "uniforme": (prufer_a_arbol(rng.integers(0, len(nda), len(nda) - 2), nda, n),
                         prufer_a_arbol(rng.integers(0, len(ndb), len(ndb) - 2), ndb, n)),
        }
        p = rng.permutation(n)
        pares["barajada"] = (ab, bb[np.ix_(p, p)])
        for t, (x, y) in pares.items():
            c = curva(x, y)
            for u in UMBRALES:
                nulos[t][str(u)].append(primer_cruce(c, u))
            areas[t].append(area(c))
            ipr_nulos[t].append(ipr(autovector(x + y)))
    fila["nulos_mediana"] = {t: {u: float(np.median(v)) for u, v in d.items()} for t, d in nulos.items()}
    fila["area_nulos_mediana"] = {t: float(np.median(v)) for t, v in areas.items()}
    fila["ipr_suma_nulos_mediana"] = {t: float(np.median(v)) for t, v in ipr_nulos.items()}
    return fila


def main():
    datos = [(s,) + C.cargar_sujeto(RAIZ, s) for s in C.sujetos(RAIZ) if C.cargar_sujeto(RAIZ, s) is not None]
    with Pool(int(os.environ.get("PROCESOS", "2"))) as pool:
        filas = pool.map(por_sujeto, [(i, s, sc, fc) for i, (s, sc, fc) in enumerate(datos)])
    resumen = {}
    for u in map(str, UMBRALES):
        real = np.array([f["real"][u] for f in filas])
        resumen[u] = {"real_mediana": float(np.median(real))}
        for t in TIPOS:
            nul = np.array([f["nulos_mediana"][t][u] for f in filas])
            w = wilcoxon(real, nul)
            resumen[u][t] = {"nulo_mediana": float(np.median(nul)), "sujetos_real_antes": int(np.sum(real < nul)),
                             "wilcoxon_p": float(w.pvalue)}
    area_real = np.array([f["area_real"] for f in filas])
    resumen["area_1_menos_rho"] = {"real_mediana": float(np.median(area_real))}
    for t in TIPOS:
        nul = np.array([f["area_nulos_mediana"][t] for f in filas])
        resumen["area_1_menos_rho"][t] = {"nulo_mediana": float(np.median(nul)),
                                          "sujetos_real_menor": int(np.sum(area_real < nul)),
                                          "wilcoxon_p": float(wilcoxon(area_real, nul).pvalue)}
    ee = json.load(open(os.path.join(AQUI, "explicacion_espectral.json"), encoding="utf-8"))
    d = np.load(os.path.join(AQUI, "explicacion_espectral.npz"))
    real95 = np.array([f["real"]["0.95"] for f in filas])
    assert [f["sujeto"] for f in filas] == ee["sujetos"]
    ipr_suma = np.array([f["ipr_real"]["suma"] for f in filas])
    res = {
        "K_por_tipo": K, "puntos_grilla": len(GRILLA), "umbrales": list(UMBRALES), "resumen": resumen,
        "comprobacion_095": {"rho_con_cruce_principal": float(spearmanr(real95, d["real_cruce"])[0]),
                             "mediana_dif_relativa": float(np.median(np.abs(real95 / d["real_cruce"] - 1)))},
        "ipr": {
            "real_dmri_mediana": float(np.median([f["ipr_real"]["dmri"] for f in filas])),
            "real_fmri_mediana": float(np.median([f["ipr_real"]["fmri"] for f in filas])),
            "real_suma_mediana": float(np.median(ipr_suma)),
            "uniforme_68_nodos": 1.0 / 68,
            "nulos_suma_mediana": {t: float(np.median([f["ipr_suma_nulos_mediana"][t] for f in filas])) for t in TIPOS},
            "rho_ipr_suma_con_log_cruce": float(spearmanr(ipr_suma, np.log(d["real_cruce"]))[0]),
            "rho_ipr_suma_con_P": float(spearmanr(ipr_suma, d["real_P"])[0]),
        },
        "por_sujeto": filas,
    }
    with open(os.path.join(AQUI, "sensibilidad_umbral.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "por_sujeto"}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
