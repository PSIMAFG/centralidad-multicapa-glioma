"""Controles pedidos por la tercera revisión.

1. Acoplamiento adimensional: omega dividido por el radio espectral medio de las dos capas.
2. Árboles aleatorios: cada región conserva su grado (secuencia de Prüfer barajada) o
   árboles uniformes, para ver si el cruce es propio de los cerebros o de cualquier árbol.
3. Sensibilidad al umbral de semejanza (0,90 / 0,95 / 0,99) e intervalo de Wilson.
"""

import json
import os
import sys

import numpy as np
from scipy.stats import spearmanr, mannwhitneyu

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import rutas
sys.path.insert(0, rutas.MODULOS)
import cargar as C
import multiplex as M

RAIZ = rutas.RAIZ
SALIDA = os.path.dirname(os.path.abspath(__file__))
OMT = np.logspace(-3, 1.5, 46)
K_NULOS = 8
SEMILLA = 20261114


def radio(m):
    return float(np.abs(np.linalg.eigvalsh(m)).max())


def rescalar(m):
    return m / radio(m)


def cruce(fila, grilla, umbral=0.95):
    sobre = np.where(fila >= umbral)[0]
    if len(sobre) == 0:
        return np.inf
    k = sobre[0]
    if k == 0:
        return float(grilla[0])
    lo = np.interp(umbral, [fila[k - 1], fila[k]], [np.log10(grilla[k - 1]), np.log10(grilla[k])])
    return float(10 ** lo)


def curva_adim(c1, c2):
    """rho(versatilidad, centralidad de la suma) en función de omega / radio medio."""
    ref = (radio(c1) + radio(c2)) / 2.0
    cs = M.centralidad_capa(c1 + c2)
    fila = np.array([spearmanr(M.versatilidad([c1, c2], w * ref), cs).statistic for w in OMT])
    return fila, ref


def arbol_a_prufer(a):
    """Secuencia de Prüfer del árbol formado por las regiones con al menos una arista.

    Si una región no tiene conexiones en la matriz original (en sub-PAT26 la capa
    estructural tiene una fila nula), el árbol de expansión máxima es un bosque con esa
    región aislada. La secuencia se calcula sobre las regiones conectadas y la aislada
    se conserva aislada en los árboles nulos.
    """
    nodos = np.where((a > 0).sum(1) > 0)[0]
    b = (a[np.ix_(nodos, nodos)] > 0).astype(float)
    k = len(nodos)
    if int(b.sum() // 2) != k - 1:
        raise ValueError("las regiones conectadas no forman un unico arbol")
    grado = b.sum(1).astype(int)
    sec = []
    for _ in range(k - 2):
        hoja = int(np.where(grado == 1)[0][0])
        vecino = int(np.where(b[hoja] > 0)[0][0])
        sec.append(vecino)
        b[hoja, vecino] = b[vecino, hoja] = 0
        grado[hoja] -= 1
        grado[vecino] -= 1
    return np.array(sec, dtype=int), nodos


def prufer_a_arbol(sec, nodos, n):
    """Árbol sobre las regiones `nodos` a partir de una secuencia de Prüfer en índices locales."""
    k = len(nodos)
    grado = np.ones(k, dtype=int)
    for x in sec:
        grado[x] += 1
    t = np.zeros((k, k))
    for x in sec:
        hoja = int(np.where(grado == 1)[0][0])
        t[hoja, x] = t[x, hoja] = 1.0
        grado[hoja] -= 1
        grado[x] -= 1
    u, v = np.where(grado == 1)[0]
    t[u, v] = t[v, u] = 1.0
    a = np.zeros((n, n))
    a[np.ix_(nodos, nodos)] = t
    return a


def main():
    rng = np.random.default_rng(SEMILLA)
    datos = [(s,) + C.cargar_sujeto(RAIZ, s) for s in C.sujetos(RAIZ) if C.cargar_sujeto(RAIZ, s) is not None]
    n = datos[0][1].shape[0]
    res = {"grilla_adimensional": OMT.tolist()}

    cur = {"binarizado": [], "ponderado_reescalado": [], "completo_reescalado": [],
           "nulo_mismos_grados": [], "nulo_uniforme": []}
    cru = {k: [] for k in cur}
    refs_bin, rho1_bin = [], []
    grados_iguales = 0
    for s, sc, fc in datos:
        ab, bb = M.arbol_maximo(sc), M.arbol_maximo(fc)
        f, ref = curva_adim(ab, bb)
        cur["binarizado"].append(f); cru["binarizado"].append(cruce(f, OMT)); refs_bin.append(ref)
        cs = M.centralidad_capa(ab + bb)
        rho1_bin.append(spearmanr(M.versatilidad([ab, bb], 1.0), cs).statistic)

        aw = rescalar(M.arbol_maximo(sc, binario=False)); bw = rescalar(M.arbol_maximo(fc, binario=False))
        f, _ = curva_adim(aw, bw)
        cur["ponderado_reescalado"].append(f); cru["ponderado_reescalado"].append(cruce(f, OMT))

        f, _ = curva_adim(rescalar(sc), rescalar(fc))
        cur["completo_reescalado"].append(f); cru["completo_reescalado"].append(cruce(f, OMT))

        (pa, nda), (pb, ndb) = arbol_a_prufer(ab), arbol_a_prufer(bb)
        for _ in range(K_NULOS):
            na = prufer_a_arbol(rng.permutation(pa), nda, n)
            nb = prufer_a_arbol(rng.permutation(pb), ndb, n)
            grados_iguales += int(np.array_equal(na.sum(1), (ab > 0).sum(1)))
            grados_iguales += int(np.array_equal(nb.sum(1), (bb > 0).sum(1)))
            f, _ = curva_adim(na, nb)
            cur["nulo_mismos_grados"].append(f); cru["nulo_mismos_grados"].append(cruce(f, OMT))
            ua = prufer_a_arbol(rng.integers(0, len(nda), len(nda) - 2), nda, n)
            ub = prufer_a_arbol(rng.integers(0, len(ndb), len(ndb) - 2), ndb, n)
            f, _ = curva_adim(ua, ub)
            cur["nulo_uniforme"].append(f); cru["nulo_uniforme"].append(cruce(f, OMT))

    np.savez(os.path.join(SALIDA, "curvas_adimensionales.npz"), grilla=OMT,
             **{k: np.array(v) for k, v in cur.items()})
    resumen = {}
    for k, v in cru.items():
        v = np.array(v)
        fin = v[np.isfinite(v)]
        resumen[k] = {"n": int(len(v)), "mediana": float(np.median(fin)), "p10": float(np.percentile(fin, 10)),
                      "p90": float(np.percentile(fin, 90)), "sin_cruce": int(np.sum(~np.isfinite(v)))}
    res["cruce_adimensional"] = resumen
    res["capas_nulas_con_grado_regional_identico"] = {"verificadas": grados_iguales,
                                                       "total": 2 * K_NULOS * len(datos)}
    res["cruces_por_sujeto"] = {k: [float(x) for x in v] for k, v in cru.items()}
    res["sujetos"] = [d[0] for d in datos]
    refs_bin = np.array(refs_bin)
    res["binarizado_radio_medio"] = {"mediana": float(np.median(refs_bin)), "min": float(refs_bin.min()),
                                     "max": float(refs_bin.max())}
    res["binarizado_omega1_adimensional_mediana"] = float(np.median(1.0 / refs_bin))
    real = np.array(cru["binarizado"]); nul = np.array(cru["nulo_mismos_grados"]); uni = np.array(cru["nulo_uniforme"])
    res["real_vs_nulo_mismos_grados_p"] = float(mannwhitneyu(real, nul).pvalue)
    res["real_vs_nulo_uniforme_p"] = float(mannwhitneyu(real, uni).pvalue)

    rho1 = np.array(rho1_bin)
    umbrales = {}
    for u in (0.90, 0.95, 0.99):
        k = int(np.sum(rho1 >= u)); nn = len(rho1); z = 1.959964
        p = k / nn
        centro = (p + z * z / (2 * nn)) / (1 + z * z / nn)
        semi = z * np.sqrt(p * (1 - p) / nn + z * z / (4 * nn * nn)) / (1 + z * z / nn)
        umbrales[f"{u:.2f}"] = {"k": k, "n": nn, "wilson_bajo": float(centro - semi), "wilson_alto": float(centro + semi)}
    res["umbrales_omega1_binarizado"] = umbrales

    with open(os.path.join(SALIDA, "controles.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "grilla_adimensional"}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
