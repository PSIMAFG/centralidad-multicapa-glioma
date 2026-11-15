"""¿Qué explica que los árboles reales converjan a su suma antes que los nulos?

En coordenadas suma y diferencia, la supra-adyacencia es [[C + wI, D], [D, C - wI]], con
C = (A + B)/2 y D = (A - B)/2. Con acoplamiento fuerte, la componente común es el autovector
principal de C + D^2/(2w), así que a primer orden la centralidad multicapa se separa de la de
la suma en delta/(2w), con

    delta = sum_{k>=2} (v_k' D^2 v_1) / (lambda_1 - lambda_k) v_k

(v_k y lambda_k de C). El cruce adimensional esperado es proporcional a
P = ||delta|| / (2 sd(v_1) r), con r el radio espectral medio de las capas: combina la
discrepancia entre capas (D, donde las aristas compartidas se cancelan) y la brecha espectral
de la suma.

Este script:
1. Repite los tres controles con 500 réplicas por sujeto y tipo (antes, 8).
2. Calcula para cada multicapa real y nulo el cruce, P, la brecha espectral relativa de C y
   el índice de Jaccard entre las aristas de las dos capas.
3. Ajusta log(cruce) ~ log(P) solo en los nulos y mide si los reales se desvían de esa
   relación (residuos pareados por sujeto).
4. Repite la estabilidad ante ruido con 20 realizaciones y guarda su dispersión.
5. Compara, sujeto a sujeto, árboles y redes con y sin pesos a cada densidad.
6. Verifica la condición de irreducibilidad: la unión de las dos capas es conexa.

El cruce se busca con una grilla gruesa (4 puntos por década) y 6 pasos de bisección en
escala logarítmica, interpolando linealmente en log(omega) dentro del último intervalo.
"""

import json
import os
import sys
from multiprocessing import Pool

import numpy as np
import scipy.linalg as sl
from scipy.sparse.csgraph import connected_components
from scipy.stats import rankdata, spearmanr, wilcoxon

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import rutas
sys.path.insert(0, rutas.MODULOS)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cargar as C
import multiplex as M
from controles import arbol_a_prufer, prufer_a_arbol, radio, rescalar
from sensibilidad import capa_densidad

RAIZ = rutas.RAIZ
SALIDA = os.path.dirname(os.path.abspath(__file__))
K = 500
REALIZACIONES = 20
SEMILLA = 20261117
GRUESA = np.logspace(-2.5, 1.5, 17)
PASOS = 6
UMBRAL = 0.95


def centralidad(m):
    v = np.abs(np.linalg.eigh(m)[1][:, -1])
    return v / v.sum()


def rho(a, b, w, rangos_suma):
    n = a.shape[0]
    s = np.block([[a, w * np.eye(n)], [w * np.eye(n), b]])
    x = np.abs(sl.eigh(s, subset_by_index=[2 * n - 1, 2 * n - 1], driver="evr")[1][:, 0])
    return float(np.corrcoef(rankdata(x[:n] + x[n:]), rangos_suma)[0, 1])


def cruce(a, b):
    """Omega adimensional en que rho con la suma alcanza el umbral por primera vez."""
    ref = (radio(a) + radio(b)) / 2.0
    rs = rankdata(centralidad(a + b))
    previo = None
    for k, g in enumerate(GRUESA):
        r = rho(a, b, g * ref, rs)
        if r >= UMBRAL:
            if k == 0:
                return float(g)
            lo, hi, rlo, rhi = GRUESA[k - 1], g, previo, r
            for _ in range(PASOS):
                mid = np.sqrt(lo * hi)
                rm = rho(a, b, mid * ref, rs)
                if rm >= UMBRAL:
                    hi, rhi = mid, rm
                else:
                    lo, rlo = mid, rm
            if rhi == rlo:
                return float(hi)
            t = (UMBRAL - rlo) / (rhi - rlo)
            return float(10 ** (np.log10(lo) + t * (np.log10(hi) - np.log10(lo))))
        previo = r
    return float("inf")


def predictores(a, b):
    ref = (radio(a) + radio(b)) / 2.0
    c, d = (a + b) / 2.0, (a - b) / 2.0
    lam, vec = np.linalg.eigh(c)
    v1 = vec[:, -1] * np.sign(vec[:, -1].sum())
    d2v1 = d @ (d @ v1)
    coef = (vec[:, :-1].T @ d2v1) / (lam[-1] - lam[:-1])
    delta = vec[:, :-1] @ coef
    ea, eb = set(zip(*np.nonzero(np.triu(a)))), set(zip(*np.nonzero(np.triu(b))))
    return {"P": float(np.linalg.norm(delta) / (2.0 * np.std(v1) * ref)),
            "brecha": float((lam[-1] - lam[-2]) / lam[-1]),
            "jaccard": float(len(ea & eb) / len(ea | eb)),
            "discrepancia": float(np.linalg.norm(d) / np.linalg.norm(c))}


def por_sujeto(args):
    indice, s, sc, fc = args
    rng = np.random.default_rng([SEMILLA, indice])
    n = sc.shape[0]
    ab, bb = M.arbol_maximo(sc), M.arbol_maximo(fc)
    filas = [{"tipo": "real", **predictores(ab, bb), "cruce": cruce(ab, bb)}]
    (pa, nda), (pb, ndb) = arbol_a_prufer(ab), arbol_a_prufer(bb)
    for _ in range(K):
        na, nb = prufer_a_arbol(rng.permutation(pa), nda, n), prufer_a_arbol(rng.permutation(pb), ndb, n)
        assert np.array_equal(na.sum(1), (ab > 0).sum(1)) and np.array_equal(nb.sum(1), (bb > 0).sum(1))
        filas.append({"tipo": "mismo_grado", **predictores(na, nb), "cruce": cruce(na, nb)})
        ua = prufer_a_arbol(rng.integers(0, len(nda), len(nda) - 2), nda, n)
        ub = prufer_a_arbol(rng.integers(0, len(ndb), len(ndb) - 2), ndb, n)
        filas.append({"tipo": "uniforme", **predictores(ua, ub), "cruce": cruce(ua, ub)})
        p = rng.permutation(n)
        bp = bb[np.ix_(p, p)]
        filas.append({"tipo": "barajada", **predictores(ab, bp), "cruce": cruce(ab, bp)})

    construcciones = {"arbol_binarizado": lambda w: capa_densidad(w, "arbol", True),
                      "arbol_ponderado": lambda w: capa_densidad(w, "arbol", False),
                      "densidad_0.1_binarizada": lambda w: capa_densidad(w, 0.10, True),
                      "densidad_0.1_ponderada": lambda w: capa_densidad(w, 0.10, False)}
    ruido = {}
    for nombre, f in construcciones.items():
        orig = cruce(f(sc), f(fc))
        pert = [cruce(f(M.perturbar(sc, 0.1, rng)), f(M.perturbar(fc, 0.1, rng))) for _ in range(REALIZACIONES)]
        ruido[nombre] = {"original": orig, "perturbados": pert}

    conexas = {}
    for nombre, (x, y) in {"cruda": (sc, fc), "arbol_binarizado": (ab, bb)}.items():
        conexas[nombre] = int(connected_components((x + y) > 0, directed=False)[0] == 1)
    return {"sujeto": s, "filas": filas, "ruido": ruido, "union_conexa": conexas}


def resumen_pareado(real, nul):
    w = wilcoxon(real, nul)
    return {"mediana_nula": float(np.median(nul)), "sujetos_real_antes": int(np.sum(real < nul)),
            "wilcoxon_W": float(w.statistic), "wilcoxon_p": float(w.pvalue)}


def main():
    datos = [(s,) + C.cargar_sujeto(RAIZ, s) for s in C.sujetos(RAIZ) if C.cargar_sujeto(RAIZ, s) is not None]
    with Pool(int(os.environ.get("PROCESOS", "2"))) as pool:
        res = pool.map(por_sujeto, [(i, s, sc, fc) for i, (s, sc, fc) in enumerate(datos)])
    sujetos = [r["sujeto"] for r in res]
    tipos = ("mismo_grado", "uniforme", "barajada")
    tabla = {t: {k: np.array([[f[k] for f in r["filas"] if f["tipo"] == t] for r in res])
                 for k in ("cruce", "P", "brecha", "jaccard", "discrepancia")} for t in tipos + ("real",)}
    for k in tabla["real"]:
        tabla["real"][k] = tabla["real"][k][:, 0]
    salida = {"K": K, "sujetos": sujetos}

    real = tabla["real"]["cruce"]
    salida["cruce_real_mediana"] = float(np.median(real))
    salida["controles_500"] = {t: resumen_pareado(real, np.median(tabla[t]["cruce"], axis=1)) for t in tipos}
    rng = np.random.default_rng(SEMILLA)
    estab = {}
    for t in tipos:
        sub = [float(np.median(np.median(tabla[t]["cruce"][:, rng.choice(K, 8, replace=False)], axis=1)))
               for _ in range(1000)]
        estab[t] = {"mediana_con_8_p2.5": float(np.percentile(sub, 2.5)),
                    "mediana_con_8_p97.5": float(np.percentile(sub, 97.5)),
                    "mediana_con_500": float(np.median(np.median(tabla[t]["cruce"], axis=1)))}
    salida["estabilidad_replicas"] = estab

    todos = {k: np.concatenate([tabla["real"][k]] + [tabla[t][k].ravel() for t in tipos])
             for k in tabla["real"]}
    lc = np.log(todos["cruce"])
    salida["correlacion_con_log_cruce_todos"] = {
        k: float(spearmanr(lc, todos[k]).statistic) for k in ("P", "brecha", "jaccard", "discrepancia")}
    salida["correlacion_con_log_cruce_reales"] = {
        k: float(spearmanr(np.log(real), tabla["real"][k]).statistic) for k in ("P", "brecha", "jaccard", "discrepancia")}
    salida["predictores_mediana"] = {t: {k: float(np.median(tabla[t][k])) for k in ("P", "brecha", "jaccard")}
                                     for t in tipos + ("real",)}

    xn = np.concatenate([np.log(tabla[t]["P"].ravel()) for t in tipos])
    yn = np.concatenate([np.log(tabla[t]["cruce"].ravel()) for t in tipos])
    pend, inter = np.polyfit(xn, yn, 1)
    r2 = 1 - np.sum((yn - (inter + pend * xn)) ** 2) / np.sum((yn - yn.mean()) ** 2)
    res_real = np.log(real) - (inter + pend * np.log(tabla["real"]["P"]))
    res_nul = {t: np.log(tabla[t]["cruce"]) - (inter + pend * np.log(tabla[t]["P"])) for t in tipos}
    soporte = [float(np.mean(np.concatenate([tabla[t]["P"][i] for t in tipos]) <= tabla["real"]["P"][i]))
               for i in range(len(real))]
    wr = wilcoxon(res_real)
    salida["modelo_en_nulos"] = {
        "pendiente": float(pend), "intercepto": float(inter), "R2": float(r2),
        "residuo_real_mediana": float(np.median(res_real)),
        "razon_real_sobre_prediccion_mediana": float(np.exp(np.median(res_real))),
        "residuo_real_wilcoxon_p": float(wr.pvalue),
        "residuo_real_menos_nulo_mediana": {t: float(np.median(res_real - np.median(res_nul[t], axis=1))) for t in tipos},
        "fraccion_nulos_con_P_menor_o_igual_al_real_mediana": float(np.median(soporte)),
        "diferencia_log_real_nulo_bruta": {t: float(np.median(np.log(real) - np.median(np.log(tabla[t]["cruce"]), axis=1)))
                                           for t in tipos},
    }

    ruido = {}
    for nombre in res[0]["ruido"]:
        orig = np.array([r["ruido"][nombre]["original"] for r in res])
        pert = np.array([r["ruido"][nombre]["perturbados"] for r in res])
        rs = [float(spearmanr(orig, pert[:, j]).statistic) for j in range(REALIZACIONES)]
        ruido[nombre] = {"mediana": float(np.median(rs)), "p5": float(np.percentile(rs, 5)),
                         "p95": float(np.percentile(rs, 95)), "realizaciones": REALIZACIONES}
    salida["ruido_20_realizaciones"] = ruido

    sens = json.load(open(os.path.join(SALIDA, "sensibilidad.json"), encoding="utf-8"))["densidad"]
    pares = {}
    for d in ("arbol", "0.05", "0.1", "0.2", "0.4"):
        bz = np.array(sens[f"binarizada_{d}"]["cruces_por_sujeto"])
        pz = np.array(sens[f"ponderada_{d}"]["cruces_por_sujeto"])
        razon = np.log(bz / pz)
        boot = [np.median(razon[rng.integers(0, len(razon), len(razon))]) for _ in range(2000)]
        pares[d] = {"razon_binarizada_ponderada_mediana": float(np.exp(np.median(razon))),
                    "ic95": [float(np.exp(np.percentile(boot, 2.5))), float(np.exp(np.percentile(boot, 97.5)))],
                    "wilcoxon_p": float(wilcoxon(razon).pvalue)}
    salida["binarizada_frente_a_ponderada"] = pares

    salida["union_de_capas_conexa"] = {k: int(sum(r["union_conexa"][k] for r in res)) for k in res[0]["union_conexa"]}
    np.savez(os.path.join(SALIDA, "explicacion_espectral.npz"),
             **{f"{t}_{k}": v for t, d in tabla.items() for k, v in d.items()})
    with open(os.path.join(SALIDA, "explicacion_espectral.json"), "w", encoding="utf-8") as fh:
        json.dump(salida, fh, ensure_ascii=False, indent=1)
    print(json.dumps(salida, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
