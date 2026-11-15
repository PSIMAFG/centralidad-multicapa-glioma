"""Comprobaciones pedidas en la segunda ronda de discusión sobre el predictor P.

1. Aristas compartidas esperadas bajo los nulos que conservan el grado (no solo bajo
   árboles uniformes independientes).
2. Nulo espacial grueso: aristas compartidas esperadas si cada capa conservara cuántas
   aristas tiene en cada categoría anatómica (mismo lóbulo y hemisferio, distinto lóbulo,
   homotópica, otra interhemisférica), con las aristas repartidas al azar dentro de cada
   categoría. No hay coordenadas de las regiones en los datos, así que no es un nulo de
   longitudes; es una cota gruesa de cuánto explica la geometría lobar.
3. Intervalo bootstrap agrupado por sujeto para el residuo de los reales.
4. Calibración dejando un sujeto fuera, con atención a la cola de P baja.
5. Comparación con nulos del mismo sujeto y P parecido (a ±10 %).
6. Error de la aproximación de primer orden en el cruce observado de cada sujeto real.
"""

import json
import os
import sys

import numpy as np
import scipy.linalg as sl
from scipy.stats import rankdata, spearmanr, wilcoxon

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import rutas
sys.path.insert(0, rutas.MODULOS)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cargar as C
import multiplex as M
from controles import radio

RAIZ = rutas.RAIZ
SALIDA = os.path.dirname(os.path.abspath(__file__))
TIPOS = ("mismo_grado", "uniforme", "barajada")
SEMILLA = 20261118
LOBULO = {
    "bankssts": "T", "caudalanteriorcingulate": "C", "caudalmiddlefrontal": "F", "cuneus": "O", "entorhinal": "T",
    "fusiform": "T", "inferiorparietal": "P", "inferiortemporal": "T", "isthmuscingulate": "C",
    "lateraloccipital": "O", "lateralorbitofrontal": "F", "lingual": "O", "medialorbitofrontal": "F",
    "middletemporal": "T", "parahippocampal": "T", "paracentral": "F", "parsopercularis": "F",
    "parsorbitalis": "F", "parstriangularis": "F", "pericalcarine": "O", "postcentral": "P",
    "posteriorcingulate": "C", "precentral": "F", "precuneus": "P", "rostralanteriorcingulate": "C",
    "rostralmiddlefrontal": "F", "superiorfrontal": "F", "superiorparietal": "P", "superiortemporal": "T",
    "supramarginal": "P", "frontalpole": "F", "temporalpole": "T", "transversetemporal": "T", "insula": "I",
}


def regiones():
    """Nombres de las 68 regiones corticales en el orden de parcellation.txt (izquierda y luego derecha)."""
    nombres = []
    with open(os.path.join(RAIZ, "derivatives", "TVB", "parcellation.txt"), encoding="utf-8") as fh:
        for linea in fh:
            p = linea.split()
            if len(p) >= 3 and p[2].startswith(("ctx-lh-", "ctx-rh-")):
                nombres.append(p[2])
    return nombres


def categorias(nombres):
    n = len(nombres)
    hemi = np.array([0 if x.startswith("ctx-lh-") else 1 for x in nombres])
    lob = np.array([LOBULO[x[7:]] for x in nombres])
    base = np.array([x[7:] for x in nombres])
    cat = np.empty((n, n), dtype=object)
    for i in range(n):
        for j in range(n):
            if hemi[i] == hemi[j]:
                cat[i, j] = "mismo_lobulo" if lob[i] == lob[j] else "distinto_lobulo"
            else:
                cat[i, j] = "homotopica" if base[i] == base[j] else "otra_interhemisferica"
    return cat


def esperadas_estratificadas(a, b, cat):
    i, j = np.triu_indices(a.shape[0], 1)
    c = cat[i, j]
    total = 0.0
    for k in np.unique(c):
        m = c == k
        total += (a[i, j][m] > 0).sum() * (b[i, j][m] > 0).sum() / m.sum()
    return total


def principal(m):
    v = np.abs(np.linalg.eigh(m)[1][:, -1])
    return v / np.linalg.norm(v)


def main():
    rng = np.random.default_rng(SEMILLA)
    d = np.load(os.path.join(SALIDA, "explicacion_espectral.npz"))
    mod = json.load(open(os.path.join(SALIDA, "explicacion_espectral.json"), encoding="utf-8"))["modelo_en_nulos"]
    datos = [C.cargar_sujeto(RAIZ, s) for s in C.sujetos(RAIZ)]
    datos = [x for x in datos if x is not None]
    res = {}

    # 1. Aristas compartidas en los nulos que conservan el grado (desde el Jaccard guardado).
    aristas = np.array([[int((np.triu(M.arbol_maximo(sc)) > 0).sum()), int((np.triu(M.arbol_maximo(fc)) > 0).sum())]
                        for sc, fc in datos])
    tot = aristas.sum(1)[:, None]
    comp = {t: d[f"{t}_jaccard"] * tot / (1 + d[f"{t}_jaccard"]) for t in TIPOS}
    reales = []
    nombres = regiones()
    cat = categorias(nombres)
    estrat, homo_fc, homo_sc, fc_homotopica, fc_otra = [], [], [], [], []
    for sc, fc in datos:
        a, b = M.arbol_maximo(sc), M.arbol_maximo(fc)
        reales.append(int(((np.triu(a) > 0) & (np.triu(b) > 0)).sum()))
        estrat.append(esperadas_estratificadas(a, b, cat))
        iu = np.triu_indices(68, 1)
        homo = cat[iu] == "homotopica"
        homo_sc.append(int((a[iu][homo] > 0).sum()))
        homo_fc.append(int((b[iu][homo] > 0).sum()))
        fc_homotopica.append(float(fc[iu][homo].mean()))
        fc_otra.append(float(fc[iu][cat[iu] == "otra_interhemisferica"].mean()))
    reales, estrat = np.array(reales), np.array(estrat)
    res["aristas_compartidas"] = {
        "reales_media": float(reales.mean()),
        "nulos_media": {t: float(comp[t].mean()) for t in TIPOS},
        "arboles_uniformes_teorico": 67 * 67 / 2278,
        "nulo_estratificado_lobulo_hemisferio_media": float(estrat.mean()),
        "reales_sobre_estratificado_mediana": float(np.median(reales / estrat)),
        "sujetos_reales_mayor_que_estratificado": int(np.sum(reales > estrat)),
        "wilcoxon_reales_vs_estratificado_p": float(wilcoxon(reales - estrat).pvalue),
        "aristas_homotopicas_media": {"dmri": float(np.mean(homo_sc)), "fmri": float(np.mean(homo_fc))},
        "fc_homotopica_vs_otra_interhemisferica": [float(np.mean(fc_homotopica)), float(np.mean(fc_otra))],
    }

    # 3 y 4. Bootstrap agrupado por sujeto y calibración dejando un sujeto fuera.
    lp = {t: np.log(d[f"{t}_P"]) for t in TIPOS}
    lc = {t: np.log(d[f"{t}_cruce"]) for t in TIPOS}
    lpr, lcr = np.log(d["real_P"]), np.log(d["real_cruce"])
    n = len(lpr)

    def ajuste(idx):
        x = np.concatenate([lp[t][idx].ravel() for t in TIPOS])
        y = np.concatenate([lc[t][idx].ravel() for t in TIPOS])
        b, a = np.polyfit(x, y, 1)
        return a, b

    boot = []
    for _ in range(1000):
        idx = rng.integers(0, n, n)
        a, b = ajuste(idx)
        boot.append(np.median(lcr[idx] - (a + b * lpr[idx])))
    res["residuo_real_bootstrap_agrupado"] = {"mediana": mod["residuo_real_mediana"],
                                              "ic95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]}
    corte = np.percentile(np.concatenate([lp[t].ravel() for t in TIPOS]), 5)
    loso_real, loso_cola = [], []
    for i in range(n):
        otros = np.array([k for k in range(n) if k != i])
        a, b = ajuste(otros)
        loso_real.append(lcr[i] - (a + b * lpr[i]))
        for t in TIPOS:
            m = lp[t][i] <= corte
            loso_cola.extend((lc[t][i][m] - (a + b * lp[t][i][m])).tolist())
    loso_real = np.array(loso_real)
    res["calibracion_dejando_un_sujeto_fuera"] = {
        "residuo_real_mediana": float(np.median(loso_real)), "wilcoxon_p": float(wilcoxon(loso_real).pvalue),
        "residuo_nulos_en_cola_P_baja_mediana": float(np.median(loso_cola)), "nulos_en_cola": len(loso_cola),
        "corte_log_P": float(corte),
    }

    # 5. Nulos del mismo sujeto con P parecido.
    emp = []
    for i in range(n):
        pn = np.concatenate([d[f"{t}_P"][i] for t in TIPOS])
        cn = np.concatenate([d[f"{t}_cruce"][i] for t in TIPOS])
        cerca = np.abs(np.log(pn / d["real_P"][i])) < 0.1
        if cerca.sum() >= 10:
            emp.append(lcr[i] - np.median(np.log(cn[cerca])))
    emp = np.array(emp)
    res["emparejados_por_P"] = {"sujetos": int(len(emp)), "razon_mediana": float(np.exp(np.median(emp))),
                                "wilcoxon_p": float(wilcoxon(emp).pvalue)}
    dentro = lpr >= np.log(np.percentile(np.exp(np.concatenate([lp[t].ravel() for t in TIPOS])), 1))
    res["reales_dentro_del_soporte"] = {"sujetos": int(dentro.sum()),
                                        "residuo_mediana": float(np.median((lcr - (mod["intercepto"] + mod["pendiente"] * lpr))[dentro])),
                                        "wilcoxon_p": float(wilcoxon((lcr - (mod["intercepto"] + mod["pendiente"] * lpr))[dentro]).pvalue)}

    # 6. Error de la aproximación de primer orden en el cruce observado.
    rho_aprox, error_rel = [], []
    for k, (sc, fc) in enumerate(datos):
        a, b = M.arbol_maximo(sc), M.arbol_maximo(fc)
        ref = (radio(a) + radio(b)) / 2.0
        w = float(d["real_cruce"][k]) * ref
        cc, dd = (a + b) / 2.0, (a - b) / 2.0
        lam, vec = np.linalg.eigh(cc)
        v1 = vec[:, -1] * np.sign(vec[:, -1].sum())
        coef = (vec[:, :-1].T @ (dd @ (dd @ v1))) / (lam[-1] - lam[:-1])
        aprox = v1 + (vec[:, :-1] @ coef) / (2 * w)
        s = np.block([[a, w * np.eye(68)], [w * np.eye(68), b]])
        x = np.abs(sl.eigh(s, subset_by_index=[135, 135], driver="evr")[1][:, 0])
        exacto = x[:68] + x[68:]
        exacto = exacto / np.linalg.norm(exacto)
        rho_aprox.append(float(spearmanr(aprox, v1).statistic))
        error_rel.append(float(np.linalg.norm((aprox - v1) - (exacto - v1)) / np.linalg.norm(exacto - v1)))
    res["primer_orden_en_el_cruce"] = {
        "rho_aproximacion_vs_suma_mediana": float(np.median(rho_aprox)),
        "rho_aproximacion_vs_suma_p10_p90": [float(np.percentile(rho_aprox, 10)), float(np.percentile(rho_aprox, 90))],
        "error_relativo_de_la_desviacion_mediana": float(np.median(error_rel)),
        "error_relativo_p10_p90": [float(np.percentile(error_rel, 10)), float(np.percentile(error_rel, 90))],
    }
    with open(os.path.join(SALIDA, "validacion_P.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1)
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
