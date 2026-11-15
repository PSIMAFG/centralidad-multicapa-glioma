"""Reproducibilidad entre sesiones con los controles de ds002080 (OpenNeuro).

Diez controles (CON02 a CON11) tienen una segunda sesión («ses-postop») procesada con el
mismo pipeline TVB. Entre la primera sesión (ds001226, copia local) y la segunda se comparan:
- el cruce adimensional (omega en que rho con la suma llega a 0,95), con CCI(2,1) y CCI(3,1)
  sobre log(cruce) e intervalo bootstrap por sujetos;
- las aristas de cada árbol (dMRI y fMRI) que se repiten entre sesiones;
- las aristas compartidas entre los árboles dMRI y fMRI de cada sesión.
Los pacientes con sesión posoperatoria se informan solo de forma descriptiva: la cirugía cambia
la red, así que no miden reproducibilidad.
"""

import json
import os
import sys

import numpy as np
from scipy.stats import spearmanr, wilcoxon

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
sys.path.insert(0, os.path.join(AQUI, "..", "controles"))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import rutas
sys.path.insert(0, rutas.MODULOS)
import cargar as C
import multiplex as M
from explicacion_espectral import cruce
from lectura import DATOS

RAIZ = rutas.RAIZ
SEMILLA = 20261122


def cargar_postop(sujeto):
    base = os.path.join(DATOS, "ds002080", "derivatives", "TVB", sujeto, "ses-postop")
    _, sc = C.leer_mat(os.path.join(base, "SCthrAn.mat"))
    _, fc = C.elegir_funcional(C.leer_mat_todas(os.path.join(base, "FC.mat")), sc.shape[0])
    return C.limpiar(sc), C.limpiar(fc)


def compartidas(a, b):
    return int(((np.triu(a, 1) > 0) & (np.triu(b, 1) > 0)).sum())


def cci(x):
    """CCI(2,1) de acuerdo absoluto y CCI(3,1) de consistencia para una matriz sujetos x sesiones."""
    n, k = x.shape
    gm = x.mean()
    msr = k * ((x.mean(1) - gm) ** 2).sum() / (n - 1)
    msc = n * ((x.mean(0) - gm) ** 2).sum() / (k - 1)
    sse = ((x - x.mean(1, keepdims=True) - x.mean(0, keepdims=True) + gm) ** 2).sum()
    mse = sse / ((n - 1) * (k - 1))
    cci2 = (msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n)
    cci3 = (msr - mse) / (msr + (k - 1) * mse)
    return float(cci2), float(cci3)


def main():
    rng = np.random.default_rng(SEMILLA)
    grupos = C.grupos(RAIZ)
    base_post = os.path.join(DATOS, "ds002080", "derivatives", "TVB")
    con_post = sorted(s for s in os.listdir(base_post) if s.startswith("sub-"))
    filas = []
    for s in con_post:
        pre = C.cargar_sujeto(RAIZ, s)
        post = cargar_postop(s)
        if pre is None:
            continue
        a1, b1 = M.arbol_maximo(pre[0]), M.arbol_maximo(pre[1])
        a2, b2 = M.arbol_maximo(post[0]), M.arbol_maximo(post[1])
        iu = np.triu_indices(68, 1)
        filas.append({
            "sujeto": s, "grupo": grupos.get(s),
            "cruce_1": cruce(a1, b1), "cruce_2": cruce(a2, b2),
            "dmri_repetidas": compartidas(a1, a2), "fmri_repetidas": compartidas(b1, b2),
            "dmri_fmri_1": compartidas(a1, b1), "dmri_fmri_2": compartidas(a2, b2),
            "rho_pesos_sc": float(spearmanr(pre[0][iu], post[0][iu]).statistic),
            "rho_pesos_fc": float(spearmanr(pre[1][iu], post[1][iu]).statistic),
        })
    ctrl = [f for f in filas if f["grupo"] == "control"]
    pac = [f for f in filas if f["grupo"] != "control"]
    x = np.log(np.array([[f["cruce_1"], f["cruce_2"]] for f in ctrl]))
    c2, c3 = cci(x)
    boot = []
    for _ in range(2000):
        idx = rng.integers(0, len(x), len(x))
        if np.ptp(x[idx, 0]) > 0:
            boot.append(cci(x[idx]))
    boot = np.array(boot)
    reales = np.load(os.path.join(AQUI, "..", "controles", "explicacion_espectral.npz"))["real_cruce"]
    dif = x[:, 1] - x[:, 0]

    def resumen(grupo):
        return {
            "n": len(grupo),
            "cruce_sesion_1_mediana": float(np.median([f["cruce_1"] for f in grupo])),
            "cruce_sesion_2_mediana": float(np.median([f["cruce_2"] for f in grupo])),
            "razon_cruce_2_sobre_1_mediana": float(np.median([f["cruce_2"] / f["cruce_1"] for f in grupo])),
            "dmri_aristas_repetidas_media": float(np.mean([f["dmri_repetidas"] for f in grupo])),
            "fmri_aristas_repetidas_media": float(np.mean([f["fmri_repetidas"] for f in grupo])),
            "dmri_fmri_compartidas_sesion_1_media": float(np.mean([f["dmri_fmri_1"] for f in grupo])),
            "dmri_fmri_compartidas_sesion_2_media": float(np.mean([f["dmri_fmri_2"] for f in grupo])),
            "rho_pesos_sc_mediana": float(np.median([f["rho_pesos_sc"] for f in grupo])),
            "rho_pesos_fc_mediana": float(np.median([f["rho_pesos_fc"] for f in grupo])),
        }

    res = {
        "controles": {**resumen(ctrl),
                      "cci_2_1_log_cruce": c2, "cci_2_1_ic95": [float(np.percentile(boot[:, 0], 2.5)), float(np.percentile(boot[:, 0], 97.5))],
                      "cci_3_1_log_cruce": c3, "cci_3_1_ic95": [float(np.percentile(boot[:, 1], 2.5)), float(np.percentile(boot[:, 1], 97.5))],
                      "spearman_cruce_entre_sesiones": float(spearmanr(x[:, 0], x[:, 1]).statistic),
                      "wilcoxon_cambio_p": float(wilcoxon(x[:, 1] - x[:, 0]).pvalue)},
        "pacientes_descriptivo": resumen(pac),
        "referencia_36_sujetos": {"cruce_mediana": float(np.median(reales)),
                                  "desviacion_log_cruce_entre_sujetos": float(np.std(np.log(reales), ddof=1)),
                                  "desviacion_log_cruce_intra_sujeto_controles": float(np.std(dif, ddof=1) / np.sqrt(2))},
        "por_sujeto": filas,
    }
    with open(os.path.join(AQUI, "test_retest.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "por_sujeto"}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
