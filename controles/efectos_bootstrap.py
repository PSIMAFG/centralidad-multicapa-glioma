"""Tamaños de efecto pareados por sujeto frente a cada nulo, con intervalos bootstrap entre sujetos.

Para cada sujeto se compara el árbol real con su propia referencia nula:

- Cruce: logaritmo natural de la razón entre el cruce real y la mediana de los cruces de sus nulos
  (500 réplicas topológicas o 100 pares con distancias). Se resume con la mediana entre sujetos,
  como el resto de los cruces del póster.
- Aristas compartidas: diferencia entre las aristas reales y las esperadas en sus nulos. Se resume
  con la media entre sujetos, como el resto de las aristas del póster.

Los intervalos remuestrean sujetos (bootstrap percentil). Describen la incertidumbre entre sujetos,
condicionada a las estimaciones Monte Carlo de cada sujeto. También se resumen la localización del
autovector (número efectivo de regiones, 1/IPR), la concordancia de las regiones más centrales y
la censura de los cruces en la grilla de sensibilidad al umbral.
"""

import json
import os

import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
B = 100000
SEMILLA = 20261114
GRILLA = np.logspace(-2.5, 1.5, 81)
TIPOS_CRUCE = ("mismo_grado", "uniforme", "barajada", "grados_y_distancia", "homotopicas_fijas")
TIPOS_ARISTAS = ("mismo_grado", "lobar", "distancia", "grados_y_distancia", "homotopicas_fijas")


def leer(nombre):
    with open(os.path.join(AQUI, nombre), encoding="utf-8") as fh:
        return json.load(fh)


def bootstrap(x, estadistico, rng):
    idx = rng.integers(0, len(x), (B, len(x)))
    reps = estadistico(x[idx], axis=1)
    return [float(v) for v in np.percentile(reps, [2.5, 97.5])]


def main():
    rng = np.random.default_rng(SEMILLA)
    ee = leer("explicacion_espectral.json")
    gd = leer("nulo_grados_distancia.json")
    ap = leer("aristas_por_sujeto.json")
    su = leer("sensibilidad_umbral.json")
    se = leer("sensibilidad.json")
    d = np.load(os.path.join(AQUI, "explicacion_espectral.npz"))
    g = np.load(os.path.join(AQUI, "nulo_grados_distancia.npz"))
    sujetos = ee["sujetos"]
    assert [f["sujeto"] for f in gd["por_sujeto"]] == sujetos == ap["sujetos"]
    assert [f["sujeto"] for f in su["por_sujeto"]] == sujetos
    real = d["real_cruce"]
    assert np.allclose(real, g["cruce_real"])
    nulo_distancia = np.median(g["cruces_nulos"], axis=1)
    assert np.allclose(nulo_distancia, [f["cruce_nulo_mediana"] for f in gd["por_sujeto"]])

    referencia_cruce = {t: np.median(d[f"{t}_cruce"], axis=1) for t in ("mismo_grado", "uniforme", "barajada")}
    referencia_cruce["grados_y_distancia"] = nulo_distancia
    referencia_cruce["homotopicas_fijas"] = np.array([f["cruce_nulo_homotopicas_mediana"] for f in gd["por_sujeto"]])
    log_razon = {t: np.log(real / referencia_cruce[t]) for t in TIPOS_CRUCE}
    cruce = {}
    for t in TIPOS_CRUCE:
        x = log_razon[t]
        m = float(np.median(x))
        ic = bootstrap(x, np.median, rng)
        cruce[t] = {"mediana_log_razon": m, "ic95_log_razon": ic, "razon_mediana": float(np.exp(m)),
                    "ic95_razon": [float(np.exp(v)) for v in ic], "sujetos_real_antes": int(np.sum(x < 0)),
                    "n": int(len(x))}

    reales = np.array(ap["reales"], dtype=float)
    esperadas = {t: np.array(ap["referencias"][t], dtype=float) for t in TIPOS_ARISTAS[:4]}
    esperadas["homotopicas_fijas"] = np.array([f["esperadas_homotopicas_fijas"] for f in gd["por_sujeto"]])
    exceso = {t: reales - esperadas[t] for t in TIPOS_ARISTAS}
    aristas = {}
    for t in TIPOS_ARISTAS:
        x = exceso[t]
        aristas[t] = {"media_exceso": float(np.mean(x)), "ic95_exceso": bootstrap(x, np.mean, rng),
                      "esperadas_media": float(np.mean(esperadas[t])),
                      "razon_mediana": float(np.median(reales / esperadas[t])),
                      "sujetos_real_mayor": int(np.sum(x > 0)), "n": int(len(x))}
    assert abs(aristas["grados_y_distancia"]["razon_mediana"] - ap["resumen"]["grados_y_distancia"]["razon_mediana"]) < 1e-9

    efectivas = {capa: [1.0 / f["ipr_real"][capa] for f in su["por_sujeto"]] for capa in ("dmri", "fmri", "suma")}
    top = se["omega1"]["arbol_binarizado"]
    localizacion = {
        "regiones_efectivas_mediana": {capa: float(np.median(v)) for capa, v in efectivas.items()},
        "regiones_efectivas_p25_p75": {capa: [float(q) for q in np.percentile(v, [25, 75])] for capa, v in efectivas.items()},
        "regiones_totales": 68,
        "top10_comunes_con_suma_omega1_arbol_media": 10 * top["top10_suma_media"],
        "top10_comunes_con_suma_omega1_arbol_min": 10 * top["top10_suma_min"],
        "top10_comunes_con_fmri_omega1_arbol_media": 10 * top["top10_fmri_media"],
    }

    censura = {}
    for u in ("0.9", "0.95", "0.99"):
        reales_u = {f["sujeto"]: f["real"][u] for f in su["por_sujeto"]}
        nulos_u = [v[u] for f in su["por_sujeto"] for v in f["nulos_mediana"].values()]
        censura[u] = {
            "reales_bajo_la_grilla": sorted(s for s, v in reales_u.items() if np.isclose(v, GRILLA[0])),
            "reales_sobre_la_grilla": sorted(s for s, v in reales_u.items() if v > GRILLA[-1]),
            "medianas_nulas_censuradas": int(sum(np.isclose(v, GRILLA[0]) or v > GRILLA[-1] for v in nulos_u)),
        }

    res = {
        "B": B, "semilla": SEMILLA,
        "metodo": "bootstrap percentil remuestreando sujetos; cruce: mediana del log natural de la razón "
                  "real/nulo; aristas: media de la diferencia real menos esperadas",
        "cruce": cruce, "aristas": aristas, "localizacion": localizacion, "censura_sensibilidad_umbral": censura,
        "por_sujeto": {"sujetos": sujetos,
                       "log_razon_cruce": {t: [float(v) for v in log_razon[t]] for t in TIPOS_CRUCE},
                       "exceso_aristas": {t: [float(v) for v in exceso[t]] for t in TIPOS_ARISTAS}},
    }
    with open(os.path.join(AQUI, "efectos_bootstrap.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "por_sujeto"}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
