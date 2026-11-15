"""Tipo, tamaño y adquisición: ¿se asocian con la centralidad multicapa en los pacientes?

Usa las medidas por sujeto ya calculadas en el árbol binarizado:
- cruce real (controles/explicacion_espectral.npz), en escala logarítmica;
- variación total entre la centralidad multicapa y la de la suma en omega = 1
  (controles/sensibilidad.json).

Contrastes (exploratorios, sin corrección por multiplicidad):
1. Glioma frente a meningioma: Mann-Whitney sin ajustar y ANCOVA de rangos ajustada por edad (en
   rangos) y sexo, como en controles/sensibilidad.py, con p de permutación de Freedman-Lane sobre los
   rangos. Se informa la diferencia de edad entre los tipos, porque con 25 pacientes tipo tumoral y
   edad no se pueden separar.
2. Volumen tumoral del registro (cm³): Spearman en los 25 pacientes.
3. TR de la fMRI (2100 frente a 2400 ms), que cambió durante la adquisición: Mann-Whitney en los 36.
"""

import csv
import json
import os
import sys

import numpy as np
from scipy.stats import mannwhitneyu, rankdata, spearmanr
from scipy.stats import t as student

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(AQUI))
import rutas

BASE = os.path.dirname(AQUI)
CTRL = os.path.join(BASE, "controles")
SEMILLA = 20261114
PERMUTACIONES = 10000


def participantes():
    with open(os.path.join(rutas.RAIZ, "participants.tsv"), encoding="utf-8") as fh:
        filas = list(csv.reader(fh, delimiter="\t"))
    cabecera = [c.strip() for c in filas[0]]
    return {f[0].strip(): dict(zip(cabecera, [x.strip() for x in f])) for f in filas[1:]}


def regresion(y, x):
    """Coeficientes y p bilaterales de MCO con intercepto."""
    x1 = np.column_stack([np.ones(len(y)), x])
    coef, *_ = np.linalg.lstsq(x1, y, rcond=None)
    resid = y - x1 @ coef
    gl = len(y) - x1.shape[1]
    ee = np.sqrt(np.diag(resid @ resid / gl * np.linalg.inv(x1.T @ x1)))
    return coef, 2 * student.sf(np.abs(coef / ee), gl)


def freedman_lane(y, grupo, covariables, rng):
    """p de permutación del coeficiente de grupo ajustado: se permutan los residuos del modelo reducido."""
    z1 = np.column_stack([np.ones(len(y)), covariables])
    coef_z, *_ = np.linalg.lstsq(z1, y, rcond=None)
    ajuste, resid = z1 @ coef_z, y - z1 @ coef_z
    x = np.column_stack([grupo, covariables])
    obs = abs(regresion(y, x)[0][1])
    cuenta = sum(abs(regresion(ajuste + rng.permutation(resid), x)[0][1]) >= obs for _ in range(PERMUTACIONES))
    return float((cuenta + 1) / (PERMUTACIONES + 1))


def main():
    rng = np.random.default_rng(SEMILLA)
    ee = json.load(open(os.path.join(CTRL, "explicacion_espectral.json"), encoding="utf-8"))
    sen = json.load(open(os.path.join(CTRL, "sensibilidad.json"), encoding="utf-8"))
    sujetos = ee["sujetos"]
    grupo = np.array(sen["asociacion_cruce"]["grupo_por_sujeto"])
    assert len(grupo) == len(sujetos)
    cruce = np.log(np.load(os.path.join(CTRL, "explicacion_espectral.npz"))["real_cruce"])
    vt = np.array(sen["omega1"]["arbol_binarizado"]["vt_por_sujeto"])
    p = participantes()
    assert all((p[s]["tumor type & grade"] == "none") == (g == "control") for s, g in zip(sujetos, grupo))
    edad = np.array([float(p[s]["age"]) for s in sujetos])
    sexo = np.array([1.0 if p[s]["sex"] == "M" else 0.0 for s in sujetos])
    volumen = np.array([float(p[s]["tumor size (cub cm)"]) for s in sujetos])
    tr = np.array([float(p[s]["fmri TR"]) for s in sujetos])

    tumor = (grupo == "glioma") | (grupo == "meningioma")
    glioma = (grupo[tumor] == "glioma").astype(float)
    cov = np.column_stack([rankdata(edad[tumor]), sexo[tumor]])
    res = {"pacientes": int(tumor.sum()), "glioma": int((grupo == "glioma").sum()),
           "meningioma": int((grupo == "meningioma").sum()),
           "edad_mediana": {g: float(np.median(edad[grupo == g])) for g in ("glioma", "meningioma", "control")},
           "edad_glioma_vs_meningioma_p": float(mannwhitneyu(edad[grupo == "glioma"], edad[grupo == "meningioma"])[1]),
           "tipo": {}}
    for nombre, y in (("variacion_total_omega1", vt), ("cruce", cruce)):
        rangos = rankdata(y[tumor])
        _, p_con = regresion(rangos, np.column_stack([glioma, cov]))
        res["tipo"][nombre] = {
            "mediana_glioma": float(np.median(y[grupo == "glioma"])),
            "mediana_meningioma": float(np.median(y[grupo == "meningioma"])),
            "mannwhitney_p": float(mannwhitneyu(y[grupo == "glioma"], y[grupo == "meningioma"])[1]),
            "p_ancova_rangos_edad_sexo": float(p_con[1]),
            "p_permutacion_edad_sexo": freedman_lane(rangos, glioma, cov, rng),
            "p_edad_en_el_modelo": float(p_con[2]),
        }
    res["tipo"]["cruce"]["mediana_glioma"] = float(np.exp(res["tipo"]["cruce"]["mediana_glioma"]))
    res["tipo"]["cruce"]["mediana_meningioma"] = float(np.exp(res["tipo"]["cruce"]["mediana_meningioma"]))
    res["volumen"] = {nombre: dict(zip(("rho", "p"), map(float, spearmanr(y[tumor], volumen[tumor]))))
                      for nombre, y in (("variacion_total_omega1", vt), ("cruce", cruce))}
    res["tr_fmri"] = {"sujetos_2100": int((tr == 2100).sum()), "sujetos_2400": int((tr == 2400).sum()),
                      "cruce_mannwhitney_p": float(mannwhitneyu(cruce[tr == 2100], cruce[tr == 2400])[1]),
                      "variacion_total_mannwhitney_p": float(mannwhitneyu(vt[tr == 2100], vt[tr == 2400])[1])}
    with open(os.path.join(AQUI, "tumor.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1)
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
