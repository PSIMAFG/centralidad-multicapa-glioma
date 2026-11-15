"""Demostración SHAP: ¿a qué atribuye importancia un clasificador glioma frente a meningioma?

Ilustra la advertencia del póster con un modelo entrenado: un rasgo multicapa puede recibir
importancia SHAP aunque su rendimiento venga de la composición de los grupos. No evalúa utilidad
clínica.

- Pacientes: los 25 con tumor (11 gliomas y 14 meningiomas); clase positiva, glioma.
- Rasgos multicapa por sujeto (árbol binarizado): log del cruce y log P
  (controles/explicacion_espectral.npz) y variación total respecto de la suma en omega = 1
  (controles/sensibilidad.json). El acoplamiento SC-FC global no entra: según la taxonomía del
  póster pertenece a relacionar las dos matrices, no a la red multicapa. Se agrega una variable de
  ruido puro como referencia de cuánta importancia recibe el azar.
- Como tipo tumoral, edad y sexo están confundidos en esta muestra, se informa también la
  correlación de cada rasgo con la edad dentro de cada tipo de tumor.
- Conjuntos: A, multicapa; B, multicapa + edad + sexo; C, edad + sexo; D, multicapa sin la parte
  lineal explicada por edad y sexo, ajustada dentro de cada pliegue de entrenamiento (regresión de
  confusores por pliegue).
- Modelos: regresión logística estandarizada y bosque aleatorio poco profundo.
- Validación: 30 repeticiones de validación cruzada estratificada de 5 pliegues; AUC de las
  predicciones fuera de pliegue. Los percentiles 2,5 y 97,5 describen la variación entre
  repeticiones, no son intervalos de confianza.
- Incertidumbre del AUC: IC 95 % por bootstrap estratificado de sujetos sobre las predicciones fuera
  de pliegue de las 30 repeticiones. Las diferencias entre conjuntos (multicapa menos edad y sexo;
  todos menos edad y sexo) se calculan sobre las mismas 30 particiones para todos los conjuntos y
  ambos modelos, pareadas por sujeto en el bootstrap. Usan generadores aparte, así que no alteran el
  resto. Cada pliegue de prueba tiene 2 o 3 gliomas y 2 o 3 meningiomas.
- Estabilidad de la atribución: |SHAP| medio por repetición (concordancia del orden de importancia
  entre repeticiones) e importancia por permutación (caída del AUC fuera de pliegue al permutar cada
  variable dentro de cada pliegue de prueba, con los modelos ya entrenados). SHAP mide cuánto usa el
  modelo una variable; la permutación, cuánto rendimiento depende de ella.
- p por permutación de etiquetas repitiendo el procedimiento completo, incluido el ajuste por edad
  y sexo del conjunto D. Cada permutación usa 3 repeticiones en vez de 30: la distribución nula
  queda más ancha, lo que hace conservador el p.
- Valores de Shapley intervencionales exactos, fuera de pliegue, con el pliegue de entrenamiento
  como fondo, en las 10 primeras repeticiones. Con cinco a siete variables se pueden enumerar todas
  las coaliciones, así que no hace falta aproximarlos (equivalen a KernelSHAP o TreeSHAP
  intervencional con el mismo fondo). Se comprueba la aditividad.
"""

import csv
import json
import os
import sys
from math import factorial

import numpy as np
from scipy.stats import rankdata, spearmanr
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(AQUI))
import rutas

CTRL = os.path.join(os.path.dirname(AQUI), "controles")
SEMILLA = 20261114
REPETICIONES = 30
REPETICIONES_SHAP = 10
REPETICIONES_PERMUTACION = 3
PERMUTACIONES = {"logistica": 300, "bosque_aleatorio": 100}
BOOTSTRAP = 2000
PLIEGUES = 5
MULTICAPA = ["log_cruce", "variacion_total", "log_P"]


def datos():
    ee = json.load(open(os.path.join(CTRL, "explicacion_espectral.json"), encoding="utf-8"))
    sen = json.load(open(os.path.join(CTRL, "sensibilidad.json"), encoding="utf-8"))
    npz = np.load(os.path.join(CTRL, "explicacion_espectral.npz"))
    suj = ee["sujetos"]
    grupo = np.array(sen["asociacion_cruce"]["grupo_por_sujeto"])
    with open(os.path.join(rutas.RAIZ, "participants.tsv"), encoding="utf-8") as fh:
        filas = list(csv.reader(fh, delimiter="\t"))
    cab = [c.strip() for c in filas[0]]
    part = {f[0].strip(): dict(zip(cab, [x.strip() for x in f])) for f in filas[1:]}
    tabla = {
        "log_cruce": np.log(npz["real_cruce"]),
        "variacion_total": np.array(sen["omega1"]["arbol_binarizado"]["vt_por_sujeto"]),
        "log_P": np.log(npz["real_P"]),
        "edad": np.array([float(part[s]["age"]) for s in suj]),
        "sexo_m": np.array([1.0 if part[s]["sex"] == "M" else 0.0 for s in suj]),
    }
    pac = grupo != "control"
    y = (grupo[pac] == "glioma").astype(int)
    return {k: v[pac] for k, v in tabla.items()}, y


def shapley_exacto(f, x_expl, x_fondo):
    """phi[i, j]: contribución exacta de la variable j a f(x_i) - E_fondo[f], enumerando las coaliciones."""
    n, m = x_expl.shape
    mascaras = np.array([[(c >> j) & 1 for j in range(m)] for c in range(2 ** m)], dtype=bool)
    v = np.empty((n, 2 ** m))
    for i in range(n):
        filas = np.where(mascaras[:, None, :], x_expl[i][None, None, :], x_fondo[None, :, :])
        v[i] = f(filas.reshape(-1, m)).reshape(2 ** m, len(x_fondo)).mean(1)
    tam = mascaras.sum(1)
    phi = np.zeros((n, m))
    for j in range(m):
        sin = np.flatnonzero(~mascaras[:, j])
        peso = np.array([factorial(t) * factorial(m - t - 1) / factorial(m) for t in tam[sin]])
        phi[:, j] = ((v[:, sin + (1 << j)] - v[:, sin]) * peso).sum(1)
    return phi, v[:, 0], v[:, -1]


def residualizar(x_ent, x_pru, z_ent, z_pru, columnas):
    """Quita a las columnas indicadas la parte lineal explicada por z (ajuste solo en entrenamiento)."""
    x_ent, x_pru = x_ent.copy(), x_pru.copy()
    for c in columnas:
        r = LinearRegression().fit(z_ent, x_ent[:, c])
        x_ent[:, c] -= r.predict(z_ent)
        x_pru[:, c] -= r.predict(z_pru)
    return x_ent, x_pru


def evaluar(modelo, x, y, rng, repeticiones, con_shap=True, residuo=None, semillas=None, rng_importancia=None):
    """AUC fuera de pliegue con validación cruzada estratificada repetida y SHAP fuera de pliegue.

    Con semillas, usa esas particiones (las mismas para todos los conjuntos que se comparan) y no
    consume rng. Con rng_importancia, calcula además la importancia por permutación.
    """
    aucs, phis, valores, probs, error_aditividad = [], [], [], [], 0.0
    shap_por_repeticion, caidas = [], []
    for r in range(repeticiones):
        semilla = int(rng.integers(1 << 30)) if semillas is None else semillas[r]
        cv = StratifiedKFold(PLIEGUES, shuffle=True, random_state=semilla)
        prob = np.empty(len(y))
        permutada = None if rng_importancia is None else np.empty((x.shape[1], len(y)))
        phi_repeticion = []
        for ent, pru in cv.split(x, y):
            x_ent, x_pru = x[ent], x[pru]
            if residuo is not None:
                z, cols = residuo
                x_ent, x_pru = residualizar(x_ent, x_pru, z[ent], z[pru], cols)
            m = clone(modelo).fit(x_ent, y[ent])
            prob[pru] = m.predict_proba(x_pru)[:, 1]
            if permutada is not None:
                for j in range(x.shape[1]):
                    xp = x_pru.copy()
                    xp[:, j] = rng_importancia.permutation(xp[:, j])
                    permutada[j, pru] = m.predict_proba(xp)[:, 1]
            if con_shap and r < REPETICIONES_SHAP:
                phi, base, completo = shapley_exacto(lambda a: m.predict_proba(a)[:, 1], x_pru, x_ent)
                error_aditividad = max(error_aditividad, float(np.abs(base + phi.sum(1) - completo).max()))
                phis.append(phi)
                valores.append(x_pru)
                phi_repeticion.append(phi)
        auc = roc_auc_score(y, prob)
        aucs.append(auc)
        probs.append(prob)
        if phi_repeticion:
            shap_por_repeticion.append(np.abs(np.vstack(phi_repeticion)).mean(0))
        if permutada is not None:
            caidas.append([auc - roc_auc_score(y, permutada[j]) for j in range(x.shape[1])])
    salida = {"auc_media": float(np.mean(aucs)), "auc_p2_5": float(np.percentile(aucs, 2.5)),
              "auc_p97_5": float(np.percentile(aucs, 97.5)), "_probs": np.array(probs)}
    if con_shap:
        phi, val = np.vstack(phis), np.vstack(valores)
        salida["shap_abs_medio"] = np.abs(phi).mean(0).tolist()
        salida["direccion"] = [float(spearmanr(val[:, j], phi[:, j])[0]) for j in range(phi.shape[1])]
        salida["error_aditividad_maximo"] = error_aditividad
        salida["estabilidad_shap"] = estabilidad(np.array(shap_por_repeticion))
    if caidas:
        c = np.array(caidas)
        salida["importancia_permutacion"] = {"caida_auc_media": c.mean(0).tolist(),
                                             "caida_auc_p2_5": np.percentile(c, 2.5, axis=0).tolist(),
                                             "caida_auc_p97_5": np.percentile(c, 97.5, axis=0).tolist()}
    return salida


def estabilidad(imp):
    """Concordancia del orden de importancia entre repeticiones (filas: repeticiones; columnas: variables).

    Devuelve la ρ de Spearman media entre pares de repeticiones y, para la variable más importante en
    promedio, la fracción de repeticiones en que también es la primera.
    """
    pares = [spearmanr(imp[a], imp[b])[0] for a in range(len(imp)) for b in range(a + 1, len(imp))]
    primera = int(np.argmax(imp.mean(0)))
    return {"rho_media_entre_repeticiones": float(np.mean(pares)), "variable_primera": primera,
            "fraccion_repeticiones_misma_primera": float(np.mean(imp.argmax(1) == primera))}


def auc_rangos(p, y):
    """AUC de cada fila de p (repeticiones x sujetos) por la U de Mann-Whitney, con rangos promedio."""
    rangos = rankdata(p, axis=1)
    n1 = int(y.sum())
    n0 = len(y) - n1
    return (rangos[:, y == 1].sum(1) - n1 * (n1 + 1) / 2) / (n1 * n0)


def bootstrap_auc(probs, y, rng, otra=None, total=BOOTSTRAP):
    """AUC medio entre repeticiones en remuestreos estratificados de sujetos; con otra, la diferencia pareada."""
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    valores = np.empty(total)
    for b in range(total):
        s = np.r_[rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]
        v = auc_rangos(probs[:, s], y[s]).mean()
        if otra is not None:
            v -= auc_rangos(otra[:, s], y[s]).mean()
        valores[b] = v
    return valores


def clases_por_pliegue(y):
    """Mínimo y máximo de gliomas y meningiomas por pliegue de prueba (no depende de la semilla)."""
    cuentas = [(int(y[pru].sum()), int(len(pru) - y[pru].sum()))
               for _, pru in StratifiedKFold(PLIEGUES, shuffle=True, random_state=0).split(y, y)]
    g, m = zip(*cuentas)
    return {"gliomas": [min(g), max(g)], "meningiomas": [min(m), max(m)]}


def p_permutacion(modelo, x, y, auc_obs, rng, total, residuo=None):
    nulos = [evaluar(modelo, x, rng.permutation(y), rng, REPETICIONES_PERMUTACION, con_shap=False,
                     residuo=residuo)["auc_media"] for _ in range(total)]
    return float((np.sum(np.array(nulos) >= auc_obs) + 1) / (total + 1))


def main():
    rng = np.random.default_rng(SEMILLA)
    tabla, y = datos()
    tabla["ruido"] = rng.normal(size=len(y))
    conjuntos = {
        "A_multicapa": MULTICAPA + ["ruido"],
        "B_multicapa_mas_edad_sexo": MULTICAPA + ["edad", "sexo_m", "ruido"],
        "C_edad_sexo": ["edad", "sexo_m", "ruido"],
    }
    modelos = {
        "logistica": make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)),
        "bosque_aleatorio": RandomForestClassifier(n_estimators=100, max_depth=3, min_samples_leaf=2,
                                                   random_state=SEMILLA),
    }
    tipos = {"glioma": y == 1, "meningioma": y == 0}
    res = {
        "n_glioma": int(y.sum()), "n_meningioma": int((1 - y).sum()), "semilla": SEMILLA,
        "repeticiones": REPETICIONES, "repeticiones_con_shap": REPETICIONES_SHAP,
        "repeticiones_por_permutacion": REPETICIONES_PERMUTACION, "permutaciones": PERMUTACIONES,
        "demografia": {t: {"edad_mediana": float(np.median(tabla["edad"][m])),
                           "mujeres": int(np.sum(tabla["sexo_m"][m] == 0)), "n": int(m.sum())}
                       for t, m in tipos.items()},
        "correlacion_con_edad_en_pacientes": {k: dict(zip(("rho", "p"), map(float, spearmanr(tabla[k], tabla["edad"]))))
                                              for k in MULTICAPA},
        "correlacion_con_edad_por_tipo": {t: {k: dict(zip(("rho", "p"), map(float, spearmanr(tabla[k][m], tabla["edad"][m]))))
                                              for k in MULTICAPA} for t, m in tipos.items()},
        "bootstrap": BOOTSTRAP,
        "pliegue_prueba": clases_por_pliegue(y),
        "particiones_comunes": {"semilla": SEMILLA + 2, "repeticiones": REPETICIONES},
        "resultados": {},
        "comparaciones": {},
    }
    rng_boot = np.random.default_rng(SEMILLA + 1)
    rng_comun = np.random.default_rng(SEMILLA + 2)
    rng_importancia = np.random.default_rng(SEMILLA + 3)
    semillas_comunes = [int(rng_comun.integers(1 << 30)) for _ in range(REPETICIONES)]
    for nm, modelo in modelos.items():
        for nc, cols in conjuntos.items():
            x = np.column_stack([tabla[c] for c in cols])
            r = evaluar(modelo, x, y, rng, REPETICIONES, rng_importancia=rng_importancia)
            r["variables"] = cols
            r["p_permutacion"] = p_permutacion(modelo, x, y, r["auc_media"], rng, PERMUTACIONES[nm])
            res["resultados"][f"{nm}/{nc}"] = r
            print(nm, nc, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()
                           if k not in ("variables", "_probs")}, flush=True)
        cols = MULTICAPA + ["ruido"]
        x = np.column_stack([tabla[c] for c in cols])
        z = np.column_stack([tabla["edad"], tabla["sexo_m"]])
        residuo = (z, list(range(len(MULTICAPA))))
        r = evaluar(modelo, x, y, rng, REPETICIONES, residuo=residuo, rng_importancia=rng_importancia)
        r["variables"] = [c + "_sin_edad_sexo" if c != "ruido" else c for c in cols]
        r["p_permutacion"] = p_permutacion(modelo, x, y, r["auc_media"], rng, PERMUTACIONES[nm], residuo=residuo)
        res["resultados"][f"{nm}/D_multicapa_residualizada"] = r
        print(nm, "D", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()
                        if k not in ("variables", "_probs")}, flush=True)
        for nc in list(conjuntos) + ["D_multicapa_residualizada"]:
            r = res["resultados"][f"{nm}/{nc}"]
            r["auc_ic95_bootstrap"] = np.percentile(bootstrap_auc(r["_probs"], y, rng_boot), [2.5, 97.5]).tolist()
        comunes = {nc: evaluar(modelo, np.column_stack([tabla[c] for c in cols]), y, None, REPETICIONES,
                               con_shap=False, semillas=semillas_comunes)["_probs"]
                   for nc, cols in conjuntos.items()}
        for nombre, a, b in (("A_menos_C", "A_multicapa", "C_edad_sexo"), ("B_menos_C", "B_multicapa_mas_edad_sexo", "C_edad_sexo")):
            por_repeticion = auc_rangos(comunes[a], y) - auc_rangos(comunes[b], y)
            d = bootstrap_auc(comunes[a], y, rng_boot, otra=comunes[b])
            res["comparaciones"][f"{nm}/{nombre}"] = {"diferencia": float(por_repeticion.mean()),
                                                      "ic95_bootstrap": np.percentile(d, [2.5, 97.5]).tolist(),
                                                      "fraccion_menor_o_igual_a_0": float(np.mean(d <= 0)),
                                                      "repeticiones_con_diferencia_positiva": int(np.sum(por_repeticion > 0))}
    for r in res["resultados"].values():
        r.pop("_probs")
    with open(os.path.join(AQUI, "shap_tipo_tumoral.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
