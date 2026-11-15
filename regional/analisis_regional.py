"""Análisis regional exploratorio: ¿la centralidad multicapa cambia cerca del tumor?

Para cada sujeto (árbol binarizado, como en Breedt y cols.) se calculan dos medidas por región:

- Reordenamiento: |rango de la versatilidad - rango de la centralidad de la suma|. Mide cuánto
  cambia la posición de la región al pasar de la suma de las capas al multicapa.
- Fracción estructural: parte de la centralidad multicapa de la región que queda en la capa dMRI,
  x_dMRI / (x_dMRI + x_fMRI). Es lo que el multicapa aporta por sobre la suma: la descomposición por
  capa.

Ambas se expresan como puntaje z frente a los 11 controles, región por región, y se correlacionan
(Spearman) con la distancia de cada región al borde tumoral dentro de cada paciente. La referencia
nula reasigna a cada paciente los mapas de distancia de otros pacientes (permutaciones de los 25
mapas), que conservan la estructura espacial de un tumor real. Acoplamiento principal: omega = 1
(configuración de Breedt); sensibilidad con omega adimensional 0,1 y 0,28.
"""

import json
import os
import sys

import numpy as np
import scipy.linalg as sl
from scipy.stats import rankdata, spearmanr, wilcoxon

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
sys.path.insert(0, os.path.join(AQUI, "..", "controles"))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import rutas
sys.path.insert(0, rutas.MODULOS)
import cargar as C
import multiplex as M
from controles import radio

RAIZ = rutas.RAIZ
PERMUTACIONES = 5000
SEMILLA = 20261124
CERCANAS = 5
ACOPLAMIENTOS = {"omega_1": None, "omega_adim_0.1": 0.1, "omega_adim_0.28": 0.28}


def medidas(a, b, omega):
    n = a.shape[0]
    s = np.block([[a, omega * np.eye(n)], [omega * np.eye(n), b]])
    x = np.abs(sl.eigh(s, subset_by_index=[2 * n - 1, 2 * n - 1], driver="evr")[1][:, 0])
    x = x / x.sum()
    v = x[:n] + x[n:]
    c = np.abs(np.linalg.eigh(a + b)[1][:, -1])
    reorden = np.abs(rankdata(v) - rankdata(c))
    fraccion = x[:n] / (x[:n] + x[n:])
    return reorden, fraccion


def rho_filas(x, y):
    """Spearman fila a fila entre dos matrices pacientes x regiones."""
    rx = np.apply_along_axis(rankdata, 1, x)
    ry = np.apply_along_axis(rankdata, 1, y)
    rx = rx - rx.mean(1, keepdims=True)
    ry = ry - ry.mean(1, keepdims=True)
    return (rx * ry).sum(1) / np.sqrt((rx ** 2).sum(1) * (ry ** 2).sum(1))


def main():
    rng = np.random.default_rng(SEMILLA)
    dist = np.load(os.path.join(AQUI, "distancias_tumor.npz"))
    pacientes = list(dist["pacientes"])
    borde = dist["distancia_borde"]
    grupos = C.grupos(RAIZ)
    controles = [s for s in C.sujetos(RAIZ) if grupos.get(s) == "control"]
    arboles = {}
    for s in controles + pacientes:
        sc, fc = C.cargar_sujeto(RAIZ, s)
        arboles[s] = (M.arbol_maximo(sc), M.arbol_maximo(fc))
    perms = [rng.permutation(len(pacientes)) for _ in range(PERMUTACIONES)]
    tipo = np.array([grupos.get(s) for s in pacientes])
    res = {"pacientes": len(pacientes), "controles": len(controles), "permutaciones": PERMUTACIONES}
    for nombre, wt in ACOPLAMIENTOS.items():
        val = {}
        for s, (a, b) in arboles.items():
            omega = 1.0 if wt is None else wt * (radio(a) + radio(b)) / 2.0
            val[s] = medidas(a, b, omega)
        bloque = {}
        for k, medida in enumerate(("reordenamiento", "fraccion_estructural")):
            ref = np.array([val[s][k] for s in controles])
            mu, sd = ref.mean(0), ref.std(0, ddof=1)
            # Una región sin variabilidad en los controles no tiene puntaje z: se excluye en vez de
            # asignarle cero. En estos datos no hay ninguna, pero queda registrado en el resultado.
            validas = sd > 0
            z = np.array([(val[s][k][validas] - mu[validas]) / sd[validas] for s in pacientes])
            dist_v = borde[:, validas]
            propia = rho_filas(dist_v, z)
            nulas = np.array([rho_filas(dist_v[p], z).mean() for p in perms])
            cerca = np.array([z[i][np.argsort(dist_v[i])[:CERCANAS]].mean() for i in range(len(pacientes))])
            resto = np.array([z[i][np.argsort(dist_v[i])[CERCANAS:]].mean() for i in range(len(pacientes))])
            obs = float(propia.mean())
            bloque[medida] = {
                "regiones_excluidas_por_sd_cero": int(np.sum(~validas)),
                "rho_distancia_z_media": obs,
                "rho_distancia_z_mediana": float(np.median(propia)),
                "pacientes_rho_negativo": int(np.sum(propia < 0)),
                "p_permutacion_bilateral": float((np.sum(np.abs(nulas - nulas.mean()) >= abs(obs - nulas.mean())) + 1)
                                                 / (PERMUTACIONES + 1)),
                "nulo_media": float(nulas.mean()), "nulo_p2.5_p97.5": [float(np.percentile(nulas, 2.5)),
                                                                        float(np.percentile(nulas, 97.5))],
                "z_5_mas_cercanas_mediana": float(np.median(cerca)), "z_resto_mediana": float(np.median(resto)),
                "wilcoxon_cercanas_vs_resto_p": float(wilcoxon(cerca - resto).pvalue),
                "por_tipo": {t: {"n": int(np.sum(tipo == t)), "rho_mediana": float(np.median(propia[tipo == t])),
                                 "z_cercanas_menos_resto_mediana": float(np.median((cerca - resto)[tipo == t]))}
                             for t in ("glioma", "meningioma")},
            }
        res[nombre] = bloque
    with open(os.path.join(AQUI, "analisis_regional.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1)
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
