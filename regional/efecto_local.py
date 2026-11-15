"""¿Las máscaras sirven para el análisis regional? Efecto local del tumor en la conectividad.

Si las máscaras y los centroides comparten el sistema de coordenadas de cada sujeto, las
regiones más cercanas al tumor deberían mostrar la mayor desviación estructural respecto de
los controles. Para cada paciente y región se calcula la fuerza (suma de pesos) de la matriz
completa, se expresa como puntaje z frente a los 11 controles (en escala logarítmica) y se
correlaciona con la distancia de la región al borde tumoral. La referencia nula usa, para
cada paciente, los vectores de distancia de los otros pacientes (mismo atlas, otro tumor).
Se repite con la fuerza funcional (suma de correlaciones positivas).
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

RAIZ = rutas.RAIZ
CERCANAS = 5


def fuerza(m, capa):
    if capa == "estructural":
        return np.log(m.sum(1) + 1e-6)
    return np.clip(m, 0, None).sum(1)


def main():
    dist = np.load(os.path.join(AQUI, "distancias_tumor.npz"))
    pacientes = list(dist["pacientes"])
    borde = dist["distancia_borde"]
    grupos = C.grupos(RAIZ)
    datos = {s: C.cargar_sujeto(RAIZ, s) for s in C.sujetos(RAIZ)}
    controles = [s for s in datos if grupos.get(s) == "control"]
    res = {"controles": len(controles), "pacientes": len(pacientes)}
    for k, capa in enumerate(("estructural", "funcional")):
        ref = np.array([fuerza(datos[s][k], capa) for s in controles])
        mu, sd = ref.mean(0), ref.std(0, ddof=1)
        z = np.array([(fuerza(datos[s][k], capa) - mu) / sd for s in pacientes])
        propia = np.array([spearmanr(borde[i], z[i]).statistic for i in range(len(pacientes))])
        cruzada = np.array([np.mean([spearmanr(borde[j], z[i]).statistic for j in range(len(pacientes)) if j != i])
                            for i in range(len(pacientes))])
        cerca = np.array([z[i][np.argsort(borde[i])[:CERCANAS]].mean() for i in range(len(pacientes))])
        resto = np.array([z[i][np.argsort(borde[i])[CERCANAS:]].mean() for i in range(len(pacientes))])
        por_tipo = {}
        for tipo in ("glioma", "meningioma"):
            m = np.array([grupos.get(s) == tipo for s in pacientes])
            por_tipo[tipo] = {"n": int(m.sum()), "rho_mediana": float(np.median(propia[m])),
                              "z_cercanas_mediana": float(np.median(cerca[m])), "z_resto_mediana": float(np.median(resto[m]))}
        res[capa] = {
            "rho_distancia_z_mediana": float(np.median(propia)),
            "pacientes_rho_positivo": int(np.sum(propia > 0)),
            "rho_con_distancias_de_otros_pacientes_mediana": float(np.median(cruzada)),
            "wilcoxon_propia_vs_cruzada_p": float(wilcoxon(propia - cruzada).pvalue),
            "z_regiones_cercanas_mediana": float(np.median(cerca)),
            "z_resto_mediana": float(np.median(resto)),
            "wilcoxon_cercanas_vs_resto_p": float(wilcoxon(cerca - resto).pvalue),
            "por_tipo": por_tipo,
        }
    with open(os.path.join(AQUI, "efecto_local.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1)
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
