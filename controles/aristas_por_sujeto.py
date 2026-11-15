"""Reúne, sujeto a sujeto, las aristas compartidas reales y las esperadas bajo cada referencia.

Referencias, de menos a más restrictiva:
- árboles con el mismo grado por región (secuencia de Prüfer permutada, 500 por sujeto);
- referencia lobar analítica (cada capa conserva cuántas aristas tiene por categoría anatómica);
- árboles sesgados por distancia, con la longitud media real (sin conservar grados);
- árboles con el mismo grado por región y la misma longitud media (cadena de Markov).
Lee los resultados ya calculados y escribe aristas_por_sujeto.json para la figura y el póster.
"""

import json
import os
import sys

import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import rutas
sys.path.insert(0, rutas.MODULOS)
sys.path.insert(0, AQUI)
import cargar as C
import multiplex as M
from validacion_P import categorias, esperadas_estratificadas, regiones

RAIZ = rutas.RAIZ


def leer(nombre):
    with open(os.path.join(AQUI, nombre), encoding="utf-8") as fh:
        return json.load(fh)


def main():
    d = np.load(os.path.join(AQUI, "explicacion_espectral.npz"))
    esp = leer("nulo_espacial_sujeto.json")["por_sujeto"]
    gd = leer("nulo_grados_distancia.json")["por_sujeto"]
    cat = categorias(regiones())
    sujetos = [s for s in C.sujetos(RAIZ) if C.cargar_sujeto(RAIZ, s) is not None]
    reales, grados, lobar = [], [], []
    for k, s in enumerate(sujetos):
        sc, fc = C.cargar_sujeto(RAIZ, s)
        a, b = M.arbol_maximo(sc), M.arbol_maximo(fc)
        reales.append(int(((np.triu(a, 1) > 0) & (np.triu(b, 1) > 0)).sum()))
        tot = int((np.triu(a, 1) > 0).sum() + (np.triu(b, 1) > 0).sum())
        j = d["mismo_grado_jaccard"][k]
        grados.append(float(np.mean(j * tot / (1 + j))))
        lobar.append(float(esperadas_estratificadas(a, b, cat)))
    assert [f["sujeto"] for f in esp] == sujetos and [f["sujeto"] for f in gd] == sujetos
    assert [f["reales"] for f in gd] == reales
    res = {"sujetos": sujetos, "reales": reales, "referencias": {
        "mismo_grado": grados,
        "lobar": lobar,
        "distancia": [f["nulo"] for f in esp],
        "grados_y_distancia": [f["esperadas"] for f in gd],
    }}
    resumen = {}
    r = np.array(reales, dtype=float)
    for k, v in res["referencias"].items():
        v = np.array(v)
        resumen[k] = {"media": float(v.mean()), "razon_mediana": float(np.median(r / v)),
                      "sujetos_real_mayor": int(np.sum(r > v))}
    res["resumen"] = resumen
    with open(os.path.join(AQUI, "aristas_por_sujeto.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1)
    print(json.dumps({"reales_media": float(r.mean()), **resumen}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
