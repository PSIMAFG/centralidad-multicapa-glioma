"""Verificación independiente de las cifras principales.

Reimplementa las matemáticas sin usar multiplex.py: árbol de expansión máxima con networkx,
supra-adyacencia y autovector con numpy, rangos con scipy. Solo reutiliza la lectura de
archivos (cargar.py). Compara cada cifra con la de los JSON y termina con error si alguna
difiere más de la tolerancia.
"""

import json
import os
import sys

import networkx as nx
import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import rutas
sys.path.insert(0, rutas.MODULOS)
import cargar as C

RAIZ = rutas.RAIZ
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GRILLA = np.logspace(-3, 1.5, 181)


def arbol(w, binario=True):
    g = nx.Graph()
    n = w.shape[0]
    g.add_nodes_from(range(n))
    i, j = np.triu_indices(n, 1)
    for a, b in zip(i, j):
        if w[a, b] > 0:
            g.add_edge(int(a), int(b), weight=float(w[a, b]))
    t = nx.maximum_spanning_tree(g, weight="weight", algorithm="kruskal")
    m = np.zeros((n, n))
    for a, b, d in t.edges(data=True):
        m[a, b] = m[b, a] = 1.0 if binario else d["weight"]
    return m


def principal(m):
    val, vec = np.linalg.eigh(m)
    v = np.abs(vec[:, np.argmax(val)])
    return v / v.sum()


def supra(a, b, w):
    n = a.shape[0]
    return np.block([[a, w * np.eye(n)], [w * np.eye(n), b]])


def versatilidad(a, b, w):
    x = principal(supra(a, b, w))
    n = a.shape[0]
    return x[:n] + x[n:], x[n:].sum()


def radio(m):
    return np.max(np.abs(np.linalg.eigvalsh(m)))


def main():
    r = json.load(open(os.path.join(BASE, "resultados", "resultados.json"), encoding="utf-8"))
    ctl = json.load(open(os.path.join(BASE, "controles", "controles.json"), encoding="utf-8"))
    per = json.load(open(os.path.join(BASE, "controles", "permutacion_cruce.json"), encoding="utf-8"))
    datos = [C.cargar_sujeto(RAIZ, s) for s in C.sujetos(RAIZ)]
    datos = [d for d in datos if d is not None]

    rho_fmri_cruda, masa_fmri, rho_suma_reesc, w1_adim, rho1_bin, cruces, jac = [], [], [], [], [], [], []
    for sc, fc in datos:
        v, mf = versatilidad(sc, fc, 1.0)
        rho_fmri_cruda.append(spearmanr(v, principal(fc)).statistic)
        masa_fmri.append(mf)
        a, b = sc / radio(sc), fc / radio(fc)
        rho_suma_reesc.append(spearmanr(versatilidad(a, b, 1.0)[0], principal(a + b)).statistic)
        ta, tb = arbol(sc), arbol(fc)
        ref = (radio(ta) + radio(tb)) / 2.0
        w1_adim.append(1.0 / ref)
        cs = principal(ta + tb)
        rho1_bin.append(spearmanr(versatilidad(ta, tb, 1.0)[0], cs).statistic)
        fila = np.array([spearmanr(versatilidad(ta, tb, w * ref)[0], cs).statistic for w in GRILLA])
        k = int(np.argmax(fila >= 0.95))
        lo = np.interp(0.95, [fila[k - 1], fila[k]], np.log10(GRILLA[[k - 1, k]]))
        cruces.append(10 ** lo)
        ea = set(zip(*np.nonzero(np.triu(ta))))
        eb = set(zip(*np.nonzero(np.triu(tb))))
        jac.append(len(ea & eb) / len(ea | eb))
    rho1_bin = np.array(rho1_bin)

    comprobaciones = [
        ("rho con la fMRI sola, tal como vienen", np.mean(rho_fmri_cruda), r["construcciones"]["cruda"]["rho_w1_vs_fmri"], 1e-4),
        ("masa del autovector en la capa fMRI", np.mean(masa_fmri), r["construcciones"]["cruda"]["fraccion_autovector_fmri_w1"], 1e-4),
        ("rho con la suma, reescaladas", np.mean(rho_suma_reesc), r["construcciones"]["reescalada"]["rho_w1_vs_suma"], 1e-4),
        ("omega = 1 en unidades adimensionales (arbol)", np.median(w1_adim), ctl["binarizado_omega1_adimensional_mediana"], 1e-6),
        ("rho con la suma en omega = 1 (arbol)", np.mean(rho1_bin), r["construcciones"]["arbol_binarizado"]["rho_w1_vs_suma"], 1e-4),
        ("sujetos con rho >= 0,90", np.sum(rho1_bin >= 0.90), ctl["umbrales_omega1_binarizado"]["0.90"]["k"], 0),
        ("sujetos con rho >= 0,95", np.sum(rho1_bin >= 0.95), ctl["umbrales_omega1_binarizado"]["0.95"]["k"], 0),
        ("sujetos con rho >= 0,99", np.sum(rho1_bin >= 0.99), ctl["umbrales_omega1_binarizado"]["0.99"]["k"], 0),
        ("cruce mediano del arbol (grilla 4 veces mas fina)", np.median(cruces), ctl["cruce_adimensional"]["binarizado"]["mediana"], 0.01),
        ("aristas compartidas por los dos arboles (Jaccard)", np.mean(jac), per["jaccard_real_media"], 1e-9),
    ]
    fallas = 0
    for nombre, propio, informado, tol in comprobaciones:
        ok = abs(float(propio) - float(informado)) <= tol
        fallas += int(not ok)
        print(f"{'ok   ' if ok else 'FALLA'} {nombre}: independiente {float(propio):.6f} · informado {float(informado):.6f}")
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "verificacion_independiente.json"), "w",
              encoding="utf-8") as fh:
        json.dump({n: {"independiente": float(a), "informado": float(b), "tolerancia": t} for n, a, b, t in comprobaciones},
                  fh, ensure_ascii=False, indent=1)
    if fallas:
        sys.exit(1)


if __name__ == "__main__":
    main()
