"""Comprobación visual y numérica de que máscara, centroides y T1 comparten coordenadas.

Usa los T1 de dos pacientes (sub-PAT01 y sub-PAT26). Para cada uno:
- compara la grilla y la matriz afín del T1 con las de la máscara tumoral. Tienen la misma
  grilla, pero la máscara está guardada con el eje izquierda-derecha invertido y su cabecera
  lo declara: hay que pasar por coordenadas de mundo con la afín de cada archivo. Leer los dos
  arreglos vóxel a vóxel deja el tumor en el hemisferio equivocado;
- compara la intensidad T1 dentro de la máscara llevada por coordenadas de mundo y dentro de
  la máscara leída vóxel a vóxel (en espejo);
- dibuja cortes del T1 con los dos contornos y los centroides de centres.txt cercanos al corte.

Además guarda fig/mascara_t1.png, un corte axial de sub-PAT26 con la máscara leída de las dos formas.
"""

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from skimage.filters import threshold_otsu

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import rutas
sys.path.insert(0, rutas.MODULOS)
import cargar as C
from lectura import leer_centros, leer_nifti, ruta_mascara

RAIZ = rutas.RAIZ
T1 = os.path.join(rutas.DATOS_T1, "ds001226", "{s}", "ses-preop", "anat", "{s}_ses-preop_T1w.nii.gz")
PACIENTES = ("sub-PAT01", "sub-PAT26")
PANEL_CORTE = "sub-PAT26"
FIGURA_CORTE = os.path.join(os.path.dirname(AQUI), "fig", "mascara_t1.png")


def panel_corte(t1, at1, mascara, espejo, eje_z, k):
    """Corte axial limpio: máscara con su matriz afín (roja) y leída por índice (espejo)."""
    import logging
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    corte = np.take(t1, k, axis=eje_z)
    mk, me = np.take(mascara, k, axis=eje_z), np.take(espejo, k, axis=eje_z)
    cabeza = np.argwhere(corte > np.percentile(corte[corte > 0], 20))
    (i0, j0), (i1, j1) = cabeza.min(0) - 4, cabeza.max(0) + 5
    i0, j0 = max(i0, 0), max(j0, 0)
    corte, mk, me = corte[i0:i1, j0:j1], mk[i0:i1, j0:j1], me[i0:i1, j0:j1]
    eje_x = int(np.where(np.abs(at1[:3, :3]).argmax(0) == 0)[0][0])
    derecha_a_la_derecha = at1[0, eje_x] > 0
    with plt.rc_context({"font.family": ["Arial", "Liberation Sans"], "font.size": 16}):
        dibujar_panel(corte, mk, me, derecha_a_la_derecha)


def dibujar_panel(corte, mk, me, derecha_a_la_derecha):
    """Se dibuja a 50 mm de ancho, para que los rótulos midan 16 pt a tamaño real."""
    alto = 1.97 * corte.shape[1] / corte.shape[0]
    fig = plt.figure(figsize=(1.97, alto))
    ax = fig.add_axes([0, 0, 1, 1])
    lo, hi = np.percentile(corte[corte > 0], [1, 99.5])
    ax.imshow(corte.T, cmap="gray", origin="lower", vmin=lo, vmax=hi, aspect="auto")
    ax.contour(mk.T.astype(float), levels=[0.5], colors="#D7301F", linewidths=1.8)
    ax.contour(me.T.astype(float), levels=[0.5], colors="#F2A900", linewidths=1.6, linestyles="dashed")
    ax.set_xticks([]); ax.set_yticks([])
    for borde in ax.spines.values():
        borde.set_visible(False)
    izquierda, derecha = ("izq.", "der.") if derecha_a_la_derecha else ("der.", "izq.")
    ax.text(0.04, 0.97, izquierda, transform=ax.transAxes, color="white", va="top", ha="left", fontsize=16)
    ax.text(0.96, 0.97, derecha, transform=ax.transAxes, color="white", va="top", ha="right", fontsize=16)
    os.makedirs(os.path.dirname(FIGURA_CORTE), exist_ok=True)
    fig.savefig(FIGURA_CORTE, dpi=300)
    plt.close(fig)


def a_voxel(afin, xyz):
    return np.linalg.solve(afin, np.c_[xyz, np.ones(len(xyz))].T).T[:, :3]


def intensidad(vol, vox):
    v = np.round(vox).astype(int)
    dentro = np.all((v >= 0) & (v < np.array(vol.shape)), axis=1)
    salida = np.zeros(len(v))
    salida[dentro] = vol[tuple(v[dentro].T)]
    return salida, dentro


def main():
    res = {}
    fig, ejes = plt.subplots(2, 2, figsize=(10, 10))
    for fila, s in enumerate(PACIENTES):
        t1, at1, _, _ = leer_nifti(T1.format(s=s))
        t1 = np.asarray(t1, dtype=float)
        if t1.ndim == 4:
            t1 = t1[..., 0]
        m, am, _, _ = leer_nifti(ruta_mascara(s))
        umbral = threshold_otsu(t1[t1 > 0])
        nombres, xyz = leer_centros(s)
        val, dentro = intensidad(t1, a_voxel(at1, xyz))
        espejo = m >= 0.5
        idx = np.argwhere(espejo)
        mundo = (am @ np.c_[idx, np.ones(len(idx))].T).T[:, :3]
        v = np.round(a_voxel(at1, mundo)).astype(int)
        v = v[np.all((v >= 0) & (v < np.array(t1.shape[:3])), axis=1)]
        mascara = np.zeros(t1.shape[:3], bool)
        mascara[tuple(v.T)] = True
        cerebro = t1 > umbral
        res[s] = {
            "forma_t1": list(t1.shape), "forma_mascara": list(m.shape),
            "misma_forma": list(t1.shape[:3]) == list(m.shape),
            "signo_eje_x": {"t1": float(np.sign(at1[0, 0])), "mascara": float(np.sign(am[0, 0]))},
            "max_dif_afin_mm": float(np.abs(at1 - am).max()),
            "intensidad_t1_mediana": {"mascara_por_coordenadas_de_mundo": float(np.median(t1[mascara])),
                                      "mascara_voxel_a_voxel_en_espejo": float(np.median(t1[espejo])),
                                      "cabeza_sobre_otsu": float(np.median(t1[cerebro]))},
            "centroides_dentro_del_volumen": int(dentro.sum()),
        }
        cen_mundo = mundo.mean(0)
        vox_t1 = a_voxel(at1, cen_mundo[None, :])[0]
        cvox = a_voxel(at1, xyz)
        ejes_t1 = np.abs(at1[:3, :3]).argmax(0)
        eje_z = int(np.where(ejes_t1 == 2)[0][0])
        eje_y = int(np.where(ejes_t1 == 1)[0][0])
        if s == PANEL_CORTE:
            panel_corte(t1, at1, mascara, espejo, eje_z, int(round(vox_t1[eje_z])))
            plt.figure(fig.number)
        for col, (eje, titulo) in enumerate(((eje_z, "axial"), (eje_y, "coronal"))):
            k = int(round(vox_t1[eje]))
            corte = np.take(t1, k, axis=eje)
            mk = np.take(mascara, k, axis=eje)
            me = np.take(espejo, k, axis=eje) if list(m.shape) == list(t1.shape[:3]) else np.zeros_like(mk)
            otros_ejes = [x for x in range(3) if x != eje]
            ax = ejes[fila, col]
            ax.imshow(corte.T, cmap="gray", origin="lower")
            if mk.any():
                ax.contour(mk.T.astype(float), levels=[0.5], colors="red", linewidths=1.2)
            if me.any():
                ax.contour(me.T.astype(float), levels=[0.5], colors="orange", linewidths=0.8, linestyles="dashed")
            cerca = np.abs(cvox[:, eje] - k) <= 6
            ax.scatter(cvox[cerca, otros_ejes[0]], cvox[cerca, otros_ejes[1]], s=18, c="cyan", edgecolors="k", linewidths=0.4)
            for n in np.where(cerca)[0]:
                ax.text(cvox[n, otros_ejes[0]] + 1.5, cvox[n, otros_ejes[1]] + 1.5, nombres[n].replace("lh_", "L ").replace("rh_", "R "),
                        fontsize=5.5, color="yellow")
            ax.set_title(f"{s} {titulo}; rojo: máscara por coordenadas de mundo;\n"
                         f"naranja: leída vóxel a vóxel (espejo); cian: centroides a 6 mm o menos", fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(os.path.join(AQUI, "alineacion_t1.png"), dpi=130)
    with open(os.path.join(AQUI, "alineacion_t1.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1)
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
