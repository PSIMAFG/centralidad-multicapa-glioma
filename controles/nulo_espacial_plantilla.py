"""Nulo espacial preliminar con centroides de plantilla (fsaverage5, atlas Desikan-Killiany).

Los datos locales no traen coordenadas por sujeto (vienen en SC.zip, que no está en la copia
local), así que se usan los centroides del atlas en fsaverage5 del ENIGMA Toolbox (GitHub,
MICA-MNI/ENIGMA). Para cada sujeto y capa se generan árboles aleatorios sesgados hacia aristas
cortas: árbol de expansión máxima sobre pesos E_ij · exp(-d_ij / lambda), con E_ij exponencial
y d_ij la distancia euclídea entre centroides. Lambda se calibra por capa para que la longitud
media de las aristas del árbol nulo iguale la del árbol real. Luego se cuentan las aristas
compartidas entre árboles nulos independientes de las dos capas y se comparan con las reales.

Es una cota espacial gruesa: usa centroides de plantilla, no los de cada sujeto, y no conserva
la secuencia de grados. La versión definitiva debería usar Centres.txt y Tract_lengths.txt de
cada sujeto (derivatives/TVB/*/ses-preop/SC.zip en OpenNeuro ds001226).
"""

import base64
import json
import os
import sys
import xml.etree.ElementTree as ET
import zlib

import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.stats import wilcoxon

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import rutas
sys.path.insert(0, rutas.MODULOS)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cargar as C
import multiplex as M
from validacion_P import categorias, regiones

RAIZ = rutas.RAIZ
ENIGMA = rutas.ENIGMA
SALIDA = os.path.dirname(os.path.abspath(__file__))
APARC = ["unknown", "bankssts", "caudalanteriorcingulate", "caudalmiddlefrontal", "corpuscallosum", "cuneus",
         "entorhinal", "fusiform", "inferiorparietal", "inferiortemporal", "isthmuscingulate", "lateraloccipital",
         "lateralorbitofrontal", "lingual", "medialorbitofrontal", "middletemporal", "parahippocampal",
         "paracentral", "parsopercularis", "parsorbitalis", "parstriangularis", "pericalcarine", "postcentral",
         "posteriorcingulate", "precentral", "precuneus", "rostralanteriorcingulate", "rostralmiddlefrontal",
         "superiorfrontal", "superiorparietal", "superiortemporal", "supramarginal", "frontalpole", "temporalpole",
         "transversetemporal", "insula"]
K = 200
SEMILLA = 20261120


def leer_gifti_puntos(ruta):
    raiz = ET.parse(ruta).getroot()
    for da in raiz.iter("DataArray"):
        if da.get("Intent") == "NIFTI_INTENT_POINTSET":
            datos = base64.b64decode(da.find("Data").text.strip())
            if da.get("Encoding") == "GZipBase64Binary":
                datos = zlib.decompress(datos)
            return np.frombuffer(datos, dtype="<f4").reshape(int(da.get("Dim0")), int(da.get("Dim1")))
    raise ValueError("sin coordenadas")


def centroides():
    lh = leer_gifti_puntos(os.path.join(ENIGMA, "surfaces", "fsa5_lh.gii"))
    rh = leer_gifti_puntos(os.path.join(ENIGMA, "surfaces", "fsa5_rh.gii"))
    etiquetas = np.loadtxt(os.path.join(ENIGMA, "parcellations", "aparc_fsa5.csv"), dtype=int)
    el, er = etiquetas[:len(lh)], etiquetas[len(lh):]
    cent = {}
    for idx in range(1, 36):
        nombre = APARC[idx]
        if nombre == "corpuscallosum":
            continue
        if np.any(el == idx):
            cent["ctx-lh-" + nombre] = lh[el == idx].mean(0)
        if np.any(er == idx + 35):
            cent["ctx-rh-" + nombre] = rh[er == idx + 35].mean(0)
    return cent


def arbol_geometrico(d, lam, rng):
    n = d.shape[0]
    e = rng.exponential(1.0, (n, n))
    w = np.triu(e * np.exp(-d / lam), 1)
    w = w + w.T
    t = minimum_spanning_tree(-np.triu(w, 1)).toarray()
    a = ((t + t.T) != 0).astype(float)
    return a


def longitud_media(a, d):
    i, j = np.nonzero(np.triu(a, 1))
    return float(d[i, j].mean())


def calibrar(objetivo, d, rng, n_arboles=12):
    lo, hi = 1.0, 400.0
    for _ in range(18):
        mid = np.sqrt(lo * hi)
        lm = np.mean([longitud_media(arbol_geometrico(d, mid, rng), d) for _ in range(n_arboles)])
        if lm > objetivo:
            hi = mid
        else:
            lo = mid
    return np.sqrt(lo * hi)


def main():
    rng = np.random.default_rng(SEMILLA)
    nombres = regiones()
    cent = centroides()
    xyz = np.array([cent[x] for x in nombres])
    chequeo = {
        "x_izquierda_media": float(xyz[:34, 0].mean()), "x_derecha_media": float(xyz[34:, 0].mean()),
        "y_polo_frontal": float(cent["ctx-lh-frontalpole"][1]), "y_occipital_lateral": float(cent["ctx-lh-lateraloccipital"][1]),
        "z_polo_temporal": float(cent["ctx-lh-temporalpole"][2]), "z_frontal_superior": float(cent["ctx-lh-superiorfrontal"][2]),
    }
    d = np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=2)
    cat = categorias(nombres)
    iu = np.triu_indices(68, 1)
    reales, nulos, lam_a, lam_b, intra_real_b, intra_nulo_b, long_real = [], [], [], [], [], [], []
    hemi = np.array([0] * 34 + [1] * 34)
    for s in C.sujetos(RAIZ):
        par = C.cargar_sujeto(RAIZ, s)
        if par is None:
            continue
        a, b = M.arbol_maximo(par[0]), M.arbol_maximo(par[1])
        reales.append(int(((np.triu(a) > 0) & (np.triu(b) > 0)).sum()))
        la, lb = calibrar(longitud_media(a, d), d, rng), calibrar(longitud_media(b, d), d, rng)
        lam_a.append(la); lam_b.append(lb)
        long_real.append((longitud_media(a, d), longitud_media(b, d)))
        comp = []
        for _ in range(K):
            na, nb = arbol_geometrico(d, la, rng), arbol_geometrico(d, lb, rng)
            comp.append(int(((np.triu(na) > 0) & (np.triu(nb) > 0)).sum()))
        nulos.append(float(np.mean(comp)))
        i, j = np.nonzero(np.triu(b, 1))
        intra_real_b.append(float(np.mean(hemi[i] == hemi[j])))
        nb = arbol_geometrico(d, lb, rng)
        i, j = np.nonzero(np.triu(nb, 1))
        intra_nulo_b.append(float(np.mean(hemi[i] == hemi[j])))
    reales, nulos = np.array(reales), np.array(nulos)
    res = {
        "chequeo_centroides": chequeo,
        "aristas_compartidas_reales_media": float(reales.mean()),
        "aristas_compartidas_nulo_espacial_media": float(nulos.mean()),
        "razon_real_sobre_nulo_espacial_mediana": float(np.median(reales / nulos)),
        "sujetos_real_mayor": int(np.sum(reales > nulos)),
        "wilcoxon_p": float(wilcoxon(reales - nulos).pvalue),
        "longitud_media_aristas_mm": {"dmri": float(np.mean([x[0] for x in long_real])),
                                      "fmri": float(np.mean([x[1] for x in long_real]))},
        "fraccion_intrahemisferica_fmri": {"real": float(np.mean(intra_real_b)), "nulo": float(np.mean(intra_nulo_b))},
        "K": K,
    }
    with open(os.path.join(SALIDA, "nulo_espacial_plantilla.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1)
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
