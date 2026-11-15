"""¿Sigue el reordenamiento regional la jerarquía unimodal–transmodal?

Análisis exploratorio sugerido por la literatura de acoplamiento estructura–función: la FC se predice
bien desde la SC en las cortezas sensoriales y mal en las de asociación (Vázquez-Rodríguez y cols.,
2019, PNAS). Pasos:

1. Jerarquía de referencia: gradiente principal de la FC grupal del Human Connectome Project en la
   misma parcelación Desikan-Killiany (matrices del ENIGMA Toolbox), con el mapa de difusión de
   Margulies y cols. (2016): 10 % de conexiones más fuertes por fila, similitud coseno y alfa = 0,5.
   De los dos primeros componentes se usa el que más separa dos conjuntos fijados de antemano
   (regiones del modo por defecto y sensoriales primarias), orientado con el modo por defecto en el
   extremo positivo. Se usa una referencia externa porque la FC de esta cohorte no reproduce el
   gradiente canónico (se informa su separación) y porque los mapas del multicapa salen de ella.
2. Acoplamiento regional SC–FC: R² ajustado por región de una regresión del perfil de FC sobre tres
   predictores de comunicación calculados desde la SC (distancia euclídea, longitud de caminos
   ponderada y comunicabilidad), como en Vázquez-Rodríguez y cols. Se calcula en el HCP (referencia)
   y en cada sujeto de la cohorte (promedio), tanto con sus matrices tal como vienen como transformadas
   igual que las del HCP (FC sin negativos y en z de Fisher; SC en escala logarítmica).
3. Mapas del multicapa en los árboles binarizados (configuración de Breedt), promediados en los 36
   sujetos: reordenamiento |rango de la versatilidad - rango de la centralidad de la suma|,
   reordenamiento con signo y fracción estructural x_dMRI / (x_dMRI + x_fMRI), con omega = 1 y con
   omega adimensional 0,1 y 0,28.
4. Contrastes entre mapas (Spearman, bilaterales) con cinco referencias nulas (Markello y Mišić, 2021):
   - permutación simple, que ignora la autocorrelación espacial;
   - rotaciones de la esfera fsaverage5 de FreeSurfer (Alexander-Bloch y cols., 2018); la rotación del
     hemisferio derecho es la reflejada de la del izquierdo. Las parcelas se reasignan de tres formas:
     asignación húngara de los centroides (la primaria), moda de los vértices rotados (Baum y cols.,
     2020) y promedio de los vértices rotados (Cornblath y cols., 2020). En las dos últimas, una
     parcela cubierta entera por la pared medial queda sin valor en esa rotación;
   - sustitutos que conservan el variograma (Burt y cols., 2020, clase Base de BrainSMASH), con
     distancias euclídeas entre los centroides de la superficie pial.
   Para cada nulo se informa también el |rho| que habría hecho falta para p < 0,05 (percentil 95 de
   |rho| nulo). La familia principal (omega = 1) usa el p de las rotaciones con asignación húngara,
   corregido por Holm.
5. Confiabilidad de cada mapa por mitades al azar de los sujetos (Spearman-Brown).
"""

import base64
import gzip
import json
import os
import sys
import xml.etree.ElementTree as ET
import zlib

import numpy as np
import scipy.linalg as sl
from scipy.optimize import linear_sum_assignment
from scipy.sparse.csgraph import shortest_path
from scipy.spatial import cKDTree
from scipy.stats import rankdata, spearmanr

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
sys.path.insert(0, os.path.join(AQUI, "..", "controles"))
sys.path.insert(0, os.path.dirname(AQUI))
import rutas
sys.path.insert(0, rutas.MODULOS)
import cargar as C
import multiplex as M
from controles import radio
from validacion_P import regiones

RAIZ = rutas.RAIZ
ENIGMA = rutas.ENIGMA
FSAVERAGE5 = rutas.FSAVERAGE5
SEMILLA = 20261114
ROTACIONES = 10000
SUSTITUTOS = 2000
MITADES = 1000
ACOPLAMIENTOS = {"omega_1": None, "omega_adim_0.1": 0.1, "omega_adim_0.28": 0.28}
NULOS = ("permutacion", "rotacion", "baum", "cornblath", "variograma")
ESPACIALES = ("rotacion", "baum", "cornblath", "variograma")
MODO_POR_DEFECTO = ("posteriorcingulate", "isthmuscingulate", "precuneus", "inferiorparietal",
                    "medialorbitofrontal", "rostralanteriorcingulate", "middletemporal")
SENSORIALES = ("pericalcarine", "cuneus", "lingual", "lateraloccipital", "precentral", "postcentral",
               "paracentral", "transversetemporal")
APARC = ["unknown", "bankssts", "caudalanteriorcingulate", "caudalmiddlefrontal", "corpuscallosum", "cuneus",
         "entorhinal", "fusiform", "inferiorparietal", "inferiortemporal", "isthmuscingulate", "lateraloccipital",
         "lateralorbitofrontal", "lingual", "medialorbitofrontal", "middletemporal", "parahippocampal",
         "paracentral", "parsopercularis", "parsorbitalis", "parstriangularis", "pericalcarine", "postcentral",
         "posteriorcingulate", "precentral", "precuneus", "rostralanteriorcingulate", "rostralmiddlefrontal",
         "superiorfrontal", "superiorparietal", "superiortemporal", "supramarginal", "frontalpole", "temporalpole",
         "transversetemporal", "insula"]


def leer_gifti(ruta):
    """Vértices y caras de una superficie GIFTI (.gii o .gii.gz)."""
    with (gzip.open(ruta) if ruta.endswith(".gz") else open(ruta, "rb")) as fh:
        raiz = ET.fromstring(fh.read())
    salida = {}
    for da in raiz.iter("DataArray"):
        datos = base64.b64decode(da.find("Data").text.strip())
        if da.get("Encoding") == "GZipBase64Binary":
            datos = zlib.decompress(datos)
        tipo = "<f4" if da.get("DataType") == "NIFTI_TYPE_FLOAT32" else "<i4"
        forma = (int(da.get("Dim0")), int(da.get("Dim1")))
        salida[da.get("Intent")] = np.frombuffer(datos, dtype=tipo).reshape(forma)
    return salida["NIFTI_INTENT_POINTSET"].astype(float), salida["NIFTI_INTENT_TRIANGLE"].astype(int)


def geometria_plantilla(nombres):
    """Centroides pial y de esfera de las 68 regiones en fsaverage5, y la región de cada vértice.

    Devuelve los centroides de superficie, los de esfera (unitarios) y, por hemisferio (0 izquierdo,
    1 derecho), los vértices de la esfera y el índice de región de cada uno (-1 en la pared medial).
    """
    etiquetas = np.loadtxt(os.path.join(ENIGMA, "parcellations", "aparc_fsa5.csv"), dtype=int)
    indice = {x: k for k, x in enumerate(nombres)}
    superficie, centro_esfera, vertices = {}, {}, {}
    for h, (lado, nombre_h, desfase) in enumerate((("left", "lh", 0), ("right", "rh", 35))):
        v, caras = leer_gifti(os.path.join(ENIGMA, "surfaces", f"fsa5_{nombre_h}.gii"))
        s, caras_esfera = leer_gifti(os.path.join(FSAVERAGE5, f"sphere_{lado}.gii.gz"))
        if not np.array_equal(caras, caras_esfera):
            raise ValueError("la esfera y la superficie de fsaverage5 no comparten la malla")
        s = s / np.linalg.norm(s, axis=1, keepdims=True)
        e = etiquetas[:len(v)] if h == 0 else etiquetas[len(v):]
        region = np.full(len(v), -1)
        for idx in range(1, 36):
            nombre = f"ctx-{nombre_h}-{APARC[idx]}"
            if nombre not in indice:
                continue
            m = e == idx + desfase
            region[m] = indice[nombre]
            superficie[nombre] = v[m].mean(0)
            c = s[m].mean(0)
            centro_esfera[nombre] = c / np.linalg.norm(c)
        vertices[h] = (s, region)
    return (np.array([superficie[x] for x in nombres]), np.array([centro_esfera[x] for x in nombres]), vertices)


def matrices_hcp(nombres):
    """FC y SC grupales del HCP en Desikan-Killiany (ENIGMA Toolbox), en el orden de nombres."""
    carpeta = os.path.join(ENIGMA, "matrices", "hcp_connectivity")
    salida = []
    for tipo in ("func", "struc"):
        etiquetas = open(os.path.join(carpeta, f"{tipo}Labels_ctx.csv"), encoding="utf-8").read().strip().split(",")
        etiquetas = [x.replace("L_", "ctx-lh-", 1) if x.startswith("L_") else x.replace("R_", "ctx-rh-", 1)
                     for x in etiquetas]
        m = np.loadtxt(os.path.join(carpeta, f"{tipo}Matrix_ctx.csv"), delimiter=",")
        idx = [etiquetas.index(x) for x in nombres]
        salida.append(m[np.ix_(idx, idx)])
    return salida


def medidas(a, b, omega):
    """Reordenamiento, reordenamiento con signo y fracción estructural de cada región."""
    n = a.shape[0]
    s = np.block([[a, omega * np.eye(n)], [omega * np.eye(n), b]])
    x = np.abs(sl.eigh(s, subset_by_index=[2 * n - 1, 2 * n - 1], driver="evr")[1][:, 0])
    x = x / x.sum()
    v = x[:n] + x[n:]
    c = np.abs(np.linalg.eigh(a + b)[1][:, -1])
    signo = rankdata(v) - rankdata(c)
    return np.abs(signo), signo, x[:n] / (x[:n] + x[n:])


def gradientes(fc, umbral=90, alfa=0.5, k=2):
    """Mapa de difusión de la FC (Margulies y cols., 2016): componentes 1..k y sus autovalores."""
    m = np.array(fc, dtype=float)
    np.fill_diagonal(m, 0.0)
    corte = np.percentile(m, umbral, axis=1, keepdims=True)
    m = np.where((m >= corte) & (m > 0), m, 0.0)
    norma = np.linalg.norm(m, axis=1, keepdims=True)
    norma[norma == 0] = 1.0
    u = m / norma
    afin = np.clip(u @ u.T, 0.0, None)
    d = afin.sum(1)
    afin = afin / np.outer(d ** alfa, d ** alfa)
    d2 = afin.sum(1)
    sim = afin / np.sqrt(np.outer(d2, d2))
    val, vec = np.linalg.eigh(sim)
    orden = np.argsort(val)[::-1]
    val, vec = val[orden], vec[:, orden]
    psi = vec / vec[:, [0]]
    return psi[:, 1:k + 1] * (val[1:k + 1] / (1.0 - val[1:k + 1])), val[1:k + 1]


def jerarquia_de(fc, base):
    """Componente que más separa el modo por defecto de las regiones sensoriales, con su orientación."""
    comps, vals = gradientes(fc)
    dmn, sen = np.isin(base, MODO_POR_DEFECTO), np.isin(base, SENSORIALES)
    sep = [float((comps[dmn, k].mean() - comps[sen, k].mean()) / comps[:, k].std()) for k in range(comps.shape[1])]
    k = int(np.argmax(np.abs(sep)))
    return comps[:, k] * np.sign(sep[k]), k + 1, sep, vals


def acoplamiento_regional(sc, fc, dist, como_hcp=False):
    """R² ajustado por región de FC ~ distancia + longitud de caminos + comunicabilidad.

    Con como_hcp=True, las matrices de un sujeto se transforman como las del HCP en el ENIGMA Toolbox:
    FC con las correlaciones negativas en cero y transformada a z de Fisher, y SC en escala logarítmica.
    """
    n = sc.shape[0]
    w = np.array(sc, dtype=float)
    np.fill_diagonal(w, 0.0)
    if como_hcp:
        w[w > 0] = np.log1p(w[w > 0] / w[w > 0].min())
        fc = np.arctanh(np.clip(fc, 0.0, 0.999999))
    w = w / w.max()
    largo = np.zeros_like(w)
    largo[w > 0] = 1.0 / w[w > 0]
    camino = shortest_path(largo, method="D", directed=False)
    fuerza = w.sum(1)
    fuerza[fuerza == 0] = 1.0
    comun = sl.expm(w / np.sqrt(np.outer(fuerza, fuerza)))
    r2 = np.full(n, np.nan)
    for i in range(n):
        j = np.array([k for k in range(n) if k != i])
        x = np.column_stack([dist[i, j], camino[i, j], comun[i, j]])
        if not np.all(np.isfinite(x)) or np.any(x.std(0) == 0):
            continue
        x = np.column_stack([np.ones(len(j)), (x - x.mean(0)) / x.std(0)])
        y = fc[i, j]
        coef, *_ = np.linalg.lstsq(x, y, rcond=None)
        res = y - x @ coef
        r2_i = 1.0 - res @ res / ((y - y.mean()) @ (y - y.mean()))
        r2[i] = 1.0 - (1.0 - r2_i) * (len(j) - 1) / (len(j) - x.shape[1])
    return r2


def rotacion_aleatoria(rng):
    """Rotación uniforme en SO(3)."""
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q = q @ np.diag(np.sign(np.diag(r)))
    if np.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


def rotar_esfera(centros, hemi, vertices, rotaciones):
    """Reasignación de parcelas para cada rotación del hemisferio izquierdo (el derecho usa la reflejada).

    Devuelve la permutación húngara de los centroides (la parcela que llega a cada posición) y los
    conteos vértice a vértice: conteos[t, p, q] es el número de vértices de la región p cuyo vértice
    rotado más cercano pertenece a la región q. Los que reciben pared medial no cuentan.
    """
    espejo = np.diag([-1.0, 1.0, 1.0])
    n = len(hemi)
    indices = {h: np.flatnonzero(hemi == h) for h in (0, 1)}
    arboles = {h: cKDTree(vertices[h][0]) for h in (0, 1)}
    hungaro = np.empty((len(rotaciones), n), dtype=int)
    conteos = np.zeros((len(rotaciones), n, n), dtype=np.int16)
    for t, rot in enumerate(rotaciones):
        for h, r in ((0, rot), (1, espejo @ rot @ espejo)):
            puntos = centros[indices[h]]
            filas, cols = linear_sum_assignment(-(puntos @ r.T) @ puntos.T)
            hungaro[t, indices[h][cols]] = indices[h][filas]
            esf, region = vertices[h]
            origen = region[arboles[h].query(esf @ r, workers=-1)[1]]
            ok = (region >= 0) & (origen >= 0)
            conteos[t] += np.bincount(region[ok] * n + origen[ok], minlength=n * n).reshape(n, n).astype(np.int16)
    return hungaro, conteos


def nulos_rotados(mapa, hungaro, conteos):
    """Mapas rotados con asignación húngara, moda (Baum) y promedio (Cornblath) de los vértices."""
    suma = conteos.sum(2)
    baum = np.where(suma > 0, conteos.argmax(2), len(mapa))
    finito = np.isfinite(mapa)
    num = np.einsum("tpq,q->tp", conteos, np.where(finito, mapa, 0.0))
    den = np.einsum("tpq,q->tp", conteos, finito.astype(float))
    cornblath = np.full(num.shape, np.nan)
    np.divide(num, den, out=cornblath, where=den > 0)
    return {"rotacion": mapa[hungaro], "baum": np.append(mapa, np.nan)[baum], "cornblath": cornblath}


def spearman_filas(nulos, referencia):
    """Spearman de cada fila con la referencia, omitiendo los pares con algún valor ausente."""
    nulos = np.atleast_2d(np.asarray(nulos, dtype=float))
    salida = np.full(len(nulos), np.nan)
    completas = np.all(np.isfinite(nulos), 1) & bool(np.all(np.isfinite(referencia)))
    if completas.any():
        r = rankdata(nulos[completas], axis=1)
        r -= r.mean(1, keepdims=True)
        ref = rankdata(referencia) - (len(referencia) + 1) / 2.0
        salida[completas] = (r @ ref) / np.sqrt((r ** 2).sum(1) * (ref ** 2).sum())
    for i in np.flatnonzero(~completas):
        k = np.isfinite(nulos[i]) & np.isfinite(referencia)
        if k.sum() > 3:
            salida[i] = spearmanr(nulos[i, k], referencia[k])[0]
    return salida


def variograma(x, dist, iu, h, ancho):
    d = dist[iu]
    nube = 0.5 * (x[iu[0]] - x[iu[1]]) ** 2
    pesos = np.exp(-0.5 * (2.68 * (d[None, :] - h[:, None]) / ancho) ** 2)
    return (pesos * nube[None, :]).sum(1) / pesos.sum(1)


def sustitutos_variograma(x, dist, total, rng, deltas=np.linspace(0.1, 0.9, 9), pv=25, nh=25):
    """Sustitutos que conservan la autocorrelación espacial (Burt y cols., 2020, clase Base)."""
    n = len(x)
    iu = np.triu_indices(n, 1)
    cerca = dist[iu] < np.percentile(dist[iu], pv)
    iu = (iu[0][cerca], iu[1][cerca])
    h = np.linspace(dist[iu].min(), dist[iu].max(), nh)
    ancho = 3.0 * (h[1] - h[0])
    objetivo = variograma(x, dist, iu, h, ancho)
    vecinos = np.argsort(dist, axis=1)
    salida = np.empty((total, n))
    orden_x = np.sort(x)
    for t in range(total):
        xp = rng.permutation(x)
        mejor = None
        for delta in deltas:
            k = max(2, int(delta * n))
            idx = vecinos[:, :k]
            dk = np.take_along_axis(dist, idx, 1)
            pes = np.exp(-dk / dk[:, -1:])
            xs = (pes * xp[idx]).sum(1) / pes.sum(1)
            vs = variograma(xs, dist, iu, h, ancho)
            a = np.column_stack([vs, np.ones_like(vs)])
            coef, *_ = np.linalg.lstsq(a, objetivo, rcond=None)
            sse = float(np.sum((objetivo - a @ coef) ** 2))
            if mejor is None or sse < mejor[0]:
                mejor = (sse, xs, coef)
        _, xs, (beta, alfa) = mejor
        sur = np.sqrt(abs(beta)) * xs + np.sqrt(abs(alfa)) * rng.normal(size=n)
        salida[t] = orden_x[np.argsort(np.argsort(sur))]
    return salida


def contraste(mapa, referencia, hungaro, conteos, sustitutos_mapa, rng):
    """Spearman entre dos mapas con p bilateral y |rho| crítico para cada referencia nula."""
    ok = np.isfinite(mapa) & np.isfinite(referencia)
    obs = float(spearmanr(mapa[ok], referencia[ok])[0])
    conjuntos = {"permutacion": spearman_filas([rng.permutation(mapa[ok]) for _ in range(len(hungaro))],
                                               referencia[ok]),
                 "variograma": spearman_filas(sustitutos_mapa[:, ok], referencia[ok])}
    for nombre, nulos in nulos_rotados(mapa, hungaro, conteos).items():
        conjuntos[nombre] = spearman_filas(nulos, referencia)
    salida = {"rho": obs}
    for nombre in NULOS:
        r = np.abs(conjuntos[nombre][np.isfinite(conjuntos[nombre])])
        salida[f"p_{nombre}"] = float((np.sum(r >= abs(obs)) + 1) / (len(r) + 1))
    salida["rho_critico"] = {nombre: float(np.percentile(np.abs(conjuntos[nombre][np.isfinite(conjuntos[nombre])]), 95))
                             for nombre in NULOS}
    salida["p_maximo_espacial"] = max(salida[f"p_{nombre}"] for nombre in ESPACIALES)
    return salida


def holm(ps):
    orden = np.argsort(ps)
    ajustados = np.empty(len(ps))
    acumulado = 0.0
    for rango, i in enumerate(orden):
        acumulado = max(acumulado, min(1.0, (len(ps) - rango) * ps[i]))
        ajustados[i] = acumulado
    return ajustados.tolist()


def confiabilidad(matriz, rng):
    """Correlación media entre los mapas promedio de dos mitades al azar, con Spearman-Brown."""
    n = matriz.shape[0]
    rs = []
    for _ in range(MITADES):
        p = rng.permutation(n)
        a, b = np.nanmean(matriz[p[: n // 2]], 0), np.nanmean(matriz[p[n // 2:]], 0)
        rs.append(spearmanr(a, b)[0])
    r = float(np.mean(rs))
    return {"mitades": r, "spearman_brown": 2 * r / (1 + r)}


def rellenar(x):
    """Reemplaza valores ausentes por la mediana para generar sustitutos (no afecta el contraste)."""
    y = np.array(x, dtype=float)
    y[~np.isfinite(y)] = np.nanmedian(y)
    return y


def main():
    rng = np.random.default_rng(SEMILLA)
    nombres = regiones()
    base = np.array([x[7:] for x in nombres])
    hemi = np.array([0 if x.startswith("ctx-lh-") else 1 for x in nombres])
    sujetos = C.sujetos(RAIZ)
    grupos = C.grupos(RAIZ)
    controles = [s for s in sujetos if grupos.get(s) == "control"]
    from lectura import leer_centros
    datos, centros = {}, {}
    for s in sujetos:
        sc, fc = C.cargar_sujeto(RAIZ, s)
        nom, xyz_s = leer_centros(s)
        assert nom == [x.replace("ctx-lh-", "lh_").replace("ctx-rh-", "rh_") for x in nombres]
        datos[s], centros[s] = (sc, fc), xyz_s

    xyz, esf, vertices = geometria_plantilla(nombres)
    dist = np.sqrt(((xyz[:, None, :] - xyz[None, :, :]) ** 2).sum(-1))
    fc_hcp, sc_hcp = matrices_hcp(nombres)
    jerarquia, comp, sep_hcp, vals_hcp = jerarquia_de(fc_hcp, base)
    fc_cohorte = np.tanh(np.mean([np.arctanh(np.clip(datos[s][1], 0, 0.999999)) for s in sujetos], 0))
    jer_cohorte, comp_c, sep_cohorte, _ = jerarquia_de(fc_cohorte, base)

    acop_hcp = acoplamiento_regional(sc_hcp, fc_hcp, dist)
    acop_suj = np.array([acoplamiento_regional(datos[s][0], datos[s][1],
                                               np.sqrt(((centros[s][:, None] - centros[s][None]) ** 2).sum(-1)))
                         for s in sujetos])
    acop_cohorte = np.nanmean(acop_suj, 0)
    acop_cohorte_hcp = np.nanmean([acoplamiento_regional(datos[s][0], datos[s][1],
                                                         np.sqrt(((centros[s][:, None] - centros[s][None]) ** 2).sum(-1)),
                                                         como_hcp=True) for s in sujetos], 0)

    mapas_suj = {}
    for nombre, wt in ACOPLAMIENTOS.items():
        r_abs, r_sig, frac = [], [], []
        for s in sujetos:
            a, b = M.arbol_maximo(datos[s][0]), M.arbol_maximo(datos[s][1])
            omega = 1.0 if wt is None else wt * (radio(a) + radio(b)) / 2.0
            m1, m2, m3 = medidas(a, b, omega)
            r_abs.append(m1)
            r_sig.append(m2)
            frac.append(m3)
        mapas_suj[nombre] = {"reordenamiento": np.array(r_abs), "reordenamiento_con_signo": np.array(r_sig),
                             "fraccion_estructural": np.array(frac)}

    hungaro, conteos = rotar_esfera(esf, hemi, vertices, [rotacion_aleatoria(rng) for _ in range(ROTACIONES)])
    sin_valor = (conteos.sum(2) == 0).sum(1)
    cache_sust = {}

    def sustitutos(clave, mapa):
        if clave not in cache_sust:
            cache_sust[clave] = sustitutos_variograma(rellenar(mapa), dist, SUSTITUTOS, rng)
        return cache_sust[clave]

    res = {
        "sujetos": len(sujetos), "regiones": len(nombres), "rotaciones": ROTACIONES,
        "sustitutos_variograma": SUSTITUTOS,
        "esfera": "fsaverage5 de FreeSurfer (misma malla que la superficie pial del ENIGMA Toolbox)",
        "regiones_sin_valor_por_rotacion": {"media": float(sin_valor.mean()), "maximo": int(sin_valor.max())},
        "jerarquia_hcp": {
            "componente_usado": comp, "autovalores": [float(v) for v in vals_hcp],
            "separacion_dmn_menos_sensoriales_en_de": sep_hcp,
            "extremo_transmodal": [nombres[i] for i in np.argsort(jerarquia)[::-1][:6]],
            "extremo_unimodal": [nombres[i] for i in np.argsort(jerarquia)[:6]],
        },
        "jerarquia_cohorte": {
            "componente_mas_separador": comp_c, "separacion_dmn_menos_sensoriales_en_de": sep_cohorte,
            "rho_con_jerarquia_hcp": float(spearmanr(jer_cohorte, jerarquia)[0]),
        },
        "acoplamiento_regional": {
            "hcp_r2_mediana": float(np.nanmedian(acop_hcp)),
            "cohorte_r2_mediana": float(np.nanmedian(acop_cohorte)),
            "cohorte_r2_p10_p90": [float(np.nanpercentile(acop_cohorte, 10)), float(np.nanpercentile(acop_cohorte, 90))],
            "rho_hcp_con_cohorte": float(spearmanr(acop_hcp, acop_cohorte, nan_policy="omit")[0]),
            "confiabilidad_cohorte": confiabilidad(acop_suj, rng),
            "cohorte_transformada_como_hcp": {
                "r2_mediana": float(np.nanmedian(acop_cohorte_hcp)),
                "rho_con_hcp": float(spearmanr(acop_hcp, acop_cohorte_hcp, nan_policy="omit")[0]),
                "rho_con_cohorte_sin_transformar": float(spearmanr(acop_cohorte, acop_cohorte_hcp, nan_policy="omit")[0]),
            },
            "hcp_mas_acopladas": [nombres[i] for i in np.argsort(acop_hcp)[::-1][:6]],
            "hcp_menos_acopladas": [nombres[i] for i in np.argsort(acop_hcp)[:6]],
        },
        "validacion": {
            "acoplamiento_hcp_vs_jerarquia": contraste(acop_hcp, jerarquia, hungaro, conteos,
                                                       sustitutos("acop_hcp", acop_hcp), rng),
            "acoplamiento_cohorte_vs_jerarquia": contraste(acop_cohorte, jerarquia, hungaro, conteos,
                                                           sustitutos("acop_cohorte", acop_cohorte), rng),
        },
        "contrastes": {},
    }
    primarios = []
    rng_secundario = np.random.default_rng(SEMILLA + 1)
    for nombre, bloque in mapas_suj.items():
        res["contrastes"][nombre] = {}
        for medida, matriz in bloque.items():
            mapa = matriz.mean(0)
            sust = sustitutos(f"{nombre}_{medida}", mapa)
            entrada = {"confiabilidad": confiabilidad(matriz, rng),
                       "regiones_mas_altas": [nombres[i] for i in np.argsort(mapa)[::-1][:5]],
                       "vs_jerarquia": contraste(mapa, jerarquia, hungaro, conteos, sust, rng),
                       "vs_acoplamiento_hcp": contraste(mapa, acop_hcp, hungaro, conteos, sust, rng),
                       "vs_acoplamiento_cohorte": contraste(mapa, acop_cohorte, hungaro, conteos, sust, rng),
                       "vs_acoplamiento_cohorte_como_hcp": contraste(mapa, acop_cohorte_hcp, hungaro, conteos, sust,
                                                                     rng_secundario),
                       "rho_vs_jerarquia_solo_controles": float(spearmanr(
                           matriz[[sujetos.index(s) for s in controles]].mean(0), jerarquia)[0])}
            res["contrastes"][nombre][medida] = entrada
            if nombre == "omega_1" and medida in ("reordenamiento", "fraccion_estructural"):
                for ref in ("jerarquia", "acoplamiento_hcp"):
                    primarios.append((f"{medida}_vs_{ref}", entrada[f"vs_{ref}"]))
    res["familia_principal_omega_1"] = {"contrastes": [c for c, _ in primarios],
                                       "rho": [e["rho"] for _, e in primarios],
                                       "p_rotacion": [e["p_rotacion"] for _, e in primarios],
                                       "p_holm": holm([e["p_rotacion"] for _, e in primarios]),
                                       "p_maximo_espacial": [e["p_maximo_espacial"] for _, e in primarios]}
    np.savez(os.path.join(AQUI, "jerarquia.npz"), jerarquia=jerarquia, acoplamiento_hcp=acop_hcp,
             acoplamiento_cohorte=acop_cohorte, acoplamiento_cohorte_como_hcp=acop_cohorte_hcp, reordenamiento=mapas_suj["omega_1"]["reordenamiento"].mean(0),
             fraccion_estructural=mapas_suj["omega_1"]["fraccion_estructural"].mean(0),
             reordenamiento_con_signo=mapas_suj["omega_1"]["reordenamiento_con_signo"].mean(0),
             xyz=xyz, esfera=esf, nombres=np.array(nombres))
    with open(os.path.join(AQUI, "jerarquia.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1)
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
