"""Nulo que conserva a la vez la secuencia de grados y la longitud de las aristas.

Para cada sujeto y capa se muestrean árboles con exactamente el mismo grado por región que el
árbol real y con una longitud media de aristas igual a la real, usando las distancias euclídeas
entre los centroides de ese sujeto (centres.txt de SC.zip, OpenNeuro ds001226).

Muestreo: cadena de Markov sobre árboles con secuencia de grados fija. Cada paso propone un
intercambio de extremos entre dos aristas, {a-b, c-d} -> {a-d, c-b}, que conserva los grados; se
descarta si crea una arista repetida o deja de ser un árbol. La aceptación es de Metropolis con
probabilidad proporcional a exp(-beta * longitud total), y beta se ajusta durante la fase de
adaptación (Robbins-Monro) hasta que la longitud media de las aristas iguala la del árbol real.
Con beta fijo, se estiman las probabilidades marginales de cada arista y las aristas compartidas
esperadas entre dos árboles nulos independientes: suma de p_dMRI(e) * p_fMRI(e).

La propuesta elige dos aristas y sus orientaciones al azar antes de descartar las inválidas, así que
es simétrica y no requiere corrección de Hastings. Beta se ajusta solo durante la adaptación; el
calentamiento y el muestreo usan beta fijo. En sub-PAT26, cuya capa dMRI tiene una región sin
conexiones, la cadena opera sobre las 67 regiones conectadas y la región aislada sigue aislada.

Comprobaciones: una segunda cadena que parte de un árbol al azar con los mismos grados (secuencia
de Prüfer permutada) debe dar el mismo valor esperado; la longitud media obtenida debe coincidir
con la real; todos los árboles muestreados conservan grados y la conexidad de las regiones
conectadas; y mezcla_cadena.py estima el tamaño efectivo de muestra.

Variante: en la capa fMRI se fijan las aristas homotópicas del árbol real (unas 20, largas) y se
calibra beta con las demás, porque un nulo de distancia casi no genera aristas homotópicas.

Además, con 100 pares de árboles nulos por sujeto se calcula el cruce (omega adimensional en que
rho con la suma llega a 0,95) para ver si los árboles reales siguen alcanzándolo antes que árboles
con los mismos grados y la misma geometría.
"""

import json
import math
import os
import random
import sys
import time
from multiprocessing import Pool

import numpy as np
from scipy.stats import wilcoxon

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import rutas
sys.path.insert(0, rutas.MODULOS)
sys.path.insert(0, AQUI)
sys.path.insert(0, os.path.join(AQUI, "..", "regional"))
import cargar as C
import multiplex as M
from controles import arbol_a_prufer, prufer_a_arbol
from explicacion_espectral import cruce, predictores
from lectura import leer_centros
from validacion_P import categorias, regiones

RAIZ = rutas.RAIZ
SEMILLA = 20261123
ADAPTACION = 60000
REFINAMIENTO = 40000
CALENTAMIENTO = 20000
CALENTAMIENTO_AZAR = 100000
MUESTREO = 800000
MUESTREO_AZAR = 400000
CADA = 10
PARES_CRUCE = 100


def aristas_de(a):
    i, j = np.nonzero(np.triu(a, 1))
    return [(int(x), int(y)) for x, y in zip(i, j)]


def a_matriz(aristas, n):
    m = np.zeros((n, n))
    for u, v in aristas:
        m[u, v] = m[v, u] = 1.0
    return m


def conexo(adj, inicio, total):
    visto = {inicio}
    pila = [inicio]
    while pila:
        u = pila.pop()
        for w in adj[u]:
            if w not in visto:
                visto.add(w)
                pila.append(w)
    return len(visto) == total


class Cadena:
    """Árbol con secuencia de grados fija que se modifica por intercambios de extremos."""

    def __init__(self, aristas, n, dist, fijas, rnd):
        self.n = n
        self.d = dist
        self.e = list(aristas)
        self.adj = [set() for _ in range(n)]
        for u, v in self.e:
            self.adj[u].add(v)
            self.adj[v].add(u)
        self.activos = [i for i in range(n) if self.adj[i]]
        self.fijas = set(fijas)
        self.libres = [k for k, x in enumerate(self.e) if x not in self.fijas]
        self.largo_libre = sum(dist[self.e[k][0]][self.e[k][1]] for k in self.libres)
        self.rnd = rnd
        self.propuestas = 0
        self.aceptadas = 0

    def media_libre(self):
        return self.largo_libre / len(self.libres)

    def paso(self, beta):
        rnd, e, adj, d = self.rnd, self.e, self.adj, self.d
        nl = len(self.libres)
        self.propuestas += 1
        i = self.libres[int(rnd.random() * nl)]
        j = self.libres[int(rnd.random() * nl)]
        if i == j:
            return
        a, b = e[i]
        c, x = e[j]
        if rnd.random() < 0.5:
            a, b = b, a
        if rnd.random() < 0.5:
            c, x = x, c
        if a == x or c == b or a == c or b == x:
            return
        if x in adj[a] or b in adj[c]:
            return
        dl = d[a][x] + d[c][b] - d[a][b] - d[c][x]
        if dl > 0 and rnd.random() >= math.exp(-beta * dl):
            return
        adj[a].discard(b); adj[b].discard(a); adj[c].discard(x); adj[x].discard(c)
        adj[a].add(x); adj[x].add(a); adj[c].add(b); adj[b].add(c)
        if not conexo(adj, self.activos[0], len(self.activos)):
            adj[a].discard(x); adj[x].discard(a); adj[c].discard(b); adj[b].discard(c)
            adj[a].add(b); adj[b].add(a); adj[c].add(x); adj[x].add(c)
            return
        e[i] = (a, x) if a < x else (x, a)
        e[j] = (c, b) if c < b else (b, c)
        self.largo_libre += dl
        self.aceptadas += 1


def adaptar(cadena, objetivo, pasos, beta0=0.05, ganancia0=0.01, escala=10000.0):
    theta = math.log(beta0)
    historia = []
    for t in range(pasos):
        cadena.paso(math.exp(theta))
        theta += ganancia0 / (1.0 + t / escala) * (cadena.media_libre() - objetivo) / objetivo
        if t >= pasos // 2:
            historia.append(theta)
    return math.exp(float(np.mean(historia)))


def muestrear(cadena, beta, pasos, cada, n, guardar=0):
    cuenta = np.zeros((n, n))
    medias, muestras = [], []
    paso_guardado = max(1, pasos // guardar) if guardar else None
    k = 0
    for t in range(pasos):
        cadena.paso(beta)
        if t % cada == 0:
            for u, v in cadena.e:
                cuenta[u, v] += 1.0
            medias.append(cadena.media_libre())
            k += 1
        if guardar and t % paso_guardado == 0 and len(muestras) < guardar:
            muestras.append(list(cadena.e))
    p = cuenta / k
    return p + p.T, float(np.mean(medias)), muestras


def comprobar_arbol(aristas, grados, n):
    m = a_matriz(aristas, n)
    if not np.array_equal(m.sum(1), grados):
        return False
    adj = [set(np.nonzero(m[i])[0].tolist()) for i in range(n)]
    activos = [i for i in range(n) if adj[i]]
    return len(aristas) == len(activos) - 1 and conexo(adj, activos[0], len(activos))


def capa(arbol, dist, fijas, semilla, guardar):
    n = arbol.shape[0]
    aristas = aristas_de(arbol)
    grados = arbol.sum(1)
    libres = [x for x in aristas if x not in set(fijas)]
    objetivo = float(np.mean([dist[u][v] for u, v in libres]))
    rnd = random.Random(semilla)
    c1 = Cadena(aristas, n, dist, fijas, rnd)
    beta = adaptar(c1, objetivo, ADAPTACION)
    beta = adaptar(c1, objetivo, REFINAMIENTO, beta0=beta, ganancia0=0.002, escala=1e9)
    for _ in range(CALENTAMIENTO):
        c1.paso(beta)
    c1.propuestas = c1.aceptadas = 0
    p1, media1, muestras = muestrear(c1, beta, MUESTREO, CADA, n, guardar)
    tasa = c1.aceptadas / c1.propuestas
    ok = all(comprobar_arbol(m, grados, n) for m in muestras) and comprobar_arbol(c1.e, grados, n)
    # Segunda cadena desde un árbol al azar con los mismos grados (sin aristas fijas).
    p2 = None
    if not fijas:
        sec, nodos = arbol_a_prufer(arbol)
        rng = np.random.default_rng(semilla)
        inicio = aristas_de(prufer_a_arbol(rng.permutation(sec), nodos, n))
        c2 = Cadena(inicio, n, dist, [], random.Random(semilla + 7))
        for _ in range(CALENTAMIENTO_AZAR):
            c2.paso(beta)
        p2, _, _ = muestrear(c2, beta, MUESTREO_AZAR, CADA, n)
    return {"p": p1, "p_azar": p2, "beta": beta, "objetivo": objetivo, "media": media1, "tasa": tasa,
            "arboles_validos": ok, "muestras": muestras}


def cuantil_ponderado(valores, pesos, q):
    orden = np.argsort(valores)
    v, w = valores[orden], pesos[orden]
    acumulado = np.cumsum(w) / w.sum()
    return float(np.interp(q, acumulado, v))


def comparar_longitudes(arbol, p, dist, cat):
    """Distribución de longitudes y categorías anatómicas del árbol real frente a la esperada en el nulo.

    p es la frecuencia con que cada arista aparece en los árboles nulos de la cadena, así que la
    distribución esperada de longitudes del nulo pondera cada arista candidata por p.
    """
    iu = np.triu_indices(arbol.shape[0], 1)
    d, pr, pn = dist[iu], arbol[iu] > 0, p[iu]
    reales = np.sort(d[pr])
    malla = np.sort(d)
    cdf_real = np.searchsorted(reales, malla, side="right") / len(reales)
    orden = np.argsort(d)
    cdf_nulo = np.interp(malla, d[orden], np.cumsum(pn[orden]) / pn.sum())
    salida = {"ks": float(np.max(np.abs(cdf_real - cdf_nulo)))}
    for q in (0.1, 0.25, 0.5, 0.75, 0.9):
        salida[f"q{int(q * 100)}_real"] = float(np.quantile(reales, q))
        salida[f"q{int(q * 100)}_nulo"] = cuantil_ponderado(d, pn, q)
    categorias_ = cat[iu]
    for c in ("mismo_lobulo", "distinto_lobulo", "homotopica", "otra_interhemisferica"):
        salida[f"{c}_real"] = int(np.sum(pr & (categorias_ == c)))
        salida[f"{c}_nulo"] = float(np.sum(pn[categorias_ == c]))
    return salida


def por_sujeto(args):
    indice, s, sc, fc = args
    t0 = time.time()
    _, xyz = leer_centros(s)
    dist = np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=2)
    dl = dist.tolist()
    cat = categorias(regiones())
    a, b = M.arbol_maximo(sc), M.arbol_maximo(fc)
    homo = [x for x in aristas_de(b) if cat[x[0], x[1]] == "homotopica"]
    base = SEMILLA * 100 + indice * 10
    ra = capa(a, dl, [], base + 1, PARES_CRUCE)
    rb = capa(b, dl, [], base + 2, PARES_CRUCE)
    rh = capa(b, dl, homo, base + 3, PARES_CRUCE)
    real = int(((np.triu(a, 1) > 0) & (np.triu(b, 1) > 0)).sum())
    esperadas = float(np.triu(ra["p"] * rb["p"], 1).sum())
    esperadas_azar = float(np.triu(ra["p_azar"] * rb["p_azar"], 1).sum())
    esperadas_h = float(np.triu(ra["p"] * rh["p"], 1).sum())
    cruces, cruces_h, p_nulos = [], [], []
    for k in range(PARES_CRUCE):
        na, nb = a_matriz(ra["muestras"][k], 68), a_matriz(rb["muestras"][k], 68)
        cruces.append(cruce(na, nb))
        p_nulos.append(predictores(na, nb)["P"])
        cruces_h.append(cruce(na, a_matriz(rh["muestras"][k], 68)))
    return {
        "sujeto": s, "reales": real,
        "esperadas": esperadas, "esperadas_cadena_desde_azar": esperadas_azar,
        "esperadas_homotopicas_fijas": esperadas_h,
        "esperadas_dmri_nulo_fmri_real": float(np.triu(ra["p"] * b, 1).sum()),
        "esperadas_dmri_real_fmri_nulo": float(np.triu(a * rb["p"], 1).sum()),
        "cruce_real": cruce(a, b), "P_real": predictores(a, b)["P"],
        "cruce_nulo_mediana": float(np.median(cruces)), "cruce_nulo_homotopicas_mediana": float(np.median(cruces_h)),
        "P_nulo_mediana": float(np.median(p_nulos)),
        "cruces_nulos": cruces, "P_nulos": p_nulos,
        "beta": {"dmri": ra["beta"], "fmri": rb["beta"], "fmri_homotopicas": rh["beta"]},
        "longitud_objetivo": {"dmri": ra["objetivo"], "fmri": rb["objetivo"], "fmri_homotopicas": rh["objetivo"]},
        "longitud_obtenida": {"dmri": ra["media"], "fmri": rb["media"], "fmri_homotopicas": rh["media"]},
        "tasa_aceptacion": {"dmri": ra["tasa"], "fmri": rb["tasa"], "fmri_homotopicas": rh["tasa"]},
        "arboles_validos": bool(ra["arboles_validos"] and rb["arboles_validos"] and rh["arboles_validos"]),
        "longitudes": {"dmri": comparar_longitudes(a, ra["p"], dist, cat),
                       "fmri": comparar_longitudes(b, rb["p"], dist, cat)},
        "segundos": time.time() - t0,
    }


def resumen_longitudes(filas):
    """Medianas entre sujetos de la comparación de longitudes y categorías, por capa."""
    salida = {}
    for capa_ in ("dmri", "fmri"):
        claves = filas[0]["longitudes"][capa_].keys()
        salida[capa_] = {k: float(np.median([f["longitudes"][capa_][k] for f in filas])) for k in claves}
    return salida


def main():
    sujetos = C.sujetos(RAIZ)
    tareas = []
    for k, s in enumerate(sujetos):
        par = C.cargar_sujeto(RAIZ, s)
        if par is not None:
            tareas.append((k, s, par[0], par[1]))
    procesos = int(os.environ.get("PROCESOS", "2"))
    with Pool(procesos) as pool:
        filas = pool.map(por_sujeto, tareas)
    real = np.array([f["reales"] for f in filas], dtype=float)
    esp = np.array([f["esperadas"] for f in filas])
    esp2 = np.array([f["esperadas_cadena_desde_azar"] for f in filas])
    esph = np.array([f["esperadas_homotopicas_fijas"] for f in filas])
    cr = np.array([f["cruce_real"] for f in filas])
    cn = np.array([f["cruce_nulo_mediana"] for f in filas])
    cnh = np.array([f["cruce_nulo_homotopicas_mediana"] for f in filas])
    lp_n = np.log(np.concatenate([f["P_nulos"] for f in filas]))
    lc_n = np.log(np.concatenate([f["cruces_nulos"] for f in filas]))
    pendiente, intercepto = np.polyfit(lp_n, lc_n, 1)
    lpr = np.log([f["P_real"] for f in filas])
    residuo = np.log(cr) - (intercepto + pendiente * lpr)
    obj = {k: np.array([f["longitud_objetivo"][k] for f in filas]) for k in ("dmri", "fmri", "fmri_homotopicas")}
    obt = {k: np.array([f["longitud_obtenida"][k] for f in filas]) for k in ("dmri", "fmri", "fmri_homotopicas")}
    res = {
        "sujetos": len(filas),
        "parametros": {"adaptacion": ADAPTACION, "refinamiento": REFINAMIENTO, "calentamiento": CALENTAMIENTO,
                       "calentamiento_desde_azar": CALENTAMIENTO_AZAR, "muestreo": MUESTREO,
                       "muestreo_desde_azar": MUESTREO_AZAR, "cada": CADA,
                       "pares_cruce": PARES_CRUCE, "semilla": SEMILLA},
        "aristas_compartidas": {
            "reales_media": float(real.mean()),
            "nulo_grados_distancia_media": float(esp.mean()),
            "razon_real_sobre_nulo_mediana": float(np.median(real / esp)),
            "sujetos_real_mayor": int(np.sum(real > esp)),
            "wilcoxon_p": float(wilcoxon(real - esp).pvalue),
            "nulo_homotopicas_fijas_media": float(esph.mean()),
            "razon_real_sobre_nulo_homotopicas_fijas_mediana": float(np.median(real / esph)),
            "sujetos_real_mayor_homotopicas_fijas": int(np.sum(real > esph)),
            "wilcoxon_homotopicas_fijas_p": float(wilcoxon(real - esph).pvalue),
            "dmri_nulo_fmri_real_media": float(np.mean([f["esperadas_dmri_nulo_fmri_real"] for f in filas])),
            "dmri_real_fmri_nulo_media": float(np.mean([f["esperadas_dmri_real_fmri_nulo"] for f in filas])),
        },
        "cruce": {
            "real_mediana": float(np.median(cr)),
            "nulo_grados_distancia_mediana": float(np.median(cn)),
            "sujetos_real_antes": int(np.sum(cr < cn)),
            "wilcoxon_p": float(wilcoxon(np.log(cr) - np.log(cn)).pvalue),
            "nulo_homotopicas_fijas_mediana": float(np.median(cnh)),
            "sujetos_real_antes_homotopicas_fijas": int(np.sum(cr < cnh)),
            "wilcoxon_homotopicas_fijas_p": float(wilcoxon(np.log(cr) - np.log(cnh)).pvalue),
            "P_real_mediana": float(np.median([f["P_real"] for f in filas])),
            "P_nulo_mediana": float(np.median([f["P_nulo_mediana"] for f in filas])),
            "modelo_en_estos_nulos": {"pendiente": float(pendiente), "intercepto": float(intercepto),
                                      "rho_log_P_log_cruce": float(np.corrcoef(lp_n, lc_n)[0, 1]),
                                      "residuo_real_mediana": float(np.median(residuo)),
                                      "residuo_real_wilcoxon_p": float(wilcoxon(residuo).pvalue)},
        },
        "comprobaciones": {
            "arboles_validos_en_todos_los_sujetos": bool(all(f["arboles_validos"] for f in filas)),
            "diferencia_cadenas_esperadas_mediana_abs": float(np.median(np.abs(esp - esp2))),
            "diferencia_cadenas_esperadas_max_abs": float(np.max(np.abs(esp - esp2))),
            "longitud_obtenida_menos_objetivo_mm_max_abs": {k: float(np.max(np.abs(obt[k] - obj[k]))) for k in obj},
            "longitud_objetivo_media_mm": {k: float(obj[k].mean()) for k in obj},
            "tasa_aceptacion_mediana": {k: float(np.median([f["tasa_aceptacion"][k] for f in filas]))
                                        for k in ("dmri", "fmri", "fmri_homotopicas")},
            "segundos_por_sujeto_mediana": float(np.median([f["segundos"] for f in filas])),
        },
        "distribucion_longitudes": resumen_longitudes(filas),
        "por_sujeto": [{k: v for k, v in f.items() if k not in ("cruces_nulos", "P_nulos")} for f in filas],
    }
    np.savez(os.path.join(AQUI, "nulo_grados_distancia.npz"),
             cruces_nulos=np.array([f["cruces_nulos"] for f in filas]),
             P_nulos=np.array([f["P_nulos"] for f in filas]), cruce_real=cr)
    with open(os.path.join(AQUI, "nulo_grados_distancia.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "por_sujeto"}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
