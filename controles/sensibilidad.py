"""Análisis de sensibilidad pedidos por la tercera revisión.

1. Densidad y binarización: capas = árbol de expansión máxima más los enlaces más fuertes
   hasta una densidad dada, con pesos (reescaladas a radio 1) o binarizadas. Separa el efecto
   de quitar enlaces del efecto de quitar magnitudes.
2. Asociación del cruce individual con el acoplamiento estructura-función, ajustada por edad,
   sexo y radios espectrales; intervalos bootstrap.
3. Concordancia de las 10 regiones más centrales en omega = 1.
4. fMRI solo con correlaciones positivas, con la curva completa y no solo un conteo.
5. Estabilidad ante ruido multiplicativo en los pesos de ambas capas.
6. Distancia de variación total entre la centralidad multicapa y la agregada, por grupo.
7. Estabilidad ante ruido de otras construcciones, para saber si la fragilidad es del árbol.

Todas las curvas usan el acoplamiento adimensional omega / radio espectral medio.
"""

import csv
import json
import os
import sys

import numpy as np
import scipy.io as sio
from scipy.stats import kruskal, mannwhitneyu, rankdata, spearmanr
from scipy.stats import f as dist_f
from scipy.stats import t as dist_t

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import rutas
sys.path.insert(0, rutas.MODULOS)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cargar as C
import multiplex as M
from controles import OMT, cruce, curva_adim, radio, rescalar

RAIZ = rutas.RAIZ
SALIDA = os.path.dirname(os.path.abspath(__file__))
DENSIDADES = ["arbol", 0.05, 0.10, 0.20, 0.40, 1.0]
SIGMAS = (0.1, 0.2)
REPETICIONES = 5
BOOTSTRAP = 2000
SEMILLA = 20261116


def capa_densidad(w, densidad, binario):
    """Árbol de expansión máxima más los enlaces más fuertes hasta la densidad pedida."""
    n = w.shape[0]
    t = M.arbol_maximo(w, binario=False)
    if densidad == "arbol":
        a = t
    else:
        objetivo = int(round(densidad * n * (n - 1) / 2))
        i, j = np.triu_indices(n, 1)
        orden = np.argsort(-w[i, j], kind="stable")
        a = t.copy()
        actuales = int((np.triu(a, 1) > 0).sum())
        for k in orden:
            if actuales >= objetivo or w[i[k], j[k]] <= 0:
                break
            if a[i[k], j[k]] == 0:
                a[i[k], j[k]] = a[j[k], i[k]] = w[i[k], j[k]]
                actuales += 1
    if binario:
        return (a > 0).astype(float)
    return rescalar(a)


def densidad_de(a):
    n = a.shape[0]
    return float((np.triu(a, 1) > 0).sum() / (n * (n - 1) / 2))


def resumen(v):
    v = np.asarray(v, dtype=float)
    fin = v[np.isfinite(v)]
    return {"n": int(len(v)), "mediana": float(np.median(fin)), "p25": float(np.percentile(fin, 25)),
            "p75": float(np.percentile(fin, 75)), "p10": float(np.percentile(fin, 10)),
            "p90": float(np.percentile(fin, 90)), "sin_cruce": int(np.sum(~np.isfinite(v)))}


def parcial_spearman(x, y, covariables):
    """Correlación de Spearman parcial: residuos de los rangos tras regresar las covariables."""
    rx, ry = rankdata(x), rankdata(y)
    z = np.column_stack([np.ones(len(x))] + [rankdata(c) for c in covariables])
    ex = rx - z @ np.linalg.lstsq(z, rx, rcond=None)[0]
    ey = ry - z @ np.linalg.lstsq(z, ry, rcond=None)[0]
    r = float(np.corrcoef(ex, ey)[0, 1])
    gl = len(x) - 2 - len(covariables)
    p = float(2 * dist_t.sf(abs(r) * np.sqrt(gl / (1 - r * r)), gl))
    return r, p


def ancova_rangos(y, grupo, covariables):
    """Prueba F de grupo sobre los rangos de y, ajustando por covariables (ANCOVA de rangos)."""
    ry = rankdata(y)
    n = len(y)
    z0 = np.column_stack([np.ones(n)] + [c if len(np.unique(c)) <= 2 else rankdata(c) for c in covariables])
    dummies = [(np.asarray(grupo) == g).astype(float) for g in ("glioma", "meningioma")]
    z1 = np.column_stack([z0] + dummies)
    rss0 = float(((ry - z0 @ np.linalg.lstsq(z0, ry, rcond=None)[0]) ** 2).sum())
    rss1 = float(((ry - z1 @ np.linalg.lstsq(z1, ry, rcond=None)[0]) ** 2).sum())
    gl1, gl2 = len(dummies), n - z1.shape[1]
    f = ((rss0 - rss1) / gl1) / (rss1 / gl2)
    return float(f), float(dist_f.sf(f, gl1, gl2))


def top(v, k=10):
    return set(np.argsort(v)[-k:].tolist())


def fc_positiva(sujeto):
    f = sio.loadmat(f"{RAIZ}/derivatives/TVB/{sujeto}/ses-preop/FC.mat")["FC_cc_DK68"].astype(float)
    f[~np.isfinite(f)] = 0.0
    bruta = f.copy()
    f = np.clip(f, 0, None)
    f = (f + f.T) / 2.0
    np.fill_diagonal(f, 0.0)
    return f, bruta


def participantes():
    with open(os.path.join(RAIZ, "participants.tsv"), encoding="utf-8") as fh:
        lector = csv.reader(fh, delimiter="\t")
        cab = [c.strip() for c in next(lector)]
        filas = {}
        for fila in lector:
            d = dict(zip(cab, [x.strip() for x in fila]))
            filas[d["participant_id"]] = d
    return filas


def main():
    rng = np.random.default_rng(SEMILLA)
    grupos = C.grupos(RAIZ)
    part = participantes()
    datos = [(s,) + C.cargar_sujeto(RAIZ, s) for s in C.sujetos(RAIZ) if C.cargar_sujeto(RAIZ, s) is not None]
    sujetos = [d[0] for d in datos]
    res = {"sujetos": sujetos}

    # 1. Densidad y binarización.
    dens = {}
    for d in DENSIDADES:
        for binario in (False, True):
            if binario and d == 1.0:
                continue
            clave = f"{'binarizada' if binario else 'ponderada'}_{d}"
            cr, dsc, dfc, razon, rho1 = [], [], [], [], []
            for s, sc, fc in datos:
                a, b = capa_densidad(sc, d, binario), capa_densidad(fc, d, binario)
                f, ref = curva_adim(a, b)
                cr.append(cruce(f, OMT))
                dsc.append(densidad_de(a)); dfc.append(densidad_de(b))
                razon.append(radio(b) / radio(a))
                rho1.append(float(spearmanr(M.versatilidad([a, b], 1.0), M.centralidad_capa(a + b)).statistic))
            dens[clave] = {"densidad_pedida": d, "densidad_dmri_mediana": float(np.median(dsc)),
                           "densidad_fmri_mediana": float(np.median(dfc)),
                           "razon_radios_mediana": float(np.median(razon)),
                           "rho_omega1_media": float(np.mean(rho1)), "cruce": resumen(cr),
                           "cruces_por_sujeto": [float(x) for x in cr]}
            print(f"{clave:>18}: dens dMRI {np.median(dsc):.3f} fMRI {np.median(dfc):.3f}  "
                  f"cruce mediana {dens[clave]['cruce']['mediana']:.3f}")
    res["densidad"] = dens
    nativa = [densidad_de(sc) for _, sc, _ in datos]
    res["densidad_nativa_dmri"] = {"mediana": float(np.median(nativa)), "min": float(np.min(nativa)),
                                   "max": float(np.max(nativa))}

    # 2. Asociación individual del cruce (árbol binarizado) con el acoplamiento estructura-función.
    i, j = np.triu_indices(datos[0][1].shape[0], 1)
    cru = np.array(dens["binarizada_arbol"]["cruces_por_sujeto"])
    acop = np.array([spearmanr(sc[i, j], fc[i, j]).statistic for _, sc, fc in datos])
    edad = np.array([float(part[s]["age"]) for s in sujetos])
    sexo = np.array([1.0 if part[s]["sex"].upper().startswith("M") else 0.0 for s in sujetos])
    radio_medio, razon = [], []
    for _, sc, fc in datos:
        a, b = M.arbol_maximo(sc), M.arbol_maximo(fc)
        radio_medio.append((radio(a) + radio(b)) / 2.0)
        razon.append(radio(b) / radio(a))
    radio_medio, razon = np.array(radio_medio), np.array(razon)
    r0 = spearmanr(cru, acop)
    rp, pp = parcial_spearman(cru, acop, [edad, sexo, radio_medio, razon])
    parciales = {nombre: parcial_spearman(cru, acop, cov) for nombre, cov in
                 (("edad_sexo", [edad, sexo]), ("radios", [radio_medio, razon]))}
    boot = []
    for _ in range(BOOTSTRAP):
        k = rng.integers(0, len(cru), len(cru))
        boot.append(spearmanr(cru[k], acop[k]).statistic)
    med_boot = [np.median(cru[rng.integers(0, len(cru), len(cru))]) for _ in range(BOOTSTRAP)]
    pacientes = [n for n, s in enumerate(sujetos) if grupos.get(s) in ("glioma", "meningioma")]
    tam = np.array([float(part[sujetos[n]]["tumor size (cub cm)"]) for n in pacientes])
    r_tam = spearmanr(cru[pacientes], tam)
    r_edad = spearmanr(cru, edad)
    u_sexo = mannwhitneyu(cru[sexo == 1], cru[sexo == 0])
    por_grupo = {g: cru[[grupos.get(s) == g for s in sujetos]] for g in ("glioma", "meningioma", "control")}
    kw = kruskal(*por_grupo.values())
    edad_grupo = {g: float(np.median(edad[[grupos.get(s) == g for s in sujetos]])) for g in por_grupo}
    acop_grupo = kruskal(*[acop[[grupos.get(s) == g for s in sujetos]] for g in por_grupo])
    res["asociacion_cruce"] = {
        "acoplamiento_media": float(acop.mean()),
        "rho_cruce_acoplamiento": float(r0.statistic), "p": float(r0.pvalue),
        "ic95_bootstrap": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
        "rho_parcial_edad_sexo_radios": rp, "p_parcial": pp,
        "rho_parcial_edad_sexo": parciales["edad_sexo"][0], "p_parcial_edad_sexo": parciales["edad_sexo"][1],
        "rho_parcial_radios": parciales["radios"][0], "p_parcial_radios": parciales["radios"][1],
        "cruce_ancova_grupo_p_edad_sexo": ancova_rangos(cru, [grupos.get(s) for s in sujetos], [edad, sexo])[1],
        "cruce_mediana_ic95_bootstrap": [float(np.percentile(med_boot, 2.5)), float(np.percentile(med_boot, 97.5))],
        "rho_cruce_tamano_tumor": float(r_tam.statistic), "p_tamano": float(r_tam.pvalue), "n_pacientes": len(pacientes),
        "rho_cruce_edad": float(r_edad.statistic), "p_edad": float(r_edad.pvalue),
        "p_sexo_mannwhitney": float(u_sexo.pvalue),
        "cruce_mediana_por_grupo": {g: float(np.median(v)) for g, v in por_grupo.items()},
        "kruskal_p_grupos": float(kw.pvalue), "edad_mediana_por_grupo": edad_grupo,
        "kruskal_p_acoplamiento_grupos": float(acop_grupo.pvalue),
        "acoplamiento_por_sujeto": acop.tolist(), "grupo_por_sujeto": [grupos.get(s) for s in sujetos],
    }
    print("asociacion:", {k: v for k, v in res["asociacion_cruce"].items() if not k.endswith("por_sujeto")})

    # 3. Concordancia de las 10 regiones más centrales y 6. variación total, en omega = 1.
    constr = {"cruda": lambda sc, fc: (sc, fc),
              "reescalada": lambda sc, fc: (rescalar(sc), rescalar(fc)),
              "arbol_binarizado": lambda sc, fc: (M.arbol_maximo(sc), M.arbol_maximo(fc))}
    conc = {}
    for nombre, f in constr.items():
        con_suma, con_fmri, tv = [], [], []
        for s, sc, fc in datos:
            a, b = f(sc, fc)
            v = M.versatilidad([a, b], 1.0)
            cs, cf = M.centralidad_capa(a + b), M.centralidad_capa(b)
            con_suma.append(len(top(v) & top(cs)) / 10.0)
            con_fmri.append(len(top(v) & top(cf)) / 10.0)
            tv.append(0.5 * np.abs(v / v.sum() - cs / cs.sum()).sum())
        tv = np.array(tv)
        kw_tv = kruskal(*[tv[[grupos.get(s) == g for s in sujetos]] for g in ("glioma", "meningioma", "control")])
        conc[nombre] = {"top10_suma_media": float(np.mean(con_suma)), "top10_suma_min": float(np.min(con_suma)),
                        "top10_fmri_media": float(np.mean(con_fmri)), "top10_fmri_min": float(np.min(con_fmri)),
                        "vt_suma_mediana": float(np.median(tv)), "vt_suma_p90": float(np.percentile(tv, 90)),
                        "vt_kruskal_p_grupos": float(kw_tv.pvalue),
                        "vt_ancova_grupo_p_edad_sexo": ancova_rangos(tv, [grupos.get(s) for s in sujetos],
                                                                     [edad, sexo])[1],
                        "vt_mediana_por_grupo": {g: float(np.median(tv[[grupos.get(s) == g for s in sujetos]]))
                                                 for g in ("glioma", "meningioma", "control")},
                        "vt_por_sujeto": tv.tolist()}
    res["omega1"] = conc
    print("omega = 1:", conc)

    # 4. fMRI solo positiva: curva completa del árbol binarizado.
    cru_pos, dif_max = [], []
    for s, sc, fc in datos:
        fp, bruta = fc_positiva(s)
        dif_max.append(float(np.abs(C.limpiar(bruta) - fc).max()))
        f, _ = curva_adim(M.arbol_maximo(sc), M.arbol_maximo(fp))
        cru_pos.append(cruce(f, OMT))
    cru_pos = np.array(cru_pos)
    res["fmri_positiva"] = {"cruce": resumen(cru_pos), "rho_con_cruce_principal": float(spearmanr(cru, cru_pos).statistic),
                            "rho_cruce_acoplamiento": float(spearmanr(cru_pos, acop).statistic),
                            "p_cruce_acoplamiento": float(spearmanr(cru_pos, acop).pvalue),
                            "comprobacion_matriz_valor_absoluto_max_dif": float(np.max(dif_max))}
    print("fMRI positiva:", res["fmri_positiva"])

    # 5. Estabilidad ante ruido multiplicativo en ambas capas.
    ruido = {}
    for sigma in SIGMAS:
        medias, rhos_acop = [], []
        por_rep = np.zeros((REPETICIONES, len(datos)))
        acop_rep = np.zeros((REPETICIONES, len(datos)))
        for n, (s, sc, fc) in enumerate(datos):
            for r in range(REPETICIONES):
                ps, pf = M.perturbar(sc, sigma, rng), M.perturbar(fc, sigma, rng)
                f, _ = curva_adim(M.arbol_maximo(ps), M.arbol_maximo(pf))
                por_rep[r, n] = cruce(f, OMT)
                acop_rep[r, n] = spearmanr(ps[i, j], pf[i, j]).statistic
        estab = [float(spearmanr(cru, por_rep[r]).statistic) for r in range(REPETICIONES)]
        asoc = [float(spearmanr(por_rep[r], acop_rep[r]).statistic) for r in range(REPETICIONES)]
        ruido[str(sigma)] = {"rho_cruce_original_vs_perturbado_mediana": float(np.median(estab)),
                             "rho_cruce_original_vs_perturbado_min": float(np.min(estab)),
                             "rho_cruce_acoplamiento_mediana": float(np.median(asoc)),
                             "rho_cruce_acoplamiento_rango": [float(np.min(asoc)), float(np.max(asoc))],
                             "cruce_mediana": float(np.median(por_rep))}
        print(f"ruido {sigma}:", ruido[str(sigma)])
    res["ruido_multiplicativo"] = ruido

    # 7. La fragilidad, ¿es propia del árbol? Misma prueba (sigma = 0,1) en otras construcciones.
    comparadas = {"arbol_ponderado": lambda w: capa_densidad(w, "arbol", False),
                  "densidad_0.1_binarizada": lambda w: capa_densidad(w, 0.10, True),
                  "densidad_0.1_ponderada": lambda w: capa_densidad(w, 0.10, False)}
    comp = {"arbol_binarizado": ruido["0.1"]["rho_cruce_original_vs_perturbado_mediana"]}
    for nombre, construir in comparadas.items():
        orig = np.array([cruce(curva_adim(construir(sc), construir(fc))[0], OMT) for _, sc, fc in datos])
        por_rep = np.zeros((REPETICIONES, len(datos)))
        for n, (s, sc, fc) in enumerate(datos):
            for r in range(REPETICIONES):
                ps, pf = M.perturbar(sc, 0.1, rng), M.perturbar(fc, 0.1, rng)
                por_rep[r, n] = cruce(curva_adim(construir(ps), construir(pf))[0], OMT)
        comp[nombre] = float(np.median([spearmanr(orig, por_rep[r]).statistic for r in range(REPETICIONES)]))
        print(f"ruido 0.1, {nombre}: rho original vs perturbado = {comp[nombre]:.3f}")
    res["ruido_estabilidad_por_construccion_sigma_0.1"] = comp

    with open(os.path.join(SALIDA, "sensibilidad.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1)
    print("listo")


if __name__ == "__main__":
    main()
