"""Calcula todos los números y curvas del póster y del resumen desde los datos crudos.

Salidas en resultados/: resultados.json (cifras), curvas.npz (barridos por sujeto) y
ejemplo.npz (matrices de un paciente para la figura de métodos).
"""

import json
import os
import sys

import numpy as np
from scipy.stats import kruskal, pearsonr, spearmanr, wilcoxon

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rutas
sys.path.insert(0, rutas.MODULOS)
import cargar as C
import multiplex as M

RAIZ = rutas.RAIZ
SALIDA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resultados")
OM = np.logspace(-4, 4, 65)
SEMILLA = 20261030
PERMUTACIONES = 50
EJEMPLO = "sub-PAT16"


def radio(m):
    return float(np.abs(np.linalg.eigvalsh(m)).max())


def rescalar(m):
    r = radio(m)
    return m / r if r > 0 else m


def triangulo(m):
    i, j = np.triu_indices(m.shape[0], 1)
    return m[i, j]


def autovector_supra(capas, omega):
    """Autovector principal de la supra-adyacencia, normalizado a suma 1."""
    s = M.supra_adyacencia(capas, omega)
    w, v = np.linalg.eigh(s)
    x = np.abs(v[:, int(np.argmax(w))])
    return x / x.sum()


def cruce(fila, umbral=0.95):
    """Primer omega donde la curva de un sujeto alcanza el umbral, interpolado en log."""
    sobre = np.where(fila >= umbral)[0]
    if len(sobre) == 0:
        return np.inf
    k = sobre[0]
    if k == 0:
        return float(OM[0])
    lo = np.interp(umbral, [fila[k - 1], fila[k]], [np.log10(OM[k - 1]), np.log10(OM[k])])
    return float(10 ** lo)


CONSTRUCCIONES = {
    "cruda": lambda sc, fc: (sc, fc),
    "reescalada": lambda sc, fc: (rescalar(sc), rescalar(fc)),
    "arbol_ponderado": lambda sc, fc: (M.arbol_maximo(sc, binario=False),
                                       M.arbol_maximo(fc, binario=False)),
    "arbol_ponderado_reescalado": lambda sc, fc: (rescalar(M.arbol_maximo(sc, binario=False)),
                                                  rescalar(M.arbol_maximo(fc, binario=False))),
    "arbol_binarizado": lambda sc, fc: (M.arbol_maximo(sc), M.arbol_maximo(fc)),
}


def main():
    os.makedirs(SALIDA, exist_ok=True)
    grupos = C.grupos(RAIZ)
    datos = []
    for s in C.sujetos(RAIZ):
        par = C.cargar_sujeto(RAIZ, s)
        if par is not None:
            datos.append((s, par[0], par[1]))
    print(f"sujetos: {len(datos)}")

    res = {"n_sujetos": len(datos), "n_regiones": int(datos[0][1].shape[0]),
           "grupos": {g: sum(1 for s, _, _ in datos if grupos.get(s) == g)
                      for g in ("glioma", "meningioma", "control")}}

    sfc = [float(spearmanr(triangulo(sc), triangulo(fc)).statistic) for _, sc, fc in datos]
    res["acoplamiento_global"] = {"media": float(np.mean(sfc)), "de": float(np.std(sfc, ddof=1))}

    curvas = {}
    res["construcciones"] = {}
    for nombre, construir in CONSTRUCCIONES.items():
        razon, frac_fmri, r_dti, r_fmri, r_suma, r_prom, cruces = [], [], [], [], [], [], []
        pe_fmri, pe_suma, tv_fmri, tv_suma = [], [], [], []
        cur_suma, cur_fmri = [], []
        for s, sc, fc in datos:
            c1, c2 = construir(sc, fc)
            n = c1.shape[0]
            razon.append(radio(c2) / radio(c1))
            ce = M.centralidad_capa(c1)
            cf = M.centralidad_capa(c2)
            cs = M.centralidad_capa(c1 + c2)
            cp = (ce + cf) / 2.0
            x = autovector_supra([c1, c2], 1.0)
            v1 = x[:n] + x[n:]
            frac_fmri.append(float(x[n:].sum()))
            r_dti.append(spearmanr(v1, ce).statistic)
            r_fmri.append(spearmanr(v1, cf).statistic)
            r_suma.append(spearmanr(v1, cs).statistic)
            r_prom.append(spearmanr(v1, cp).statistic)
            vn, fn, sn = v1 / v1.sum(), cf / cf.sum(), cs / cs.sum()
            pe_fmri.append(pearsonr(vn, fn).statistic)
            pe_suma.append(pearsonr(vn, sn).statistic)
            tv_fmri.append(0.5 * np.abs(vn - fn).sum())
            tv_suma.append(0.5 * np.abs(vn - sn).sum())
            fs, ff = [], []
            for om in OM:
                v = M.versatilidad([c1, c2], om)
                fs.append(spearmanr(v, cs).statistic)
                ff.append(spearmanr(v, cf).statistic)
            fs = np.array(fs)
            cur_suma.append(fs)
            cur_fmri.append(np.array(ff))
            cruces.append(cruce(fs))
        cr = np.array(cruces)
        fin = cr[np.isfinite(cr)]
        r_suma = np.array(r_suma)
        curvas[f"{nombre}_suma"] = np.array(cur_suma)
        curvas[f"{nombre}_fmri"] = np.array(cur_fmri)
        res["construcciones"][nombre] = {
            "razon_radios_mediana": float(np.median(razon)),
            "razon_radios_media": float(np.mean(razon)),
            "fraccion_autovector_fmri_w1": float(np.mean(frac_fmri)),
            "rho_w1_vs_dti": float(np.mean(r_dti)),
            "rho_w1_vs_fmri": float(np.mean(r_fmri)),
            "rho_w1_vs_suma": float(np.mean(r_suma)),
            "rho_w1_vs_suma_min": float(np.min(r_suma)),
            "rho_w1_vs_promedio": float(np.mean(r_prom)),
            "pearson_w1_vs_fmri": float(np.mean(pe_fmri)),
            "pearson_w1_vs_suma": float(np.mean(pe_suma)),
            "dif_total_valores_w1_vs_fmri": float(np.mean(tv_fmri)),
            "dif_total_valores_w1_vs_suma": float(np.mean(tv_suma)),
            "sujetos_w1_suma_sobre_095": int(np.sum(r_suma >= 0.95)),
            "sujetos_w1_suma_sobre_090": int(np.sum(r_suma > 0.90)),
            "cruce_mediana": float(np.median(fin)) if len(fin) else None,
            "cruce_p10": float(np.percentile(fin, 10)) if len(fin) else None,
            "cruce_p90": float(np.percentile(fin, 90)) if len(fin) else None,
            "sin_cruce": int(np.sum(~np.isfinite(cr))),
            "cruce_bajo_1": int(np.sum(fin < 1.0)),
            "cruce_en_transicion_05_2": int(np.sum((fin > 0.5) & (fin < 2.0))),
            "cruces": [float(c) for c in cr],
            "sujetos": [s for s, _, _ in datos],
        }
        print(f"{nombre:>28}: razon {np.median(razon):7.3f}  fMRI {np.mean(frac_fmri):.4f}  "
              f"vs_dti {np.mean(r_dti):.4f}  vs_fmri {np.mean(r_fmri):.4f}  "
              f"vs_suma {np.mean(r_suma):.4f}  vs_prom {np.mean(r_prom):.4f}  "
              f"cruce {res['construcciones'][nombre]['cruce_mediana']}")

    rng = np.random.default_rng(SEMILLA)
    obs, nul, por_grupo = [], [], {"glioma": [], "meningioma": [], "control": []}
    for s, sc, fc in datos:
        capas = [M.arbol_maximo(sc), M.arbol_maximo(fc)]
        n = capas[0].shape[0]
        v = M.versatilidad(capas, 1.0)
        prom = (M.centralidad_capa(capas[0]) + M.centralidad_capa(capas[1])) / 2.0
        obs.append(spearmanr(v, prom).statistic)
        nul.append(np.mean([spearmanr(M.versatilidad_nula(capas, 1.0, rng.permutation(n)),
                                      prom).statistic for _ in range(PERMUTACIONES)]))
        g = grupos.get(s)
        if g in por_grupo:
            suma = M.centralidad_capa(capas[0] + capas[1])
            por_grupo[g].append(spearmanr(v, suma).statistic)
    obs, nul = np.array(obs), np.array(nul)
    w = wilcoxon(obs, nul)
    res["nulo"] = {"observado": float(obs.mean()), "nulo": float(nul.mean()),
                   "W": float(w.statistic), "p": float(w.pvalue),
                   "sujetos_misma_direccion": int(np.sum(obs > nul)),
                   "por_sujeto_obs": obs.tolist(), "por_sujeto_nulo": nul.tolist()}
    kw = kruskal(*[por_grupo[g] for g in ("glioma", "meningioma", "control")])
    res["grupos_w1_vs_suma"] = {g: {"media": float(np.mean(v)), "min": float(np.min(v)), "n": len(v)}
                                for g, v in por_grupo.items()}
    res["grupos_w1_vs_suma"]["kruskal_p"] = float(kw.pvalue)
    res["grupos_w1_vs_suma"]["kruskal_H"] = float(kw.statistic)
    print(f"nulo: {obs.mean():.3f} -> {nul.mean():.3f}  W={w.statistic} p={w.pvalue:.2g}  "
          f"misma direccion {np.sum(obs > nul)}/{len(obs)}")
    print("grupos:", {g: (round(np.mean(v), 3), round(np.min(v), 3)) for g, v in por_grupo.items()},
          f"KW p={kw.pvalue:.3f}")

    rng_al = np.random.default_rng(20261113)
    supera, zs = 0, []
    for s, sc, fc in datos:
        o = spearmanr(triangulo(sc), triangulo(fc)).statistic
        nul = []
        for _ in range(1000):
            q = rng_al.permutation(sc.shape[0])
            nul.append(spearmanr(triangulo(sc), triangulo(fc[np.ix_(q, q)])).statistic)
        nul = np.array(nul)
        supera += int(o > nul.max())
        zs.append(float((o - nul.mean()) / nul.std()))
    res["alineacion_capas"] = {"permutaciones": 1000, "sujetos_sobre_maximo_nulo": supera,
                               "z_medio": float(np.mean(zs)), "z_min": float(np.min(zs))}
    print(f"alineacion: acoplamiento real supera al maximo de 1000 reordenamientos en {supera}/{len(datos)}")

    import scipy.io as sio
    sens = {}
    for nombre in ("cruda", "reescalada", "arbol_binarizado"):
        vf, vs, crs = [], [], []
        for s, sc, _ in datos:
            f = sio.loadmat(f"{RAIZ}/derivatives/TVB/{s}/ses-preop/FC.mat")["FC_cc_DK68"].astype(float)
            f[~np.isfinite(f)] = 0.0
            f = np.clip(f, 0, None)
            f = (f + f.T) / 2.0
            np.fill_diagonal(f, 0.0)
            c1, c2 = CONSTRUCCIONES[nombre](sc, f)
            v = M.versatilidad([c1, c2], 1.0)
            cs = M.centralidad_capa(c1 + c2)
            vf.append(spearmanr(v, M.centralidad_capa(c2)).statistic)
            vs.append(spearmanr(v, cs).statistic)
            if nombre == "arbol_binarizado":
                fila = np.array([spearmanr(M.versatilidad([c1, c2], om), cs).statistic for om in OM])
                crs.append(cruce(fila))
        vs = np.array(vs)
        sens[nombre] = {"rho_w1_vs_fmri": float(np.mean(vf)), "rho_w1_vs_suma": float(np.mean(vs)),
                        "sujetos_w1_suma_sobre_095": int(np.sum(vs >= 0.95))}
        if crs:
            sens[nombre]["cruce_mediana"] = float(np.median(crs))
    res["sensibilidad_fc_positiva"] = sens
    print("sensibilidad (solo FC positiva):", {k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in sens.items()})

    sc, fc = next((a, b) for s, a, b in datos if s == EJEMPLO)
    np.savez(os.path.join(SALIDA, "ejemplo.npz"), sc=sc, fc=fc,
             arbol_sc=M.arbol_maximo(sc), arbol_fc=M.arbol_maximo(fc))
    np.savez(os.path.join(SALIDA, "curvas.npz"), omegas=OM, **curvas)
    with open(os.path.join(SALIDA, "resultados.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1)
    print("listo:", SALIDA)


if __name__ == "__main__":
    main()
