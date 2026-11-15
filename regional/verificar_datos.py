"""Verificación de los derivados descargados de OpenNeuro (ds001226 y ds002080).

Comprueba que las máscaras tumorales y los centroides de SC.zip sirven para los análisis
espaciales y regionales:

1. Los pesos de SC.zip coinciden con las matrices SCthrAn locales y el orden de regiones de
   centres.txt es el de parcellation.txt.
2. El volumen de cada máscara concuerda con el tamaño tumoral del registro de participantes.
3. Máscaras y centroides comparten el sistema de coordenadas de cada sujeto: la región más
   cercana al centroide tumoral cae en el lóbulo que informa el registro, y coincide más con
   los centroides del propio sujeto que con los de otros sujetos (los sistemas de coordenadas
   nativos difieren entre sujetos, sobre todo en el eje vertical).
4. La región sin conexiones estructurales de sub-PAT26 y su distancia al tumor.

Guarda la distancia de cada región al tumor (al vóxel tumoral más cercano y al centroide)
para el análisis regional.
"""

import json
import os
import sys

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
sys.path.insert(0, os.path.join(AQUI, "..", "controles"))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import rutas
sys.path.insert(0, rutas.MODULOS)
import cargar as C
from lectura import leer_centros, leer_nifti, leer_pesos_zip, ruta_mascara
from validacion_P import LOBULO, regiones

RAIZ = rutas.RAIZ
UMBRAL_MASCARA = 0.5
LOBULOS_REGISTRO = {"Frontal": {"F"}, "Parietal": {"P"}, "Temporal": {"T"}, "Fronto-temporal": {"F", "T"},
                    "Frontal skullbase": {"F"}}


def voxeles_mundo(vol, afin, umbral):
    idx = np.argwhere(vol >= umbral)
    if len(idx) == 0:
        idx = np.argwhere(vol > 0)
    pesos = vol[tuple(idx.T)]
    mundo = (afin @ np.c_[idx, np.ones(len(idx))].T).T[:, :3]
    return mundo, pesos


def main():
    tabla = C.tabla(RAIZ)
    sujetos = C.sujetos(RAIZ)
    pacientes = [s for s in sujetos if s.startswith("sub-PAT")]
    nombres_local = regiones()
    res = {}

    # 1. Orden de regiones y coincidencia de pesos.
    mismo_orden, rho_pesos, ceros_iguales = [], [], []
    iu = np.triu_indices(68, 1)
    centros = {}
    for s in sujetos:
        nombres, xyz = leer_centros(s)
        centros[s] = xyz
        mismo_orden.append([a[4:6] + a[7:] for a in nombres_local] == [b[:2] + b[3:] for b in nombres])
        w = leer_pesos_zip(s)
        sc, _ = C.cargar_sujeto(RAIZ, s)
        rho_pesos.append(float(spearmanr(w[iu], sc[iu]).statistic))
        ceros_iguales.append(bool(np.array_equal(w[iu] == 0, sc[iu] == 0)))
    res["orden_y_pesos"] = {"sujetos": len(sujetos), "mismo_orden_que_parcellation": int(sum(mismo_orden)),
                            "rho_pesos_zip_vs_scthran_min": float(min(rho_pesos)),
                            "mismos_ceros": int(sum(ceros_iguales))}

    # 2 y 3. Volumen, centroide y lóbulo por paciente.
    filas, dist_borde, dist_centroide = [], [], []
    vol_mascara, vol_registro = [], []
    ajeno = []
    for s in pacientes:
        vol, afin, _, info = leer_nifti(ruta_mascara(s))
        mundo, pesos = voxeles_mundo(vol, afin, UMBRAL_MASCARA)
        centroide = (mundo * pesos[:, None]).sum(0) / pesos.sum()
        arbol = cKDTree(mundo)
        db = arbol.query(centros[s])[0]
        dc = np.linalg.norm(centros[s] - centroide, axis=1)
        dist_borde.append(db)
        dist_centroide.append(dc)
        cerca = np.argsort(dc)[:3]
        lob = [LOBULO[nombres_local[k][7:]] for k in cerca]
        lugar = tabla[s].get("tumor location", "")
        esperado = LOBULOS_REGISTRO.get(lugar)
        if esperado is not None:
            otros = [LOBULO[nombres_local[int(np.argmin(np.linalg.norm(centros[t] - centroide, axis=1)))][7:]]
                     for t in sujetos if t != s]
            ajeno.append(float(np.mean([x in esperado for x in otros])))
        volumen_suave = float(vol.sum()) / 1000.0
        vol_mascara.append(volumen_suave)
        vol_registro.append(float(tabla[s]["tumor size (cub cm)"]))
        filas.append({
            "sujeto": s, "tipo": tabla[s].get("tumor type & grade"), "lugar_registro": lugar,
            "volumen_registro_cm3": vol_registro[-1], "volumen_mascara_cm3": round(volumen_suave, 2),
            "voxeles_sobre_umbral": int(len(pesos)), "centroide_mm": [round(float(x), 1) for x in centroide],
            "regiones_mas_cercanas": [nombres_local[k] for k in cerca],
            "distancia_region_mas_cercana_mm": round(float(dc.min()), 1),
            "coincide_region_mas_cercana": None if esperado is None else bool(lob[0] in esperado),
            "coincide_alguna_de_3": None if esperado is None else bool(set(lob) & esperado),
            "sform_code": info["sform_code"],
        })
    evaluables = [f for f in filas if f["coincide_region_mas_cercana"] is not None]
    centro_cerebral = np.array([centros[t].mean(0) for t in sujetos])
    res["mascaras"] = {
        "pacientes": len(filas),
        "rho_volumen_mascara_vs_registro": float(spearmanr(vol_mascara, vol_registro).statistic),
        "razon_volumen_mediana": float(np.median(np.array(vol_mascara) / np.array(vol_registro))),
        "lobulo_evaluables": len(evaluables),
        "lobulo_coincide_region_mas_cercana": int(sum(f["coincide_region_mas_cercana"] for f in evaluables)),
        "lobulo_coincide_alguna_de_3": int(sum(f["coincide_alguna_de_3"] for f in evaluables)),
        "lobulo_coincide_con_centroides_de_otros_sujetos_media": float(np.sum(ajeno)),
        "desviacion_centro_cerebral_entre_sujetos_mm": [round(float(x), 1) for x in centro_cerebral.std(0)],
        "por_paciente": filas,
    }

    # 4. Región aislada de sub-PAT26.
    sc26, _ = C.cargar_sujeto(RAIZ, "sub-PAT26")
    aislada = [int(k) for k in np.where(sc26.sum(1) == 0)[0]]
    k26 = pacientes.index("sub-PAT26")
    vol, afin, _, _ = leer_nifti(ruta_mascara("sub-PAT26"))
    vox = np.round(np.linalg.solve(afin, np.r_[centros["sub-PAT26"][aislada[0]], 1.0])[:3]).astype(int)
    dentro = bool(np.all(vox >= 0) and np.all(vox < vol.shape) and vol[tuple(vox)] >= UMBRAL_MASCARA)
    orden_db = np.argsort(dist_borde[k26])
    res["sub_PAT26"] = {
        "region_sin_conexiones": [nombres_local[k] for k in aislada],
        "distancia_al_borde_tumoral_mm": float(dist_borde[k26][aislada[0]]),
        "centroide_dentro_del_tumor": dentro,
        "puesto_por_cercania_al_tumor": int(np.where(orden_db == aislada[0])[0][0]) + 1,
    }

    np.savez(os.path.join(AQUI, "distancias_tumor.npz"), pacientes=np.array(pacientes),
             regiones=np.array(nombres_local), distancia_borde=np.array(dist_borde),
             distancia_centroide=np.array(dist_centroide), volumen_mascara=np.array(vol_mascara))
    with open(os.path.join(AQUI, "verificacion_datos.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1)
    resumen = {k: v for k, v in res.items()}
    resumen["mascaras"] = {k: v for k, v in res["mascaras"].items() if k != "por_paciente"}
    print(json.dumps(resumen, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
