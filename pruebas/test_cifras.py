"""Verificación independiente de las cifras del estudio.

Recalcula con otra implementación las cifras que sostienen las conclusiones (árboles con networkx,
descriptor P por seudoinversa, cruce con barrido denso) y revisa la coherencia interna de los
resultados guardados: efectos pareados, contrastes regionales, tipo tumoral y demostración SHAP.
"""

import json
import os
import sys

import networkx as nx
import numpy as np
import pytest
from scipy.stats import rankdata

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import rutas
sys.path.insert(0, rutas.MODULOS)
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, "controles"))
sys.path.insert(0, os.path.join(BASE, "regional"))
import cargar as C

RAIZ = rutas.RAIZ


def leer(*partes):
    with open(os.path.join(BASE, *partes), encoding="utf-8") as fh:
        return json.load(fh)


SIN_DATOS = not os.path.isfile(os.path.join(RAIZ, "participants.tsv"))


@pytest.fixture(scope="module")
def datos():
    if SIN_DATOS:
        pytest.skip("faltan los datos de OpenNeuro ds001226 (ver «Datos» en el README)")
    salida = {}
    for s in C.sujetos(RAIZ):
        par = C.cargar_sujeto(RAIZ, s)
        if par is not None:
            salida[s] = par
    return salida


def arbol_nx(w):
    g = nx.Graph()
    n = w.shape[0]
    g.add_nodes_from(range(n))
    i, j = np.nonzero(np.triu(w, 1))
    for a, b in zip(i, j):
        g.add_edge(int(a), int(b), weight=float(w[a, b]))
    t = nx.maximum_spanning_tree(g)
    m = np.zeros((n, n))
    for a, b in t.edges():
        m[a, b] = m[b, a] = 1.0
    return m


def test_aristas_compartidas_con_networkx(datos):
    ap = leer("controles", "aristas_por_sujeto.json")
    compartidas = []
    for s in ap["sujetos"]:
        a, b = arbol_nx(datos[s][0]), arbol_nx(datos[s][1])
        compartidas.append(int(((np.triu(a, 1) > 0) & (np.triu(b, 1) > 0)).sum()))
    assert compartidas == ap["reales"]
    assert abs(np.mean(compartidas) - 11.25) < 1e-9


def test_descriptor_P_por_seudoinversa(datos):
    from controles import radio
    d = np.load(os.path.join(BASE, "controles", "explicacion_espectral.npz"))
    sujetos = leer("controles", "explicacion_espectral.json")["sujetos"]
    for k, s in enumerate(sujetos):
        a, b = arbol_nx(datos[s][0]), arbol_nx(datos[s][1])
        c, dd = (a + b) / 2.0, (a - b) / 2.0
        lam, vec = np.linalg.eigh(c)
        v1 = vec[:, -1] * np.sign(vec[:, -1].sum())
        proy = np.eye(len(v1)) - np.outer(v1, v1)
        delta = np.linalg.pinv(lam[-1] * np.eye(len(v1)) - c, rcond=1e-10) @ (proy @ (dd @ (dd @ v1)))
        p = np.linalg.norm(delta) / (2.0 * np.std(v1) * (radio(a) + radio(b)) / 2.0)
        assert p == pytest.approx(d["real_P"][k], rel=1e-6)


@pytest.mark.parametrize("indice", [0, 12, 31])
def test_cruce_con_barrido_denso(datos, indice):
    from controles import radio
    d = np.load(os.path.join(BASE, "controles", "explicacion_espectral.npz"))
    s = leer("controles", "explicacion_espectral.json")["sujetos"][indice]
    a, b = arbol_nx(datos[s][0]), arbol_nx(datos[s][1])
    n = a.shape[0]
    ref = (radio(a) + radio(b)) / 2.0
    suma = np.abs(np.linalg.eigh(a + b)[1][:, -1])
    for w in np.logspace(-2.5, 1.5, 400):
        s_ = np.block([[a, w * ref * np.eye(n)], [w * ref * np.eye(n), b]])
        x = np.abs(np.linalg.eigh(s_)[1][:, -1])
        r = np.corrcoef(rankdata(x[:n] + x[n:]), rankdata(suma))[0, 1]
        if r >= 0.95:
            break
    assert w == pytest.approx(d["real_cruce"][indice], rel=0.05)


def test_comprobaciones_de_los_analisis_nuevos():
    v = leer("regional", "verificacion_datos.json")
    assert v["orden_y_pesos"]["mismo_orden_que_parcellation"] == 36
    assert v["orden_y_pesos"]["mismos_ceros"] == 36
    assert v["orden_y_pesos"]["rho_pesos_zip_vs_scthran_min"] > 0.999999
    assert v["mascaras"]["rho_volumen_mascara_vs_registro"] > 0.95
    assert v["sub_PAT26"]["region_sin_conexiones"] == ["ctx-rh-transversetemporal"]
    g = leer("controles", "nulo_grados_distancia.json")["comprobaciones"]
    assert g["arboles_validos_en_todos_los_sujetos"]
    assert g["diferencia_cadenas_esperadas_max_abs"] < 0.5
    assert max(g["longitud_obtenida_menos_objetivo_mm_max_abs"].values()) < 2.0
    m = leer("controles", "mezcla_cadena.json")
    assert m["propuesta_simetrica"] and m["beta_fijo_durante_el_muestreo"]
    assert m["ess_dmri"]["minimo"] >= 50 and m["ess_fmri"]["minimo"] >= 50
    e = leer("regional", "espejo_mascaras.json")
    assert e["misma_grilla"] == e["pacientes"] == e["coinciden_con_t1_invertido"] == 25
    assert e["min_dif_con_t1_sin_invertir_mm"] > 100
    a = leer("regional", "alineacion_t1.json")
    for s in ("sub-PAT01", "sub-PAT26"):
        assert a[s]["misma_forma"] and a[s]["signo_eje_x"]["t1"] == -a[s]["signo_eje_x"]["mascara"]
    t = a["sub-PAT26"]["intensidad_t1_mediana"]
    assert t["mascara_por_coordenadas_de_mundo"] < t["mascara_voxel_a_voxel_en_espejo"]


def test_umbral_mezcla_y_regional():
    su = leer("controles", "sensibilidad_umbral.json")
    for u in ("0.9", "0.99"):
        for t in ("mismo_grado", "uniforme", "barajada"):
            f = su["resumen"][u][t]
            assert su["resumen"][u]["real_mediana"] < f["nulo_mediana"], (u, t)
            assert f["sujetos_real_antes"] >= 30 and f["wilcoxon_p"] < 0.001, (u, t)
    assert su["comprobacion_095"]["rho_con_cruce_principal"] > 0.99
    area = su["resumen"]["area_1_menos_rho"]
    for t in ("mismo_grado", "uniforme", "barajada"):
        assert area["real_mediana"] < area[t]["nulo_mediana"] and area[t]["wilcoxon_p"] < 0.001
    dl = leer("controles", "nulo_grados_distancia.json")["distribucion_longitudes"]
    assert dl["dmri"]["ks"] < dl["fmri"]["ks"]
    m = leer("controles", "mezcla_cadena.json")["ess_entre_pares_nulos"]
    assert m["cruce"]["ess_mediana"] >= 80 and m["P"]["ess_mediana"] >= 80
    reg = leer("regional", "analisis_regional.json")
    for bloque in ("omega_1", "omega_adim_0.1", "omega_adim_0.28"):
        for medida in ("reordenamiento", "fraccion_estructural"):
            assert reg[bloque][medida]["regiones_excluidas_por_sd_cero"] == 0


def test_efectos_pareados_son_coherentes():
    ef = leer("controles", "efectos_bootstrap.json")
    gd = leer("controles", "nulo_grados_distancia.json")
    ee = leer("controles", "explicacion_espectral.json")
    ap = leer("controles", "aristas_por_sujeto.json")
    d = np.load(os.path.join(BASE, "controles", "explicacion_espectral.npz"))
    for t in ("mismo_grado", "uniforme", "barajada"):
        propio = np.log(d["real_cruce"] / np.median(d[f"{t}_cruce"], axis=1))
        assert np.allclose(propio, ef["por_sujeto"]["log_razon_cruce"][t])
        assert ef["cruce"][t]["sujetos_real_antes"] == ee["controles_500"][t]["sujetos_real_antes"]
    c, a = ef["cruce"], ef["aristas"]
    assert c["grados_y_distancia"]["sujetos_real_antes"] == gd["cruce"]["sujetos_real_antes"]
    assert c["homotopicas_fijas"]["sujetos_real_antes"] == gd["cruce"]["sujetos_real_antes_homotopicas_fijas"]
    assert a["grados_y_distancia"]["sujetos_real_mayor"] == gd["aristas_compartidas"]["sujetos_real_mayor"]
    esperado = np.mean(ap["reales"]) - ap["resumen"]["grados_y_distancia"]["media"]
    assert abs(a["grados_y_distancia"]["media_exceso"] - esperado) < 1e-9
    for t, r in c.items():
        v = np.array(ef["por_sujeto"]["log_razon_cruce"][t])
        assert np.isclose(np.median(v), r["mediana_log_razon"])
        assert r["ic95_log_razon"][0] < r["mediana_log_razon"] < r["ic95_log_razon"][1]
    for t, r in a.items():
        v = np.array(ef["por_sujeto"]["exceso_aristas"][t])
        assert np.isclose(np.mean(v), r["media_exceso"])
        assert r["ic95_exceso"][0] < r["media_exceso"] < r["ic95_exceso"][1]
    # la ventaja se atenúa al agregar restricciones, sin desaparecer
    assert c["grados_y_distancia"]["mediana_log_razon"] < c["homotopicas_fijas"]["mediana_log_razon"] < 0
    assert c["homotopicas_fijas"]["ic95_log_razon"][1] < 0 and a["homotopicas_fijas"]["ic95_exceso"][0] > 0
    assert ef["censura_sensibilidad_umbral"]["0.9"]["reales_bajo_la_grilla"] == ["sub-PAT24"]
    assert all(not ef["censura_sensibilidad_umbral"][u]["reales_bajo_la_grilla"] for u in ("0.95", "0.99"))



def test_jerarquia_regional_es_coherente_con_sus_mapas():
    import jerarquia as J
    from scipy.stats import spearmanr
    j = leer("regional", "jerarquia.json")
    z = np.load(os.path.join(BASE, "regional", "jerarquia.npz"))
    nombres = [str(x) for x in z["nombres"]]
    base = np.array([x[7:] for x in nombres])
    jer = z["jerarquia"]
    h = j["jerarquia_hcp"]
    assert abs(h["separacion_dmn_menos_sensoriales_en_de"][h["componente_usado"] - 1]) > 1.5
    assert jer[np.isin(base, J.MODO_POR_DEFECTO)].mean() > jer[np.isin(base, J.SENSORIALES)].mean()
    assert j["jerarquia_cohorte"]["rho_con_jerarquia_hcp"] < 0.5
    c = j["contrastes"]["omega_1"]
    for medida in ("reordenamiento", "reordenamiento_con_signo", "fraccion_estructural"):
        for ref, clave in ((jer, "vs_jerarquia"), (z["acoplamiento_hcp"], "vs_acoplamiento_hcp"),
                           (z["acoplamiento_cohorte"], "vs_acoplamiento_cohorte")):
            ok = np.isfinite(z[medida]) & np.isfinite(ref)
            e = c[medida][clave]
            assert spearmanr(z[medida][ok], ref[ok])[0] == pytest.approx(e["rho"], abs=1e-9)
            espaciales = [e[f"p_{x}"] for x in J.ESPACIALES]
            assert e["p_maximo_espacial"] == max(espaciales)
            assert all(1.0 / (j["rotaciones"] + 1) <= e[f"p_{x}"] <= 1.0 for x in J.NULOS)
    f = j["familia_principal_omega_1"]
    assert np.allclose(J.holm(f["p_rotacion"]), f["p_holm"])
    assert j["regiones_sin_valor_por_rotacion"]["media"] < 3
    assert j["acoplamiento_regional"]["cohorte_transformada_como_hcp"]["rho_con_cohorte_sin_transformar"] > 0.9

    jr, jv = c["reordenamiento"], j["validacion"]["acoplamiento_hcp_vs_jerarquia"]
    assert jv["rho"] < 0 and min(jv[f"p_{x}"] for x in J.ESPACIALES) > 0.05
    assert all(e["p_rotacion"] > 0.05 for e in (jr["vs_jerarquia"], jr["vs_acoplamiento_hcp"]))


def test_tipo_tumoral_y_volumen():
    from scipy.stats import mannwhitneyu, spearmanr
    tu = leer("regional", "tumor.json")
    sen = leer("controles", "sensibilidad.json")
    grupo = np.array(sen["asociacion_cruce"]["grupo_por_sujeto"])
    cruce = np.load(os.path.join(BASE, "controles", "explicacion_espectral.npz"))["real_cruce"]
    t = tu["tipo"]["cruce"]
    assert mannwhitneyu(cruce[grupo == "glioma"], cruce[grupo == "meningioma"])[1] == pytest.approx(t["mannwhitney_p"])
    assert np.median(cruce[grupo == "glioma"]) == pytest.approx(t["mediana_glioma"])
    assert t["mannwhitney_p"] < 0.05 < t["p_ancova_rangos_edad_sexo"] and t["p_permutacion_edad_sexo"] > 0.05
    assert tu["edad_mediana"]["meningioma"] > tu["edad_mediana"]["glioma"] and tu["glioma"] + tu["meningioma"] == 25
    if not SIN_DATOS:
        import tumor as T
        p = T.participantes()
        suj = leer("controles", "explicacion_espectral.json")["sujetos"]
        pac = grupo != "control"
        vol = np.array([float(p[x]["tumor size (cub cm)"]) for x in suj])
        assert spearmanr(np.log(cruce[pac]), vol[pac])[0] == pytest.approx(tu["volumen"]["cruce"]["rho"])
    vd = leer("regional", "verificacion_datos.json")["mascaras"]
    assert vd["lobulo_coincide_region_mas_cercana"] >= 0.75 * vd["lobulo_evaluables"]


def test_demostracion_shap():
    sh = leer("regional", "shap_tipo_tumoral.json")
    r = sh["resultados"]
    modelos = ("logistica", "bosque_aleatorio")
    assert sh["n_glioma"] == 11 and sh["n_meningioma"] == 14
    assert max(v["error_aditividad_maximo"] for v in r.values()) < 1e-9
    for m in modelos:
        a, c, d = r[f"{m}/A_multicapa"], r[f"{m}/C_edad_sexo"], r[f"{m}/D_multicapa_residualizada"]
        assert a["p_permutacion"] < 0.05 and c["p_permutacion"] < 0.05 and d["p_permutacion"] > 0.05
        assert d["auc_media"] < c["auc_media"] < a["auc_media"] + 0.05 and d["p_permutacion"] > 0.05
        assert r[f"{m}/D_multicapa_residualizada"]["variables"][-1] == "ruido"
        assert r[f"{m}/A_multicapa"]["variables"] == ["log_cruce", "variacion_total", "log_P", "ruido"]
    demo = sh["demografia"]
    assert (demo["glioma"]["mujeres"], demo["meningioma"]["mujeres"]) == (4, 11)
    if not SIN_DATOS:
        import tumor as T
        p = T.participantes()
        sen = leer("controles", "sensibilidad.json")
        suj = leer("controles", "explicacion_espectral.json")["sujetos"]
        grupo = dict(zip(suj, sen["asociacion_cruce"]["grupo_por_sujeto"]))
        for tipo in ("glioma", "meningioma"):
            sexo = [p[x]["sex"] for x in suj if grupo[x] == tipo]
            assert sexo.count("F") == demo[tipo]["mujeres"]
    por_tipo = [c["rho"] for t in sh["correlacion_con_edad_por_tipo"].values() for c in t.values()]
    assert len(por_tipo) == 6 and max(map(abs, por_tipo)) < 0.4
    assert sh["pliegue_prueba"] == {"gliomas": [2, 3], "meningiomas": [2, 3]}
    for clave, v in r.items():
        a, b = v["auc_ic95_bootstrap"]
        assert a < v["auc_media"] < b, clave
        assert set(v["estabilidad_shap"]) == {"rho_media_entre_repeticiones", "variable_primera",
                                              "fraccion_repeticiones_misma_primera"}
        assert len(v["importancia_permutacion"]["caida_auc_media"]) == len(v["variables"])
    for m in modelos:
        for nombre in ("A_menos_C", "B_menos_C"):
            c = sh["comparaciones"][f"{m}/{nombre}"]
            assert c["ic95_bootstrap"][0] < c["diferencia"] < c["ic95_bootstrap"][1]
            assert 0 <= c["repeticiones_con_diferencia_positiva"] <= sh["particiones_comunes"]["repeticiones"]
        assert sh["comparaciones"][f"{m}/A_menos_C"]["ic95_bootstrap"][0] < 0
    residual = r["logistica/D_multicapa_residualizada"]["shap_abs_medio"]
    assert max(residual[:-1]) == residual[1] > residual[-1]
