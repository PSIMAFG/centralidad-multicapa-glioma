"""Pruebas de los métodos nuevos: muestreadores de árboles, lector NIfTI, desempaquetado y nulos espaciales.

Se corren con: python3 -m pytest -q pruebas
"""

import gzip
import io
import itertools
import json
import os
import random
import struct
import sys
import tarfile
from collections import Counter

import numpy as np
import pytest

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import rutas
sys.path.insert(0, rutas.MODULOS)
sys.path.insert(0, os.path.join(BASE, "controles"))
sys.path.insert(0, os.path.join(BASE, "regional"))

import jerarquia as J
import nulo_grados_distancia as N
from controles import arbol_a_prufer, prufer_a_arbol
from desempaquetar import desempaquetar
from lectura import leer_nifti


def prufer_a_aristas(sec, n):
    grado = [1] * n
    for x in sec:
        grado[x] += 1
    aristas = []
    for x in sec:
        hoja = min(i for i in range(n) if grado[i] == 1)
        aristas.append(tuple(sorted((hoja, x))))
        grado[hoja] -= 1
        grado[x] -= 1
    u, v = [i for i in range(n) if grado[i] == 1]
    aristas.append((u, v))
    return frozenset(aristas)


def arboles_con_grados(n, grados):
    """Todos los árboles etiquetados de n nodos con la secuencia de grados dada (vía Prüfer)."""
    salida = set()
    for sec in itertools.product(range(n), repeat=n - 2):
        g = [1 + sec.count(i) for i in range(n)]
        if g == list(grados):
            salida.add(prufer_a_aristas(sec, n))
    return sorted(salida, key=sorted)


def test_cadena_markov_alcanza_la_distribucion_de_boltzmann():
    """En un caso pequeño enumerable, la cadena muestrea exp(-beta * longitud) entre árboles con los grados fijos."""
    n, grados = 7, (3, 2, 2, 1, 1, 1, 2)
    todos = arboles_con_grados(n, grados)
    assert len(todos) > 20
    rng = np.random.default_rng(1)
    xyz = rng.normal(size=(n, 3)) * 30
    d = np.linalg.norm(xyz[:, None] - xyz[None], axis=2).tolist()
    beta = 0.03
    peso = np.array([np.exp(-beta * sum(d[u][v] for u, v in t)) for t in todos])
    exacta = peso / peso.sum()
    cad = N.Cadena(sorted(todos[0]), n, d, [], random.Random(3))
    cuenta = Counter()
    for t in range(400000):
        cad.paso(beta)
        if t >= 5000 and t % 5 == 0:
            cuenta[frozenset(cad.e)] += 1
    total = sum(cuenta.values())
    empirica = np.array([cuenta[t] / total for t in todos])
    assert set(cuenta) <= set(todos)
    assert 0.5 * np.abs(empirica - exacta).sum() < 0.03


def test_prufer_permutada_es_uniforme_entre_arboles_con_los_mismos_grados():
    n, grados = 7, (3, 2, 2, 1, 1, 1, 2)
    todos = arboles_con_grados(n, grados)
    a = np.zeros((n, n))
    for u, v in todos[0]:
        a[u, v] = a[v, u] = 1
    sec, nodos = arbol_a_prufer(a)
    rng = np.random.default_rng(2)
    cuenta = Counter()
    for _ in range(60000):
        b = prufer_a_arbol(rng.permutation(sec), nodos, n)
        i, j = np.nonzero(np.triu(b, 1))
        cuenta[frozenset(zip(i.tolist(), j.tolist()))] += 1
    frec = np.array([cuenta[t] for t in todos]) / 60000
    assert set(cuenta) <= set(todos)
    assert 0.5 * np.abs(frec - 1 / len(todos)).sum() < 0.03


@pytest.mark.skipif(not os.path.isfile(os.path.join(rutas.RAIZ, "participants.tsv")),
                    reason="faltan los datos de OpenNeuro ds001226 (ver «Datos» en el README)")
def test_cadena_conserva_grados_y_conexidad_en_un_sujeto_real():
    import cargar as C
    import multiplex as M
    sc, fc = C.cargar_sujeto(N.RAIZ, "sub-PAT26")
    a = M.arbol_maximo(sc)
    xyz = np.random.default_rng(0).normal(size=(68, 3)) * 40
    d = np.linalg.norm(xyz[:, None] - xyz[None], axis=2).tolist()
    cad = N.Cadena(N.aristas_de(a), 68, d, [], random.Random(5))
    for _ in range(20000):
        cad.paso(0.02)
    assert cad.aceptadas > 100
    assert N.comprobar_arbol(cad.e, a.sum(1), 68)


def escribir_nifti(ruta, datos, afin, usar_sform=True):
    cab = bytearray(352)
    struct.pack_into("<i", cab, 0, 348)
    dim = [3] + list(datos.shape) + [1] * (7 - datos.ndim)
    struct.pack_into("<8h", cab, 40, *dim)
    struct.pack_into("<h", cab, 70, 16)
    struct.pack_into("<h", cab, 72, 32)
    struct.pack_into("<8f", cab, 76, 1.0, 1.0, 1.0, 1.0, 0, 0, 0, 0)
    struct.pack_into("<f", cab, 108, 352.0)
    struct.pack_into("<2f", cab, 112, 1.0, 0.0)
    struct.pack_into("<2h", cab, 252, 0 if usar_sform else 1, 1 if usar_sform else 0)
    if not usar_sform:
        struct.pack_into("<3f", cab, 256, 0.0, 0.0, 0.0)
        struct.pack_into("<3f", cab, 268, *afin[:3, 3])
    for k in range(3):
        struct.pack_into("<4f", cab, 280 + 16 * k, *afin[k])
    cab[344:348] = b"n+1\x00"
    with open(ruta, "wb") as fh:
        fh.write(bytes(cab) + datos.astype("<f4").tobytes(order="F"))


@pytest.mark.parametrize("usar_sform", [True, False])
def test_lector_nifti_recupera_datos_y_afin(tmp_path, usar_sform):
    datos = np.arange(4 * 5 * 6, dtype=float).reshape(4, 5, 6)
    afin = np.array([[-1.0, 0, 0, 10], [0, 1.0, 0, -20], [0, 0, 1.0, 30], [0, 0, 0, 1]])
    if not usar_sform:
        afin[0, 0] = 1.0
    ruta = str(tmp_path / "x.nii")
    escribir_nifti(ruta, datos, afin, usar_sform)
    vol, a, _, info = leer_nifti(ruta)
    assert np.array_equal(vol, datos)
    assert np.allclose(a, afin)
    with open(ruta, "rb") as fh, gzip.open(ruta + ".gz", "wb") as gz:
        gz.write(fh.read())
    vol2, a2, _, _ = leer_nifti(ruta + ".gz")
    assert np.array_equal(vol2, datos) and np.allclose(a2, afin)


def test_desempaquetar_ida_y_vuelta_y_rechazo_de_rutas(tmp_path):
    original = {"ds/a/uno.bin": os.urandom(5000), "ds/b/dos.txt": b"hola" * 100}
    ruta_tar = str(tmp_path / "x.tar")
    manif = [{"ruta": k, "bytes": len(v)} for k, v in original.items()]
    with tarfile.open(ruta_tar, "w") as t:
        for k, v in original.items():
            z = gzip.compress(v)
            info = tarfile.TarInfo(k + ".gz")
            info.size = len(z)
            t.addfile(info, io.BytesIO(z))
        m = json.dumps(manif).encode()
        info = tarfile.TarInfo("manifiesto.json")
        info.size = len(m)
        t.addfile(info, io.BytesIO(m))
    destino = tmp_path / "salida"
    destino.mkdir()
    n, errores = desempaquetar(ruta_tar, str(destino))
    assert n == 2 and not errores
    for k, v in original.items():
        assert (destino / k).read_bytes() == v
    malo = str(tmp_path / "malo.tar")
    with tarfile.open(malo, "w") as t:
        info = tarfile.TarInfo("../fuera.txt")
        info.size = 3
        t.addfile(info, io.BytesIO(b"abc"))
    with pytest.raises(ValueError):
        desempaquetar(malo, str(destino))


def test_ruido_tiene_la_desviacion_declarada_por_arista():
    import multiplex as M
    rng = np.random.default_rng(3)
    w = np.abs(rng.normal(1.0, 0.2, (68, 68)))
    w = (w + w.T) / 2.0
    np.fill_diagonal(w, 0.0)
    i, j = np.triu_indices(68, 1)
    relativos = []
    for k in range(100):
        p = M.perturbar(w, 0.1, np.random.default_rng(k))
        assert np.allclose(p, p.T) and np.all(np.diag(p) == 0)
        relativos.append(p[i, j] / w[i, j] - 1.0)
    assert np.std(np.concatenate(relativos)) == pytest.approx(0.1, rel=0.02)


def test_matriz_funcional_se_elige_por_nombre_y_falla_si_falta():
    import cargar as C
    m68, m113 = np.eye(68), np.eye(113)
    nombre, m = C.elegir_funcional([("FC_cc", m113), ("FC_cc_DK68_deconv", m68), ("FC_cc_DK68", m68)], 68)
    assert nombre == "FC_cc_DK68"
    with pytest.raises(ValueError):
        C.elegir_funcional([("FC_cc", m113), ("FC_cc_DK68_deconv", m68)], 68)


def esfera_por_regiones(rng, regiones=8, puntos=3000):
    """Dos hemisferios sintéticos: el derecho es el izquierdo reflejado, con un casquete de pared medial."""
    p = rng.normal(size=(puntos, 3))
    p /= np.linalg.norm(p, axis=1, keepdims=True)
    semillas = p[rng.choice(puntos, regiones, replace=False)]
    region = np.argmax(p @ semillas.T, axis=1)
    region[p[:, 0] > 0.9] = -1
    centros = np.array([p[region == k].mean(0) / np.linalg.norm(p[region == k].mean(0)) for k in range(regiones)])
    espejo = np.array([-1.0, 1.0, 1.0])
    vertices = {0: (p, region), 1: (p * espejo, np.where(region >= 0, region + regiones, -1))}
    return np.vstack([centros, centros * espejo]), np.repeat([0, 1], regiones), vertices


def test_rotaciones_de_la_esfera_reasignan_dentro_de_cada_hemisferio():
    rng = np.random.default_rng(7)
    centros, hemi, vertices = esfera_por_regiones(rng)
    n = len(hemi)
    hungaro, conteos = J.rotar_esfera(centros, hemi, vertices, [np.eye(3)])
    tamanos = np.bincount(np.concatenate([vertices[h][1][vertices[h][1] >= 0] for h in (0, 1)]), minlength=n)
    assert np.array_equal(hungaro[0], np.arange(n))
    assert np.array_equal(conteos[0], np.diag(tamanos))
    mapa = rng.normal(size=n)
    for nulo in J.nulos_rotados(mapa, hungaro, conteos).values():
        assert np.allclose(nulo[0], mapa)
    hungaro, conteos = J.rotar_esfera(centros, hemi, vertices, [J.rotacion_aleatoria(rng) for _ in range(40)])
    k = n // 2
    for perm, c in zip(hungaro, conteos):
        assert np.array_equal(np.sort(perm), np.arange(n)) and np.array_equal(hemi[perm], hemi)
        assert np.array_equal(perm[k:], perm[:k] + k)
        assert not c[:k, k:].any() and not c[k:, :k].any()
        assert np.all(c.sum(1) <= tamanos)
    nulos = J.nulos_rotados(mapa, hungaro, conteos)
    finitos = np.isfinite(nulos["baum"])
    assert np.all(np.isin(nulos["baum"][finitos], mapa))
    assert np.all(nulos["cornblath"][np.isfinite(nulos["cornblath"])] <= mapa.max() + 1e-12)
    assert np.array_equal(np.isfinite(nulos["baum"]), np.isfinite(nulos["cornblath"]))


def test_spearman_por_filas_y_holm():
    from scipy.stats import spearmanr
    rng = np.random.default_rng(11)
    ref = rng.normal(size=30)
    filas = rng.normal(size=(5, 30))
    filas[2, 4] = np.nan
    r = J.spearman_filas(filas, ref)
    for i in range(5):
        k = np.isfinite(filas[i])
        assert r[i] == pytest.approx(spearmanr(filas[i, k], ref[k])[0])
    assert np.allclose(J.holm([0.01, 0.04, 0.03, 0.005]), [0.03, 0.06, 0.06, 0.02])


def test_sustitutos_conservan_valores_y_autocorrelacion():
    rng = np.random.default_rng(5)
    xyz = rng.uniform(-1, 1, size=(80, 3))
    dist = np.sqrt(((xyz[:, None] - xyz[None]) ** 2).sum(-1))
    mapa = xyz[:, 0] + 0.1 * rng.normal(size=80)
    vecinos = np.argsort(dist, axis=1)[:, 1:6]

    def autocorrelacion(x):
        return np.corrcoef(x, x[vecinos].mean(1))[0, 1]

    sustitutos = J.sustitutos_variograma(mapa, dist, 40, rng)
    assert all(np.array_equal(np.sort(s), np.sort(mapa)) for s in sustitutos)
    propia = np.mean([autocorrelacion(s) for s in sustitutos])
    permutada = np.mean([autocorrelacion(rng.permutation(mapa)) for _ in range(40)])
    assert propia > 0.5 and abs(permutada) < 0.2


def test_shapley_exacto_en_un_modelo_lineal():
    import shap_tipo_tumoral as S
    rng = np.random.default_rng(3)
    fondo = rng.normal(size=(12, 4))
    x = rng.normal(size=(5, 4))
    w, b = np.array([0.5, -1.0, 2.0, 0.0]), 0.3
    phi, base, completo = S.shapley_exacto(lambda a: a @ w + b, x, fondo)
    assert np.allclose(phi, (x - fondo.mean(0)) * w)
    assert np.allclose(base + phi.sum(1), completo)


def test_residualizar_ajusta_solo_con_entrenamiento():
    import shap_tipo_tumoral as S
    rng = np.random.default_rng(4)
    z = rng.normal(size=(30, 2))
    x = np.column_stack([z @ [1.5, -0.5] + rng.normal(scale=0.1, size=30), rng.normal(size=30)])
    ent, pru = np.arange(20), np.arange(20, 30)
    x_ent, x_pru = S.residualizar(x[ent], x[pru], z[ent], z[pru], [0])
    assert abs(np.corrcoef(x_ent[:, 0], z[ent, 0])[0, 1]) < 1e-8
    assert np.allclose(x_ent[:, 1], x[ent, 1]) and np.allclose(x_pru[:, 1], x[pru, 1])
    assert np.std(x_pru[:, 0]) < 0.5
