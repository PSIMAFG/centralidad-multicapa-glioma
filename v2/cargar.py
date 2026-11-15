"""Lectura de los derivados TVB de ds001226.

Los .mat de este conjunto pueden venir en formato v5 (scipy) o v7.3 (HDF5), y el
nombre de la variable que contiene la matriz no está documentado. Por eso aquí no
se asume ningún nombre: se abre el archivo, se recorren sus variables y se toma la
primera matriz cuadrada numérica de tamaño razonable.
"""

import csv
import os
import zipfile

import numpy as np
import scipy.io


def _candidatas(d):
    """Devuelve las matrices cuadradas numéricas de un diccionario de variables."""
    salida = []
    for clave, valor in d.items():
        if clave.startswith("__"):
            continue
        a = np.asarray(valor)
        if a.ndim == 2 and a.shape[0] == a.shape[1] and a.shape[0] >= 10:
            if np.issubdtype(a.dtype, np.number):
                salida.append((clave, a.astype(float)))
    return salida


def leer_mat_todas(ruta):
    """Devuelve TODAS las matrices cuadradas de un .mat v5 o v7.3, como lista."""
    try:
        d = scipy.io.loadmat(ruta)
        c = _candidatas(d)
        if c:
            return c
    except NotImplementedError:
        pass
    except Exception:
        pass

    import h5py

    with h5py.File(ruta, "r") as f:
        d = {}
        for clave in f.keys():
            try:
                d[clave] = np.array(f[clave])
            except Exception:
                continue
    # HDF5 guarda traspuesto respecto de MATLAB; para una matriz simétrica da igual.
    return _candidatas(d)


def leer_mat(ruta):
    """Lee un .mat y devuelve (nombre_variable, matriz) de la mayor cuadrada."""
    c = leer_mat_todas(ruta)
    if not c:
        raise ValueError("no se encontró ninguna matriz cuadrada en %s" % ruta)
    return max(c, key=lambda x: x[1].shape[0])


VARIABLE_FC = "FC_cc_DK68"


def elegir_funcional(candidatas, n):
    """Devuelve la matriz funcional que corresponde a la parcelación estructural.

    El FC.mat de ds001226 y ds002080 trae cuatro matrices: FC_cc y FC_mi, de 113 regiones, y
    FC_cc_DK68 y FC_cc_DK68_deconv, en la parcelación Desikan-Killiany de 68 regiones que comparte
    con SCthrAn. Se usa siempre la correlación cruda en Desikan-Killiany (FC_cc_DK68), sin
    heurísticas: si falta o no tiene el tamaño de la matriz estructural, se detiene con un error
    en vez de elegir otra matriz.
    """
    for k, m in candidatas:
        if k == VARIABLE_FC:
            if m.shape[0] != n:
                raise ValueError("%s tiene %d regiones y la estructural %d" % (VARIABLE_FC, m.shape[0], n))
            return k, m
    raise ValueError("FC.mat sin la variable %s (variables: %s)" % (VARIABLE_FC, ", ".join(k for k, _ in candidatas)))


def leer_sc_zip(ruta):
    """Lee weights.txt dentro de un SC.zip con formato de The Virtual Brain."""
    with zipfile.ZipFile(ruta) as z:
        nombres = [n for n in z.namelist() if n.lower().endswith("weights.txt")]
        if not nombres:
            raise ValueError("SC.zip sin weights.txt: %s" % ruta)
        with z.open(nombres[0]) as fh:
            texto = fh.read().decode("utf-8", errors="replace")
    filas = [l.split() for l in texto.strip().splitlines() if l.strip()]
    return np.array(filas, dtype=float)


def limpiar(m):
    """Deja la matriz simétrica, sin diagonal, sin negativos ni valores no finitos."""
    m = np.asarray(m, dtype=float).copy()
    m[~np.isfinite(m)] = 0.0
    m = np.abs(m)
    m = (m + m.T) / 2.0
    np.fill_diagonal(m, 0.0)
    return m


def sujetos(raiz):
    """Lista los sujetos presentes en 03_datos/derivatives/TVB, en orden."""
    base = os.path.join(raiz, "derivatives", "TVB")
    if not os.path.isdir(base):
        return []
    s = [d for d in sorted(os.listdir(base)) if d.startswith("sub-")]
    return s


def ruta_sujeto(raiz, sujeto, archivo):
    return os.path.join(raiz, "derivatives", "TVB", sujeto, "ses-preop", archivo)


def cargar_sujeto(raiz, sujeto, devolver_nombre=False):
    """Devuelve (estructural, funcional) ya limpias, o None si falta algún archivo."""
    fc = ruta_sujeto(raiz, sujeto, "FC.mat")
    sc_mat = ruta_sujeto(raiz, sujeto, "SCthrAn.mat")
    sc_zip = ruta_sujeto(raiz, sujeto, "SC.zip")

    if not os.path.exists(fc):
        return None

    if os.path.exists(sc_mat):
        _, msc = leer_mat(sc_mat)
    elif os.path.exists(sc_zip):
        msc = leer_sc_zip(sc_zip)
    else:
        return None

    n = msc.shape[0]
    nombre, mfc = elegir_funcional(leer_mat_todas(fc), n)
    if devolver_nombre:
        return limpiar(msc), limpiar(mfc), nombre
    return limpiar(msc), limpiar(mfc)


def grupos(raiz):
    """Clasifica cada sujeto en control, glioma o meningioma según participants.tsv."""
    ruta = os.path.join(raiz, "participants.tsv")
    if not os.path.exists(ruta):
        return {}
    salida = {}
    with open(ruta, encoding="utf-8", errors="replace") as fh:
        for fila in csv.DictReader(fh, delimiter="\t"):
            pid = (fila.get("participant_id") or "").strip()
            col = [c for c in fila if "tumor type" in c.lower()]
            tipo = (fila.get(col[0]) if col else "").strip().lower()
            if not pid:
                continue
            if tipo in ("none", "", "n/a"):
                salida[pid] = "control"
            elif "meningioma" in tipo:
                salida[pid] = "meningioma"
            else:
                salida[pid] = "glioma"
    return salida


def tabla(raiz, columnas=None):
    """Lee participants.tsv completo y devuelve {sujeto: {columna: valor}}.

    Los numéricos se convierten; el resto queda como texto limpio. La tabla usa
    n/a para los ausentes, que se omiten.
    """
    ruta = os.path.join(raiz, "participants.tsv")
    if not os.path.exists(ruta):
        return {}
    salida = {}
    with open(ruta, encoding="utf-8", errors="replace") as fh:
        lector = csv.DictReader(fh, delimiter="\t")
        campos = {(c or "").strip(): c for c in (lector.fieldnames or [])}
        pedidas = columnas if columnas else list(campos.keys())
        for fila in lector:
            pid = (fila.get(campos.get("participant_id", "participant_id")) or "").strip()
            if not pid:
                continue
            d = {}
            for nombre in pedidas:
                col = campos.get(nombre)
                if col is None:
                    continue
                crudo = (fila.get(col) or "").strip()
                if crudo in ("", "n/a", "N/A", "na"):
                    continue
                v = _num(crudo)
                d[nombre] = v if v is not None else crudo
            salida[pid] = d
    return salida


def _num(x):
    try:
        v = float(str(x).strip().replace(",", "."))
        return v if np.isfinite(v) else None
    except Exception:
        return None


# Medidas cognitivas preespecificadas de la batería CANTAB que trae la cohorte.
# signo = +1 cuando más alto es mejor rendimiento, -1 cuando más alto es peor.
COGNITIVAS = [
    ("RVP_A", +1, "Atención sostenida (sensibilidad A')"),
    ("SSP_spanlength", +1, "Memoria de trabajo espacial (amplitud)"),
    ("SOC_prob_minmoves", +1, "Planificación (problemas en el mínimo de movimientos)"),
    ("RTI_fiveRT", -1, "Tiempo de reacción de elección"),
    ("MOT_latency_mean", -1, "Latencia motora"),
]


def cognitivo(raiz):
    """Lee edad y las medidas cognitivas preespecificadas de participants.tsv.

    Devuelve {sujeto: {"edad": float, "<medida>": float, ...}} omitiendo los
    valores ausentes o no numéricos (la tabla usa n/a).
    """
    ruta = os.path.join(raiz, "participants.tsv")
    if not os.path.exists(ruta):
        return {}
    salida = {}
    with open(ruta, encoding="utf-8", errors="replace") as fh:
        lector = csv.DictReader(fh, delimiter="\t")
        campos = {c.strip(): c for c in (lector.fieldnames or [])}
        for fila in lector:
            pid = (fila.get("participant_id") or "").strip()
            if not pid:
                continue
            d = {}
            col_edad = campos.get("age")
            if col_edad:
                e = _num(fila.get(col_edad))
                if e is not None:
                    d["edad"] = e
            for nombre, _, _ in COGNITIVAS:
                col = campos.get(nombre)
                if col:
                    v = _num(fila.get(col))
                    if v is not None:
                        d[nombre] = v
            salida[pid] = d
    return salida
