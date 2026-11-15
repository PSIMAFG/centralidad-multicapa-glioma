"""Lectura de los datos descargados de OpenNeuro (ds001226 y ds002080) sin dependencias externas.

Incluye un lector mínimo de NIfTI-1 (cabecera de 348 bytes, .nii o .nii.gz) y lectores
de los centroides de SC.zip (formato TVB) y de las matrices funcionales del posoperatorio.
"""

import gzip
import io
import os
import struct
import sys
import zipfile

import numpy as np
import scipy.io as sio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import rutas

DATOS = rutas.DATOS
TIPOS_NIFTI = {2: "u1", 4: "i2", 8: "i4", 16: "f4", 64: "f8", 256: "i1", 512: "u2", 768: "u4"}


def leer_nifti(ruta):
    """Devuelve (volumen, afín vóxel→mundo, tamaño de vóxel, info) de un NIfTI-1 (.nii o .nii.gz)."""
    with (gzip.open(ruta, "rb") if ruta.endswith(".gz") else open(ruta, "rb")) as fh:
        crudo = fh.read()
    orden = "<" if struct.unpack("<i", crudo[:4])[0] == 348 else ">"
    dim = struct.unpack(orden + "8h", crudo[40:56])
    tipo = struct.unpack(orden + "h", crudo[70:72])[0]
    pixdim = struct.unpack(orden + "8f", crudo[76:108])
    desplazamiento = int(struct.unpack(orden + "f", crudo[108:112])[0])
    pendiente, intercepto = struct.unpack(orden + "2f", crudo[112:120])
    qform, sform = struct.unpack(orden + "2h", crudo[252:256])
    forma = tuple(int(x) for x in dim[1:1 + dim[0]])
    n = int(np.prod(forma))
    dt = np.dtype(orden + TIPOS_NIFTI[tipo])
    vol = np.frombuffer(crudo, dtype=dt, count=n, offset=desplazamiento).reshape(forma, order="F")
    if pendiente not in (0.0, 1.0) or intercepto != 0.0:
        vol = vol * pendiente + intercepto
    if sform > 0:
        filas = [struct.unpack(orden + "4f", crudo[280 + 16 * k:296 + 16 * k]) for k in range(3)]
        afin = np.vstack([np.array(filas), [0, 0, 0, 1]])
    else:
        b, c, d = struct.unpack(orden + "3f", crudo[256:268])
        off = struct.unpack(orden + "3f", crudo[268:280])
        a = np.sqrt(max(0.0, 1.0 - (b * b + c * c + d * d)))
        r = np.array([[a * a + b * b - c * c - d * d, 2 * (b * c - a * d), 2 * (b * d + a * c)],
                      [2 * (b * c + a * d), a * a + c * c - b * b - d * d, 2 * (c * d - a * b)],
                      [2 * (b * d - a * c), 2 * (c * d + a * b), a * a + d * d - b * b - c * c]])
        qfac = -1.0 if pixdim[0] < 0 else 1.0
        escala = np.diag([pixdim[1], pixdim[2], pixdim[3] * qfac])
        afin = np.eye(4)
        afin[:3, :3] = r @ escala
        afin[:3, 3] = off
    info = {"qform_code": int(qform), "sform_code": int(sform), "tipo": dt.str}
    return vol, afin, tuple(float(x) for x in pixdim[1:4]), info


def ruta_mascara(sujeto):
    return os.path.join(DATOS, "ds001226", "derivatives", "tumor_masks", sujeto, "anat",
                        f"{sujeto}_space_T1_label-tumor.nii")


def ruta_sc(sujeto, sesion="preop"):
    ds = "ds001226" if sesion == "preop" else "ds002080"
    return os.path.join(DATOS, ds, "derivatives", "TVB", sujeto, f"ses-{sesion}", "SC.zip")


def leer_centros(sujeto, sesion="preop"):
    """Nombres y coordenadas (mm) de las 68 regiones de centres.txt dentro de SC.zip."""
    with zipfile.ZipFile(ruta_sc(sujeto, sesion)) as z:
        texto = z.read("centres.txt").decode("utf-8")
    nombres, xyz = [], []
    for linea in texto.splitlines():
        p = linea.split()
        if len(p) == 4:
            nombres.append(p[0])
            xyz.append([float(v) for v in p[1:]])
    return nombres, np.array(xyz)


def leer_pesos_zip(sujeto, sesion="preop"):
    with zipfile.ZipFile(ruta_sc(sujeto, sesion)) as z:
        return np.loadtxt(io.StringIO(z.read("weights.txt").decode("utf-8")))


def leer_fc_postop(sujeto):
    ruta = os.path.join(DATOS, "ds002080", "derivatives", "TVB", sujeto, "ses-postop", "FC.mat")
    return sio.loadmat(ruta)


def leer_sc_postop(sujeto):
    ruta = os.path.join(DATOS, "ds002080", "derivatives", "TVB", sujeto, "ses-postop", "SCthrAn.mat")
    return sio.loadmat(ruta)["SCthrAn"]
