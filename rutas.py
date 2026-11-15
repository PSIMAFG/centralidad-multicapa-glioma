"""Rutas de datos y módulos, en un solo lugar.

Cada ruta se toma de una variable de entorno o, si no está definida, de la primera ubicación
relativa a esta carpeta que exista. Por defecto se usan datos/ y v2/ dentro del repositorio; si
el paquete está en Poster_Multicapa_Glioma/05_poster/<carpeta>/, también se encuentra 03_datos.

Variables de entorno opcionales:
- RAIZ_DATOS: copia de OpenNeuro ds001226 con derivatives/TVB y participants.tsv.
- DATOS_OPENNEURO: carpeta con ds001226/ y ds002080/ (máscaras tumorales, SC.zip y sesión posoperatoria).
- DATOS_T1: carpeta con ds001226/sub-*/ses-preop/anat/ (T1 para la comprobación de alineación).
- MODULOS_ANALISIS: carpeta con cargar.py y multiplex.py.
- ENIGMA_DATOS: carpeta enigmatoolbox/datasets (nulo con centroides de plantilla y análisis de jerarquía).
- FSAVERAGE5_DATOS: carpeta con sphere_left.gii.gz y sphere_right.gii.gz de fsaverage5 (análisis de
  jerarquía). Si no está definida, se usa la copia que trae nilearn (nilearn/datasets/data/fsaverage5).
"""

import os

BASE = os.path.dirname(os.path.abspath(__file__))


def _fsaverage5():
    valor = os.environ.get("FSAVERAGE5_DATOS")
    if valor:
        return os.path.abspath(valor)
    try:
        import nilearn
    except ImportError:
        return os.path.join(BASE, "fsaverage5")
    return os.path.join(os.path.dirname(nilearn.__file__), "datasets", "data", "fsaverage5")


def _primera(variable, *candidatas):
    """Variable de entorno si está definida; si no, la primera candidata que exista (o la primera de la lista)."""
    valor = os.environ.get(variable)
    if valor:
        return os.path.abspath(valor)
    for c in candidatas:
        if os.path.isdir(c):
            return os.path.abspath(c)
    return os.path.abspath(candidatas[0])


RAIZ = _primera("RAIZ_DATOS", os.path.join(BASE, "datos", "ds001226"), os.path.join(BASE, "..", "..", "03_datos"))
DATOS = _primera("DATOS_OPENNEURO", os.path.join(BASE, "datos"),
                 os.path.join(RAIZ, "openneuro_2026-11-14", "crudo"))
DATOS_T1 = _primera("DATOS_T1", os.path.join(BASE, "datos"), os.path.join(RAIZ, "openneuro_2026-11-14", "t1"))
MODULOS = _primera("MODULOS_ANALISIS", os.path.join(BASE, "v2"))
ENIGMA = _primera("ENIGMA_DATOS", os.path.join(BASE, "enigmatoolbox", "datasets"))
FSAVERAGE5 = _fsaverage5()
