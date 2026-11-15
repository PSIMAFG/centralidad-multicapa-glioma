"""Desempaqueta los archivos descargados de OpenNeuro (ds001226 y ds002080).

Los .tar armados en el navegador traen cada archivo original comprimido por separado con gzip
y un manifiesto (manifiesto.json o manifiesto_fc.json) con su tamaño original. Este script
los extrae en una carpeta de destino con la estructura de OpenNeuro, los descomprime y
comprueba los tamaños. El .tar de los T1 no tiene manifiesto: sus .nii.gz se copian tal cual.

Uso (Windows o Linux, solo biblioteca estándar):

    python desempaquetar.py CARPETA_DESTINO btc_mascaras_y_sc.tar btc_fc_postop.tar btc_t1_pat01_pat26.tar
"""

import gzip
import json
import os
import shutil
import sys
import tarfile


def ruta_segura(destino, nombre):
    final = os.path.normpath(os.path.join(destino, nombre))
    if os.path.commonpath([os.path.abspath(destino), os.path.abspath(final)]) != os.path.abspath(destino):
        raise ValueError("ruta fuera de la carpeta de destino: %s" % nombre)
    return final


def desempaquetar(ruta_tar, destino):
    with tarfile.open(ruta_tar) as t:
        miembros = [m for m in t.getmembers() if m.isfile()]
        manifiesto = {}
        for m in miembros:
            if os.path.basename(m.name).startswith("manifiesto"):
                for fila in json.load(t.extractfile(m)):
                    manifiesto[fila["ruta"]] = fila["bytes"]
        escritos, errores = 0, []
        for m in miembros:
            if os.path.basename(m.name).startswith("manifiesto"):
                continue
            comprimido_aqui = m.name.endswith(".gz") and m.name[:-3] in manifiesto
            nombre = m.name[:-3] if comprimido_aqui else m.name
            final = ruta_segura(destino, nombre)
            os.makedirs(os.path.dirname(final), exist_ok=True)
            fuente = t.extractfile(m)
            with open(final, "wb") as fh:
                if comprimido_aqui:
                    with gzip.GzipFile(fileobj=fuente) as gz:
                        shutil.copyfileobj(gz, fh)
                else:
                    shutil.copyfileobj(fuente, fh)
            if comprimido_aqui and os.path.getsize(final) != manifiesto[nombre]:
                errores.append(nombre)
            escritos += 1
    return escritos, errores


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    destino = sys.argv[1]
    os.makedirs(destino, exist_ok=True)
    for ruta in sys.argv[2:]:
        n, errores = desempaquetar(ruta, destino)
        estado = "sin errores" if not errores else "tamaño distinto en %d archivos: %s" % (len(errores), errores[:5])
        print("%s: %d archivos, %s" % (os.path.basename(ruta), n, estado))


if __name__ == "__main__":
    main()
