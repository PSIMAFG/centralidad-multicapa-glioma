"""¿El espejo izquierda-derecha de las máscaras es sistemático y está declarado en su cabecera?

Compara, para los 25 pacientes, la matriz afín de cada máscara con la del T1 del mismo paciente
(cabeceras leídas de OpenNeuro, cabeceras_t1_openneuro.json). Si la máscara es el T1 con el eje i
invertido, su afín debe ser la del T1 con la primera columna cambiada de signo y el origen
desplazado al otro extremo: A_mascara = A_T1 · F, con F la inversión de i en una grilla de N_i
vóxeles. En ese caso, pasar por coordenadas de mundo con la afín de cada archivo ubica la máscara
correctamente, como se comprobó a la vista en sub-PAT01 y sub-PAT26.
"""

import json
import os
import sys

import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
from lectura import leer_nifti, ruta_mascara

# Las matrices del T1 se guardaron con tres decimales: un error de redondeo de 0,0005 en un término
# fuera de la diagonal, multiplicado por hasta 255 vóxeles, desplaza el origen hasta unos 0,13 mm.
TOLERANCIA = 0.15


def main():
    cab = json.load(open(os.path.join(AQUI, "cabeceras_t1_openneuro.json"), encoding="utf-8"))["t1"]
    filas = []
    for nombre, ni, srow in cab:
        s = "sub-" + nombre
        vol, am, _, info = leer_nifti(ruta_mascara(s))
        at = np.vstack([np.array(srow).reshape(3, 4), [0, 0, 0, 1]])
        f = np.eye(4)
        f[0, 0] = -1.0
        f[0, 3] = ni - 1
        esperada = at @ f
        filas.append({"sujeto": s, "misma_grilla": int(vol.shape[0]) == int(ni),
                      "max_dif_con_t1_invertido_mm": float(np.abs(am - esperada).max()),
                      "max_dif_con_t1_sin_invertir_mm": float(np.abs(am - at).max())})
    res = {
        "pacientes": len(filas),
        "misma_grilla": int(sum(f["misma_grilla"] for f in filas)),
        "coinciden_con_t1_invertido": int(sum(f["max_dif_con_t1_invertido_mm"] < TOLERANCIA for f in filas)),
        "tolerancia_mm": TOLERANCIA,
        "max_dif_con_t1_invertido_mm": float(max(f["max_dif_con_t1_invertido_mm"] for f in filas)),
        "min_dif_con_t1_sin_invertir_mm": float(min(f["max_dif_con_t1_sin_invertir_mm"] for f in filas)),
        "por_paciente": filas,
    }
    with open(os.path.join(AQUI, "espejo_mascaras.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "por_paciente"}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
