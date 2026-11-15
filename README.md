# ¿Qué mide realmente una centralidad multicapa estructura-función?

Auditoría exploratoria dMRI + fMRI en 25 pacientes con tumor cerebral y 11 controles. Este repositorio reúne el código y los resultados del estudio, postulado al Workshop IA Interpretable en Neurooncología 2026 (UNAB – UC – FALP, Santiago, 10 de diciembre de 2026).

Autor: Matías Alonso Fernández Gómez · estudiante de pregrado de Ingeniería Civil Industrial, Universidad Andrés Bello, Chile.

## Pregunta

Cuando la conectividad estructural (dMRI) y la funcional (fMRI) se unen en una red multicapa con un acoplamiento ω entre capas (un parámetro matemático elegido, no una interacción fisiológica medida), la centralidad de autovector multicapa (versatilidad) se mueve entre dos límites conocidos: con acoplamiento débil domina la capa de mayor radio espectral y con acoplamiento fuerte converge a la centralidad de la suma de las capas. El trabajo mide dónde quedan conectomas reales entre esos límites y qué parte de ese comportamiento fijan las unidades, la topología elegida y la geometría, frente a lo que aportan los datos.

## Resultados principales

- Sin reescalar, el radio espectral de la fMRI es 27,8 veces el de la difusión y la centralidad ordena como la fMRI sola (ρ = 0,9999). Con las capas reescaladas, ordena como la suma (ρ = 0,996). La centralidad es el autovector principal de la supra-adyacencia, normalizado a suma 1 y sumado por región; la FC entra en valor absoluto porque el teorema de Perron-Frobenius, en que se apoya la centralidad de autovector, pide capas no negativas. Con solo las correlaciones positivas, el cruce mediano es 0,28 (0,29 con el valor absoluto).
- Con el acoplamiento adimensional ω̃ = ω / r̄ (r̄: media de los dos radios espectrales), los árboles de expansión máxima binarizados llegan a ρ = 0,95 con su suma en ω̃ = 0,29 (mediana), antes que árboles nulos con el mismo grado por región (0,50), uniformes (0,52), con la correspondencia entre capas barajada (0,62) o con el mismo grado y una distancia media similar (0,46; 32 de 36 sujetos; p = 6,0 × 10⁻⁹). Si además se fijan las aristas homotópicas del fMRI, ese nulo cruza en 0,37 y el real cruza antes en 30 de 36 sujetos. En logaritmo de la razón entre el cruce real y el nulo de cada sujeto, la mediana es −0,49 con distancias (IC 95 % bootstrap: −0,63 a −0,38) y −0,26 con las homotópicas fijas (−0,38 a −0,16): la ventaja se atenúa con cada restricción sin desaparecer, una atenuación descriptiva y no una estimación causal del papel de la homotopía. Frente a los nulos topológicos, la ventaja se mantiene con umbrales de 0,90 y 0,99 y con una medida continua sin umbral (área bajo 1 − ρ).
- Un descriptor perturbativo interno, *P*, se asocia con ese cruce (ρ = 0,77). Al condicionar en *P*, la diferencia de los árboles reales con los nulos se reduce y deja de ser concluyente (con distancias: p = 0,25), sin que eso demuestre equivalencia. *P* y el cruce dependen de las mismas matrices: su asociación no es una validación predictiva independiente, aunque tampoco es una identidad matemática.
- Los árboles dMRI y fMRI comparten 11,25 aristas en promedio. La mediana de las razones real/esperado es 4,9 con el mismo grado y 1,1 si además se aproxima la distancia media (9,7 aristas esperadas en promedio). El exceso medio por sujeto baja de +8,9 aristas a +1,6 (IC 95 %: 0,8 a 2,4) y a +1,4 con las homotópicas fijas. Gran parte de la concordancia entre capas es compatible con grado y cercanía. Ese nulo reproduce la distribución de longitudes del árbol dMRI, pero no la del fMRI, que tiene más aristas cortas y homotópicas; la diferencia residual requiere controles geométricos más completos.
- En 10 controles con dos sesiones, la reproducibilidad del cruce individual fue incierta (CCI = 0,54; IC 95 %: −0,45 a 0,87).
- Tumor: el cruce es menor en glioma que en meningioma (0,25 frente a 0,36; Mann-Whitney p = 0,02), pero la diferencia no es concluyente al ajustar por edad y sexo (ANCOVA de rangos, p = 0,18): los meningiomas tienen, en mediana, 16 años más, y 11 de 14 son mujeres (gliomas: 4 de 11). No se detectaron asociaciones concluyentes con el volumen tumoral (ρ = −0,32; p = 0,12), con la distancia a la lesión ni con el cambio de TR de la fMRI. Las 25 máscaras tumorales comparten la grilla del T1 con orientación reflejada: hay que leerlas con la matriz afín de cada archivo. Así se calcularon las distancias de cada región al tumor (en coordenadas de mundo); el lóbulo de la región más cercana al centroide coincide con el del registro en 19 de 24 pacientes evaluables, y en 22 de 24 si se miran las tres regiones más cercanas. Los 11 gliomas están registrados con grado histológico II–III, con una clasificación anterior a la OMS 2021 y sin datos moleculares, e incluyen un ependimoma.
- Exploratorio: el mapa de reordenamiento regional es estable entre mitades de la muestra (Spearman-Brown: 0,79), pero no se detectó asociación con la jerarquía unimodal–transmodal normativa del HCP (ρ = 0,06; rotaciones de la esfera: p = 0,67) ni con su acoplamiento SC–FC regional (ρ = 0,03), con ninguno de los cinco nulos. Con 68 regiones, los nulos espaciales exigen |ρ| de 0,28 a 0,44 para p < 0,05, y la propia referencia del HCP (acoplamiento frente a jerarquía, ρ = −0,36) queda en el límite (p = 0,07 a 0,14): es ausencia de evidencia, no evidencia de ausencia.
- Demostración SHAP (glioma frente a meningioma, 25 pacientes): con los tres rasgos multicapa (log del cruce, log P y variación total respecto de la suma), una regresión logística y un bosque aleatorio discriminan el tipo tumoral (AUC 0,79 y 0,83; IC 95 % por bootstrap de sujetos: 0,59 a 0,94 y 0,67 a 0,96; p por permutación = 0,007 y 0,020), sin superar claramente a edad y sexo solos (0,75 y 0,80). Sobre las mismas 30 particiones, la diferencia es de 0,04 en la logística (IC 95 %: −0,22 a 0,30) y de 0,05 en el bosque (−0,15 a 0,25). Al quitarles a los rasgos, dentro de cada pliegue y con parámetros estimados solo en entrenamiento, la parte lineal que explican edad y sexo, el AUC ya no supera con claridad al azar (0,54 y 0,68; p = 0,38 y 0,06); aun así, en la regresión logística SHAP sigue dando a la variación total más importancia que al ruido puro (0,16 frente a 0,07). Ese orden es estable entre particiones (ρ media entre repeticiones de 0,84; la variación total es la primera en las 10 repeticiones con SHAP), y la importancia por permutación coincide en el rasgo principal, pero su caída de AUC varía mucho entre repeticiones (percentiles 2,5 y 97,5: −0,10 y 0,27): el modelo usa el rasgo sin que el AUC se distinga del azar. En el bosque sin edad ni sexo, ningún rasgo se distingue del ruido, ni en SHAP (0,07 a 0,08 todos) ni por permutación, y el orden SHAP es inestable (ρ media de 0,21). La variable de ruido es un único sorteo y, por azar, se asocia algo con el tipo tumoral (AUC de 0,58 por sí sola): en el bosque recibe tanta importancia como los rasgos, un recordatorio de que SHAP también reparte importancia al azar. Cada pliegue de prueba tiene 2 o 3 gliomas y 2 o 3 meningiomas. Dentro de cada tipo de tumor, la correlación de los rasgos con la edad es débil (|ρ| ≤ 0,32): en esta muestra, tipo tumoral, edad y sexo no se pueden separar. SHAP explica qué usa el modelo, no por qué funciona: puede premiar un rasgo confundido con edad y sexo. El acoplamiento SC–FC global no entra en el conjunto, porque relaciona las dos matrices en vez de formar la red multicapa; incluirlo da la misma lectura (AUC 0,75 y 0,81; sin edad ni sexo, 0,52 y 0,66). Es una demostración, no una validación.

Es una auditoría metodológica exploratoria (n = 36, una cohorte, una parcelación, sin desenlace clínico ni validación externa), no la validación de un biomarcador. Otros límites: gliomas sin clasificación molecular, señal BOLD que en glioma puede reflejar desacople neurovascular y movimiento cefálico no controlado. La interpretabilidad es de dos tipos: matemática (descomposición espectral por capa, límites del acoplamiento y teoría de perturbaciones) y aplicada (la demostración SHAP).

## Contenido

| Ruta | Qué contiene |
|---|---|
| `rutas.py` | Todas las rutas de datos y módulos en un solo lugar (ver «Datos»). |
| `v2/` | `cargar.py` y `multiplex.py`: lectura de las matrices TVB y construcción de las capas y los árboles. |
| `datos.py` | Cifras principales: construcciones de las capas, semejanza con cuatro referencias y verificación de implementación. Escribe `resultados/`. |
| `controles/controles.py`, `permutacion_cruce.py`, `sensibilidad.py`, `curvas_adim.py`, `heterogeneidad.py`, `verificacion_independiente.py` | Acoplamiento adimensional, nulos topológicos, densidad, umbrales, ruido, ajuste por edad y sexo, y verificación con networkx. |
| `controles/explicacion_espectral.py` | 500 réplicas por sujeto y tipo de nulo, descriptor espectral *P*, ruido con 20 realizaciones y comparación binarizada frente a ponderada. |
| `controles/sensibilidad_umbral.py` | Cruce con umbrales de 0,90, 0,95 y 0,99 y área bajo 1 − ρ (medida sin umbral) frente a 100 nulos topológicos de cada tipo por sujeto, y localización del autovector (razón de participación inversa). |
| `controles/validacion_P.py`, `verificaciones_finales.py` | Calibración de *P* agrupada por sujeto, referencia lobar, desglose anatómico de las aristas compartidas y comprobación del primer cruce con 400 puntos. |
| `controles/nulo_espacial_plantilla.py`, `nulo_espacial_sujeto.py` | Nulo de distancias con centroides de plantilla (preliminar; requiere ENIGMA Toolbox) y con los centroides de cada sujeto. |
| `controles/nulo_grados_distancia.py` | Nulo que conserva el grado de cada región y aproxima la distancia euclídea media entre regiones conectadas: cadena de Markov con propuesta simétrica, β calibrado por Robbins-Monro y fijo durante el muestreo. En sub-PAT26 opera sobre las 67 regiones conectadas. También compara la distribución completa de longitudes y las categorías anatómicas de los árboles reales con las esperadas en el nulo. |
| `controles/mezcla_cadena.py` | Tamaño efectivo de muestra de esa cadena, con las coincidencias con el árbol real de la otra capa y, entre los 100 pares nulos de cada sujeto, con el cruce y *P*. |
| `controles/aristas_por_sujeto.py` | Aristas compartidas reales y esperadas bajo cada referencia, sujeto a sujeto. |
| `controles/efectos_bootstrap.py` | Tamaños de efecto pareados por sujeto frente a cada nulo: logaritmo de la razón entre el cruce real y el nulo (mediana entre sujetos) y aristas reales menos esperadas (media), con intervalos bootstrap que remuestrean sujetos. Resume también la localización del autovector (1/IPR), la concordancia de las 10 regiones más centrales y la censura de los cruces en la grilla de umbrales. |
| `regional/lectura.py` | Lector de NIfTI-1 (.nii y .nii.gz) sin dependencias externas y lectores de los derivados TVB. |
| `regional/verificar_datos.py`, `alineacion_t1.py`, `espejo_mascaras.py` | Verificación de los datos de OpenNeuro: orden de regiones, pesos, volumen de las máscaras, alineación con el T1 (visual en dos pacientes; por cabecera en los 25, con `cabeceras_t1_openneuro.json`) y región aislada de sub-PAT26. `alineacion_t1.py` dibuja además `fig/mascara_t1.png`, un corte axial de sub-PAT26 con la máscara leída de las dos formas. |
| `regional/test_retest.py` | Reproducibilidad entre sesiones en los 10 controles de ds002080. |
| `regional/tumor.py` | Tipo tumoral (glioma frente a meningioma, sin ajustar y con ANCOVA de rangos por edad y sexo, más permutación de Freedman-Lane), volumen tumoral y cambio de TR de la fMRI frente al cruce. |
| `regional/shap_tipo_tumoral.py` | Demostración SHAP: glioma frente a meningioma con regresión logística y bosque aleatorio, cuatro conjuntos de rasgos (multicapa: log del cruce, log P y variación total; multicapa + edad + sexo; edad + sexo; multicapa sin la parte lineal de edad y sexo, ajustada dentro de cada pliegue solo con el entrenamiento) más una variable de ruido puro, 30 repeticiones de validación cruzada estratificada de 5 pliegues, p por permutación de etiquetas que repite el procedimiento completo y valores de Shapley intervencionales exactos fuera de pliegue (enumeración de coaliciones, con la aditividad comprobada). Informa además el IC 95 % del AUC por bootstrap estratificado de sujetos, las diferencias de AUC entre conjuntos sobre las mismas particiones (con IC pareado por sujeto), las clases por pliegue de prueba, la estabilidad del orden SHAP entre repeticiones y la importancia por permutación. Estos cálculos usan generadores aparte, así que no cambian las demás cifras. Unos 8 minutos. |
| `regional/jerarquia.py` | Jerarquía unimodal–transmodal (gradiente principal de la FC del HCP), acoplamiento SC–FC regional con tres predictores de comunicación y contraste de los mapas del multicapa con cinco nulos: permutación simple, rotaciones de la esfera fsaverage5 con reasignación húngara (la primaria), de Baum y de Cornblath, y sustitutos de variograma. Requiere los datos del ENIGMA Toolbox y la esfera fsaverage5 (ver «Datos»). |
| `regional/efecto_local.py`, `analisis_regional.py` | Efecto local del tumor en la fuerza nodal y análisis regional exploratorio de la centralidad frente a la distancia a la lesión. |
| `regional/desempaquetar.py` | Desempaqueta los `.tar` con manifiesto usados por el autor para descargar los archivos; con una descarga directa de OpenNeuro no hace falta. |
| `pruebas/` | Pruebas automáticas (pytest): muestreadores de árboles contra la distribución exacta, desviación efectiva del ruido, selección de la matriz funcional, lector NIfTI, desempaquetado, rotaciones de la esfera y sustitutos de variograma, valores de Shapley exactos en un modelo lineal y ajuste por confusores dentro del pliegue, recálculo independiente de cifras y coherencia de los resultados guardados (efectos pareados, contrastes regionales, tipo tumoral y demostración SHAP). |
| `requirements.txt` | Versiones de las dependencias. |
| `resultados/`, `controles/*.json`, `controles/*.npz`, `regional/*.json`, `regional/*.npz`, `regional/alineacion_t1.png` | Resultados guardados de todos los análisis, con los que se comprueban las cifras del estudio, y la comprobación visual de alineación de las máscaras. |

## Datos

Todos los datos son públicos (CC0) y están en OpenNeuro:

- **BTC_preop, ds001226 v5.0.1** (Aerts y cols., 2018): `participants.tsv`; `derivatives/TVB/sub-*/ses-preop/` (`FC.mat`, `SCthrAn.mat` y `SC.zip`, que trae los centroides en `centres.txt`); `derivatives/tumor_masks/`; y los T1 de sub-PAT01 y sub-PAT26, que solo usa la comprobación visual de alineación.
- **BTC_postop, ds002080 v4.0.1**: `derivatives/TVB/sub-*/ses-postop/` (`FC.mat` y `SCthrAn.mat`) para la segunda sesión de los controles.

Por defecto, el código busca esta estructura dentro del repositorio:

```
datos/
  ds001226/
    participants.tsv
    derivatives/TVB/sub-*/ses-preop/FC.mat, SCthrAn.mat, SC.zip
    derivatives/tumor_masks/sub-*/anat/sub-*_space_T1_label-tumor.nii
    sub-PAT01/ses-preop/anat/sub-PAT01_ses-preop_T1w.nii.gz
    sub-PAT26/ses-preop/anat/sub-PAT26_ses-preop_T1w.nii.gz
  ds002080/
    derivatives/TVB/sub-*/ses-postop/FC.mat, SCthrAn.mat
```

El análisis de jerarquía (`regional/jerarquia.py`) usa además dos recursos públicos: las matrices grupales del Human Connectome Project y la parcelación Desikan-Killiany en fsaverage5 que distribuye el ENIGMA Toolbox (carpeta `enigmatoolbox/datasets` de github.com/MICA-MNI/ENIGMA), y la esfera fsaverage5 de FreeSurfer que trae nilearn (`nilearn/datasets/data/fsaverage5`). Sus ubicaciones se indican con `ENIGMA_DATOS` y, si no se instala nilearn, con `FSAVERAGE5_DATOS`.

Para usar otra ubicación, se definen las variables de entorno `RAIZ_DATOS` (carpeta de ds001226), `DATOS_OPENNEURO` (carpeta que contiene `ds001226/` y `ds002080/`), `DATOS_T1` y `MODULOS_ANALISIS`. Los detalles están en `rutas.py`.

**Advertencia sobre las máscaras tumorales:** comparten la grilla del T1, pero están guardadas en espejo izquierda-derecha y su cabecera lo declara. Hay que pasar siempre por coordenadas de mundo con la matriz afín de cada archivo; leídas vóxel a vóxel junto al T1, el tumor queda en el hemisferio contrario.

## Requisitos

Python 3.10 o superior. Las versiones con que se obtuvieron los resultados están en `requirements.txt` (`pip install -r requirements.txt`): numpy, scipy, matplotlib, networkx, scikit-image, scikit-learn y pytest. h5py solo hace falta si algún `.mat` viene en formato v7.3, y nilearn, para la esfera fsaverage5 del análisis de jerarquía (o se indica su ubicación con `FSAVERAGE5_DATOS`).

La matriz funcional se lee siempre de la variable `FC_cc_DK68` de cada `FC.mat` (correlación en la parcelación Desikan-Killiany de 68 regiones); si falta, el código se detiene con un error. El ruido de las pruebas de estabilidad es multiplicativo y gaussiano, con un valor independiente por arista no dirigida.

## Cómo reproducir

Desde la carpeta del repositorio, en este orden. Las semillas están fijas, así que los resultados se repiten exactamente:

1. `python datos.py`
2. `python controles/controles.py`, `python controles/permutacion_cruce.py`, `python controles/sensibilidad.py`, `python controles/curvas_adim.py`, `python controles/heterogeneidad.py`, `python controles/verificacion_independiente.py`
3. `python controles/explicacion_espectral.py` (unos 10 minutos con dos núcleos), `python controles/validacion_P.py`, `python controles/verificaciones_finales.py`, `python controles/sensibilidad_umbral.py` (unos 5 minutos)
4. `python regional/verificar_datos.py`, `python regional/alineacion_t1.py`, `python regional/espejo_mascaras.py`, `python regional/efecto_local.py`
5. `python controles/nulo_espacial_sujeto.py`, `python controles/nulo_grados_distancia.py` (unos 10 minutos con dos núcleos), `python controles/mezcla_cadena.py`, `python controles/aristas_por_sujeto.py`, `python controles/efectos_bootstrap.py`
6. `python regional/test_retest.py`, `python regional/analisis_regional.py`, `python regional/tumor.py`, `python regional/shap_tipo_tumoral.py` (unos 8 minutos), `python regional/jerarquia.py` (unos 4 minutos; requiere los recursos del ENIGMA Toolbox y la esfera fsaverage5)
7. `python -m pytest -q pruebas`: son 24 pruebas y con los datos de OpenNeuro pasan todas. Sin esos datos pasan 18, y las 6 que los necesitan se omiten con un aviso.

Los scripts que usan varios núcleos leen la variable `PROCESOS` (por defecto, 2).

## Referencias

Datos: Aerts H, et al. Modeling brain dynamics in brain tumor patients using The Virtual Brain. eNeuro. 2018;5(3). Aerts H, Marinazzo D. BTC_preop (OpenNeuro ds001226 v5.0.1) y BTC_postop (OpenNeuro ds002080 v4.0.1).

Análisis de jerarquía: Larivière S, et al. The ENIGMA Toolbox: multiscale neural contextualization of multisite neuroimaging datasets. Nat Methods. 2021;18:698-700. Las matrices grupales provienen del Human Connectome Project, WU-Minn Consortium (investigadores principales: David Van Essen y Kamil Ugurbil; 1U54MH091657), financiado por los 16 institutos y centros de los NIH que apoyan el NIH Blueprint for Neuroscience Research y por el McDonnell Center for Systems Neuroscience de la Washington University. Contexto neurooncológico: Franco P, et al. MAGMA. 2026. doi:10.1007/s10334-026-01346-7; Montalba C, et al. ESMRMB 2026, resumen 54477.

Demostración SHAP: Lundberg SM, Lee SI. A unified approach to interpreting model predictions. NeurIPS. 2017; Snoek L, Miletić S, Scholte HS. How to control for confounds in decoding analyses of neuroimaging data. NeuroImage. 2019;184:741-760; Chyzhyk D, et al. How to remove or control confounds in predictive models, with applications to brain biomarkers. GigaScience. 2022;11:giac014; Jauny G, et al. Cortex. 2025;183:405-419.

Antecedentes y límites: Breedt LC, et al. The association between structure-function multilayer networks and executive functioning in glioma. Neuro-Oncology. 2025;27(Supl 3):iii72; Hernàndez Tey J, Cozzo E. Phys Rev E. 2026;114:014307; Martin T, Zhang X, Newman MEJ. Localization and centrality in networks. Phys Rev E. 2014;90:052808; Pak RW, et al. J Cereb Blood Flow Metab. 2017;37(11):3475-3487; Chabert S, et al. Hemodynamic response function description in patients with glioma. J Neuroradiol. 2024;51(4):101156; Louis DN, et al. The 2021 WHO Classification of Tumors of the Central Nervous System: a summary. Neuro Oncol. 2021;23(8):1231-1251. Integración SC–FC en glioma sin red multicapa: Colpo M, et al. Anomaly detection for structural and functional connectivity in glioma patients. NMR Biomed. 2026;39(4):e70238.

Métodos del análisis de jerarquía: Vázquez-Rodríguez B, et al. PNAS. 2019;116(42):21219-21227; Margulies DS, et al. PNAS. 2016;113(44):12574-12579; Alexander-Bloch AF, et al. NeuroImage. 2018;178:540-551; Markello RD, Mišić B. NeuroImage. 2021;236:118052; Burt JB, et al. NeuroImage. 2020;220:117038; Baum GL, et al. PNAS. 2020;117(1):771-778; Cornblath EJ, et al. Commun Biol. 2020;3:261.
