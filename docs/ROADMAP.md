# Roadmap del TFG

## Objetivo

Desarrollar un sistema de clasificacion de etapas del sueno entrenado de forma supervisada con accelerometria triaxial y PPG/BVP de wearables, usando anotaciones PSG como ground truth. El modelo candidato inicial sera una arquitectura temporal basada en CNN + Mamba o una variante comparable para secuencias largas.

## Principios de trabajo

- Avanzar codigo y memoria en paralelo.
- Versionar solo artefactos ligeros y reproducibles.
- Mantener datos, checkpoints y resultados pesados fuera de Git.
- Empezar con un pipeline simple y verificable antes de aumentar complejidad del modelo.
- Documentar decisiones metodologicas cuando se toman, no al final.

## Fase 0. Preparacion del repo

Estado: en curso.

Objetivo: dejar una estructura minima para trabajar con datos, notebooks, scripts, entorno Python y memoria.

Entregables:

- `.gitignore` robusto para datos, venv, caches y checkpoints.
- `README.md` con resumen del proyecto.
- Roadmap y backlog inicial.
- Entorno Python reproducible.

Memoria:

- Definir titulo provisional, alcance, motivacion y objetivos.
- Empezar el capitulo de introduccion con el problema de la clasificacion de sueno con wearables.

## Fase 1. Extraccion y normalizacion de datos

Estado: siguiente gran bloque.

Objetivo: extraer de cada base de datos solo lo necesario para el modelo:

- Accelerometria triaxial.
- PPG/BVP o heart-rate derivado si el dataset no contiene PPG crudo.
- Etiquetas PSG por epoca.
- Metadatos minimos por sujeto, noche, dispositivo y frecuencia de muestreo.

Datasets iniciales:

- DREAMT.
- psg-newcastle-db.
- odc-tbi.
- Motion and Heart Rate from Wearable Devices with PSG.

Entregables:

- Inventario de formatos por dataset.
- Extractores por dataset.
- Esquema comun de datos procesados.
- Validaciones de alineacion senal-etiqueta.
- Reporte de cobertura, clases y sujetos utiles.

Memoria:

- Seccion de datasets.
- Tabla comparativa de bases de datos.
- Criterios de inclusion y exclusion.
- Descripcion del preprocesamiento.

## Fase 2. Preprocesado comun y ventanas de entrenamiento

Estado: pendiente.

Objetivo: convertir datasets heterogeneos en tensores comparables para entrenamiento.

Decisiones a cerrar:

- Longitud de epoca: 30 s como referencia PSG estandar.
- Frecuencia objetivo para ACC y PPG.
- Normalizacion por sujeto, noche o dataset.
- Etiquetado 5 clases frente a 3 clases.
- Tratamiento de segmentos invalidos, artefactos y missing data.

Entregables:

- Pipeline reproducible de preprocesado.
- Dataset final en formato ligero para entrenamiento.
- Estadisticas de balance de clases.
- Split por sujetos, evitando fuga entre entrenamiento y validacion.

Memoria:

- Justificacion de ventanas.
- Justificacion de normalizacion.
- Discusion de fuga de datos y particionado por sujeto.

## Fase 3. Baselines

Estado: pendiente.

Objetivo: construir una referencia solida antes del modelo principal.

Baselines:

- Clasificador mayoritario.
- Modelo clasico con features agregadas por epoca.
- CNN temporal simple.
- CNN + recurrente o Transformer ligero si procede.

Entregables:

- Scripts de entrenamiento.
- Metricas por dataset y globales.
- Matrices de confusion.

Memoria:

- Seccion de metodologia experimental.
- Metricas y protocolos.
- Primeros resultados.

## Fase 4. Modelo principal

Estado: pendiente.

Objetivo: implementar y evaluar la arquitectura principal, probablemente CNN + Mamba.

Preguntas de diseno:

- Procesar ACC y PPG como canales conjuntos o ramas separadas.
- Contexto temporal: una epoca aislada frente a secuencias de epocas.
- Fusion temprana frente a fusion tardia.
- Manejo de desbalance de clases.

Entregables:

- Modelo entrenable.
- Ablaciones minimas.
- Comparacion contra baselines.

Memoria:

- Arquitectura.
- Hiperparametros.
- Resultados y analisis.

## Fase 5. Evaluacion final y cierre

Estado: pendiente.

Objetivo: cerrar el proyecto con resultados defendibles y una memoria coherente.

Entregables:

- Evaluacion final.
- Figuras y tablas finales.
- Limitaciones.
- Reproducibilidad.
- PDF final de la memoria.

Memoria:

- Resultados.
- Discusion.
- Conclusiones.
- Trabajo futuro.

## Ritmo recomendado

- Cada bloque tecnico debe terminar con una actualizacion de memoria.
- Cada dataset debe tener una ficha corta: formato, senales disponibles, labels, problemas y decisiones.
- Cada experimento debe registrar configuracion, split, metricas y commit asociado.
