# Sleep classification with wearables

Trabajo Fin de Grado sobre clasificacion supervisada de etapas del sueno usando senales wearable como entrada y PSG como ground truth.

El objetivo del proyecto es entrenar y validar un modelo con accelerometria triaxial y PPG/BVP procedentes de dispositivos wearable. Las bases de datos iniciales son DREAMT, psg-newcastle-db, odc-tbi y Motion and Heart Rate from Wearable Devices with PSG de PhysioNet.

## Roadmap

- [docs/ROADMAP.md](docs/ROADMAP.md): fases del proyecto y relacion con la memoria.
- [docs/BACKLOG.md](docs/BACKLOG.md): lista de tareas tipo issue.
- [docs/DATA_PREPROCESSING.md](docs/DATA_PREPROCESSING.md): plan tecnico para extraccion y preprocesado.

## Working rules

- Los datos se guardan en `data/` y no se versionan.
- Los experimentos, checkpoints y salidas intermedias no se versionan.
- La memoria LaTeX se trabaja en local y solo se sube el PDF compilado en `doc/main.pdf`.

## Primera version sintética

Esta rama incorpora una primera version del pipeline centrada solo en DREAMT a nivel conceptual, pero usando datos sinteticos para validar el flujo completo antes del preprocesado real:

- `src/generate_synthetic_data.py`: genera un dataset tabular sintetico con epochs de 30 segundos.
- `src/train_random_forest.py`: entrena y evalua un `RandomForestClassifier` con separacion por sujeto.
- `src/explain_shap.py`: genera interpretabilidad global y por clase con SHAP.
- `src/run_synthetic_pipeline.py`: ejecuta el pipeline completo.

Dependencias:

```bash
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Ejecucion:

```bash
.\.venv\Scripts\python.exe -m src.run_synthetic_pipeline
```
