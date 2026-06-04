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
