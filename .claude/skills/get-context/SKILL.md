---
name: get-context
description: Obtener y analizar el contexto técnico del proyecto Neural 3D CME Reconstruction antes de planificar, implementar, diagnosticar o revisar cambios en el repositorio.
---

# Obtener contexto del proyecto

Analiza este repositorio antes de responder la solicitud del usuario. Esta etapa es
de solo lectura, salvo que el usuario también haya pedido explícitamente realizar cambios.

## Contexto base

1. Lee `README.md` para conocer el objetivo científico, la estructura del proyecto,
   los comandos disponibles y los artefactos generados.
2. Lee `pyproject.toml` para identificar la versión de Python, dependencias,
   configuración de pytest y requisitos de cobertura.
3. Revisa `main.py` para entender la orquestación de entrenamiento, validación,
   inferencia, reportes y debug inference.
4. Ejecuta `git status --short --branch` para detectar la rama actual y cambios
   existentes que deban preservarse.
5. Busca instrucciones locales como `CLAUDE.md` y aplícalas si existen.

No inspecciones `.venv`, artefactos binarios ni directorios de resultados completos,
salvo que sean relevantes para la solicitud.

## Profundización según la tarea

Lee únicamente las áreas relacionadas con la solicitud:

- Configuración:
  `training/config/configuration.py` y los YAML pertinentes de `training/config/`.
- Modelos:
  `models/__init__.py`, `models/base_model.py` y el modelo concreto afectado.
- Datos y normalización:
  `training/dataloaders/`, incluyendo
  `training/dataloaders/normalization/NORMALIZATION.md` cuando corresponda.
- Entrenamiento:
  `training/train_model.py`, métricas, persistencia y seguimiento de ejecuciones.
- Validación:
  `training/validate.py`, `training/validation_artifacts.py` y los análisis
  relacionados.
- Geometría y GCS:
  `utils/gcs_headers.py`, `utils/gcs_masks.py`, `visualization/gcs.py` y las
  pruebas de geometría.
- Inferencia:
  `inference/run_inference.py` y los contratos relacionados con headers,
  selección MDN y overlays.
- Visualización y reportes:
  `visualization/` y `analysis/`.
- Logging:
  `logger/`.
- Comportamiento esperado:
  las pruebas de `tests/` que cubran el componente o contrato afectado.

Usa `rg` y `rg --files` para localizar definiciones, usos, configuraciones y pruebas.
Sigue imports y llamadas solamente hasta comprender el flujo relevante.

## Criterios del análisis

Antes de proponer o realizar cambios, identifica:

- el punto de entrada y el flujo afectado;
- las configuraciones y contratos involucrados;
- los artefactos de entrada y salida;
- las pruebas existentes que expresan el comportamiento esperado;
- compatibilidad hacia atrás relevante;
- cambios locales existentes que no deben sobrescribirse;
- incertidumbres o contradicciones entre documentación, implementación y pruebas.

Considera las pruebas como contratos ejecutables, pero comprueba también la
implementación: no asumas que el README o los archivos descriptivos están actualizados.

## Resultado

Resume de forma concisa:

- arquitectura y flujo relevantes para la solicitud;
- archivos principales involucrados;
- restricciones y contratos detectados;
- riesgos, inconsistencias o información faltante;
- comandos de verificación apropiados.

Después, continúa con la solicitud original del usuario respetando el alcance y
las autorizaciones indicadas.
