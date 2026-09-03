# DIAO2A

Este proyecto usa **Python** y **uv**.

## Trabajos prácticos

- [TP1](TP1/)
- [TP2 - Algoritmos genéticos con triángulos](TP2/README.md)

`uv` prepara automáticamente Python y las librerías necesarias para el trabajo práctico. También crea un entorno separado para evitar conflictos con otros proyectos.

## 1. Instalar uv

Esto se hace una sola vez en cada computadora.

### Windows

Abrir **PowerShell** y ejecutar:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Cerrar y volver a abrir PowerShell. Comprobar que funciona:

```powershell
uv --version
```

### macOS o Linux

Abrir una terminal y ejecutar:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Cerrar y volver a abrir la terminal. Comprobar que funciona:

```bash
uv --version
```

## 2. Preparar el proyecto

Entrar desde la terminal en la carpeta del repositorio:

```powershell
cd ruta/a/DIAO2A
```

Por ejemplo, en Windows:

```powershell
cd C:\Users\mi_usuario\Documents\DIAO2A
```

Después ejecutar:

```powershell
uv sync
```

`uv sync` instala la versión correcta de Python y todas las librerías del proyecto. También crea la carpeta `.venv`, que contiene el entorno virtual.

La primera ejecución puede tardar un poco. Luego será mucho más rápida.

## 3. Usar el proyecto

La forma más sencilla es ejecutar todo con `uv run`. No hace falta activar el entorno.

```powershell
# Ver la versión de Python
uv run python --version

# Ejecutar un archivo
uv run python mi_archivo.py

# Abrir Python de forma interactiva
uv run python
```

Todos estos comandos deben ejecutarse desde la carpeta `DIAO2A`.

### Abrir los notebooks

Los archivos `.ipynb` se pueden abrir con Visual Studio Code y la extensión **Jupyter**.

Cuando Visual Studio Code pida elegir un intérprete o kernel, seleccionar el Python de `.venv`:

- Windows: `.venv\Scripts\python.exe`
- macOS/Linux: `.venv/bin/python`

También se puede abrir Jupyter Lab desde la terminal:

```powershell
uv run --with jupyter jupyter lab
```

Para detenerlo, volver a la terminal y presionar `Ctrl + C`.

## Activar y desactivar el entorno (opcional)

`uv` no se activa ni se desactiva. Lo que se activa es el entorno virtual `.venv`.

Activarlo permite escribir `python` directamente en lugar de `uv run python`, pero no es obligatorio.

### Activar en Windows PowerShell

```powershell
.\.venv\Scripts\Activate.ps1
```

### Activar en Windows CMD

```bat
.venv\Scripts\activate.bat
```

### Activar en macOS o Linux

```bash
source .venv/bin/activate
```

Cuando está activo, normalmente aparece `(.venv)` al comienzo de la terminal.

Para desactivarlo:

```powershell
deactivate
```

Cerrar la terminal también lo desactiva. Esto no borra nada.

## Agregar o quitar una librería

Para agregar una librería al trabajo práctico:

```powershell
uv add nombre-de-la-libreria
```

Por ejemplo:

```powershell
uv add requests
```

Para quitarla:

```powershell
uv remove requests
```

No usar `pip install` para agregar librerías al proyecto. `uv add` permite que todos los integrantes del grupo reciban las mismas dependencias.

Después de agregar o quitar una librería, se deben subir a Git los cambios de `pyproject.toml` y `uv.lock`.

## Después de descargar cambios del equipo

Si otro integrante agregó o cambió librerías, ejecutar:

```powershell
uv sync
```

Así el entorno local queda actualizado.

## Archivos importantes

- `.python-version`: versión de Python usada por el proyecto.
- `pyproject.toml`: lista de librerías necesarias.
- `uv.lock`: versiones exactas de esas librerías. No se edita a mano.
- `.venv/`: entorno local creado por `uv`. No se sube a Git.

Para este proyecto no hace falta ejecutar `pip install -r requirements.txt`; usar `uv sync`.

## Problemas frecuentes

### `uv` no se reconoce o aparece `command not found`

Cerrar y volver a abrir la terminal. Después probar nuevamente:

```powershell
uv --version
```

Si todavía no funciona, repetir la instalación o consultar la [guía oficial de uv](https://docs.astral.sh/uv/getting-started/installation/).

### PowerShell no permite activar el entorno

No es necesario cambiar la configuración de PowerShell. Usar el comando sin activación:

```powershell
uv run python mi_archivo.py
```

### Python indica que falta una librería

Desde la carpeta del proyecto, ejecutar:

```powershell
uv sync
```

Luego ejecutar el código con `uv run` o seleccionar el intérprete de `.venv` en Visual Studio Code.

## Resumen rápido

Después de instalar `uv`, normalmente solo hace falta:

```powershell
cd ruta/a/DIAO2A
uv sync
uv run python
```

Más información en la [documentación oficial de uv](https://docs.astral.sh/uv/).
