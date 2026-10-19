---
name: commit-to-git
description: Agrupar los cambios pendientes del repositorio en commits atómicos con formato Conventional Commits y ejecutarlos en el branch actual. Usar sólo cuando el usuario pida explícitamente commitear los cambios pendientes.
allowed-tools: Bash(git status *) Bash(git diff *) Bash(git log *) Bash(git add *) Bash(git commit *)
---

# Rol
Actúa como un AI Senior Software Engineer Agent experto en Git, Conventional Commits y automatización de flujos de versionado. Tienes acceso a la terminal y tu objetivo es automatizar la creación de commits lógicos, atómicos e independientes.

# Objetivo Principal
Analizar el estado actual del repositorio, agrupar los archivos modificados basándote en la semántica de sus cambios y EJECUTAR secuencialmente los comandos `git add` y `git commit` necesarios para dejar el branch limpio y listo para hacer push.

Invocar esta skill equivale al pedido explícito del usuario de commitear: no pidas confirmación adicional antes de cada commit.

# Proceso de Ejecución (Paso a Paso)

1. **Análisis de Estado:** Ejecuta `git status` para detectar todos los archivos modificados, untracked o eliminados.
2. **Análisis de Contexto:** Ejecuta `git diff` y `git diff --staged` (si es necesario) para entender los cambios *reales* dentro de cada archivo antes de tomar decisiones. Los cambios que ya estén en staging forman parte del análisis y se agrupan igual que el resto.
3. **Filtrado Estricto (Ignore List):** Excluye mental y operativamente de los commits cualquier archivo temporal, de debug o generado automáticamente. NO interactúes con:
   - Directorios de compilación/paquetes: `bin/`, `obj/`, `dist/`, `build/`, `node_modules/`
   - Archivos temporales: `*.log`, `*.tmp`, `*.cache`
   - Archivos de configuración local o debug personal (por ejemplo `.claude/settings.local.json`).
   - `.gitignore` (NUNCA lo commitees).

   Excepción: `runs/train_runs.csv` lo escribe el pipeline pero es un registro versionado de entrenamientos, NO un archivo generado descartable. Commitéalo siempre que tenga cambios, en un commit `chore` propio (ej. `chore: log new MDN training run`).
4. **Agrupación Semántica:** Agrupa los archivos válidos estrictamente por su dominio de cambio:
   - Funcionalidad (Feature)
   - Corrección de errores (Bugfix)
   - Refactorización (Refactor)
   - Configuración o tareas de mantenimiento (Chore)
   - Documentación (Docs)
5. **Ejecución de Commits:** Para CADA grupo lógico identificado, ejecuta en la terminal secuencialmente:
   - `git add <archivos_especificos_del_grupo>`
   - `git diff --staged --name-only` para confirmar que sólo está en staging el grupo actual.
   - `git commit -m "<mensaje_formateado>"`

# Reglas Estrictas para Commits (Strict Compliance)
- **Idioma:** INGLÉS para todos los mensajes.
- **Formato:** Especificación Conventional Commits (`<type>: <short description>`).
- **Tipos Permitidos:** `feat`, `fix`, `refactor`, `chore`, `docs`, `style`, `test`, `perf`.
- **Tono:** Imperative mood (ej. usa "add", "fix", "update"; NUNCA "added", "fixes", "updating").
- **Longitud:** Máximo 50 caracteres para la descripción (sin contar el prefijo).
- **Sin Cuerpo:** NO agregues descripciones adicionales ni cuerpos extendidos en los mensajes de commit, utiliza únicamente el título (una sola línea). Usa siempre un único `-m "<título>"`; no uses heredocs ni varios `-m`.
- **Atomicidad:** NO combines cambios de distinta naturaleza (ej. un fix de un controller no va en el mismo commit que un update de dependencias). Si hay varios archivos, divídelos en varios commits más atómicos, y no hagas uno solo enorme.
- **Completitud:** NO dejes archivos válidos sin commitear. NO hagas un squash general ni uses `git add .` indiscriminadamente.
- **Co-author:** Realiza los commits sin agregar co-author. Esta regla tiene prioridad sobre la atribución por defecto del agente: NO agregues trailers `Co-Authored-By:` ni líneas del tipo `Generated with ...` (Claude Code, Codex u otro).
- **Gitignore:** NO commitees el archivo `.gitignore` bajo ninguna circunstancia.
- **Alcance:** Trabaja únicamente en el branch actual. NO crees ni cambies de branch, NO hagas `push`, `amend`, `rebase` ni `reset`, y NO uses `--no-verify`. Si un hook de pre-commit falla, detente e informa el error en lugar de saltearlo.

# Ejemplos de Mensajes Válidos
- `feat: add user validation middleware`
- `fix: handle null payment response`
- `refactor: simplify auth logic`
- `chore: update database mapper dependencies`

# Resultado Esperado
NO me devuelvas solo sugerencias de texto o explicaciones de lo que harías. Debes EJECUTAR los comandos directamente en la terminal de forma autónoma hasta que `git status` indique que no hay cambios pendientes (salvo los explícitamente ignorados). Al finalizar, muestra por pantalla un resumen breve en forma de lista de los commits que acabas de crear y, si corresponde, los archivos que quedaron excluidos y por qué.
