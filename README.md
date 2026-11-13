# Práctica 2 de programación orientada a objetos

Regresión lineal simple y múltiple en C++17. Incluye escalado, mínimos cuadrados por QR Householder, métricas, lectura de CSV, separación reproducible de datos, análisis de memoria y 32 pruebas. **El programa solo necesita la biblioteca estándar de C++**. No usa Eigen, Armadillo, OpenCV, Python, NumPy ni bibliotecas de regresión.

## Inicio rápido

Desde esta carpeta, con GCC y Make:

```bash
make
make test
make run
```

Con GCC sin Make (Linux, macOS o Windows con MinGW-w64):

```bash
g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic -Iinclude src/regression.cpp src/dataset.cpp src/main.cpp -o regression_demo
g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic -Iinclude src/regression.cpp src/dataset.cpp tests/test_regression.cpp -o regression_tests
./regression_tests data
./regression_demo data results
```

En PowerShell usa `./regression_tests.exe data` y `./regression_demo.exe data results`. En Windows, con GCC, añade `-finput-charset=UTF-8 -fexec-charset=UTF-8` si el compilador interpreta mal los comentarios o nombres de columnas. Abre la terminal en la carpeta que contiene este README.

También se incluye CMake (ruta alternativa, no ejecutada en el entorno de preparación):

```bash
cmake -S . -B build-cmake
cmake --build build-cmake --config Release
ctest --test-dir build-cmake -C Release --output-on-failure
```

Ejecuta luego `build-cmake/regression_demo data results` (generador monoconfiguración) o `build-cmake/Release/regression_demo.exe data results` (Visual Studio).

## Archivos

- `include/regression.hpp`, `src/regression.cpp`: modelo, QR, validaciones y métricas.
- `include/dataset.hpp`, `src/dataset.cpp`: CSV numérico y partición de datos.
- `src/main.cpp`: experimentos y memoria global, local y dinámica.
- `tests/test_regression.cpp`: 32 pruebas sin framework externo.
- `data/`: los dos archivos originales, sin modificación.
- `results/`: ejecución, pruebas, métricas, coeficientes, todas las predicciones, escalado y memoria.
- `docs/Report_document.docx`: informe editable con documentación completa.
- `docs/METHODS.md`: referencia rápida de cada función y método.
- `docs/VIDEO_GUIDE.md`: guía de sustentación de 9 minutos y 30 segundos.

## Decisiones del experimento

La última columna de cada CSV es la respuesta; todas las anteriores son atributos. Se usa una partición 80/20, semilla 42, Fisher-Yates con `mt19937` y selección uniforme mediante rechazo. La cantidad de entrenamiento se redondea hacia abajo: 39/10 para helados y 160/40 para estudiantes. Las filas exportadas conservan su `row_id` original (primera fila de datos = 1).

`fit` aprende medias y desviaciones poblacionales solo en entrenamiento. Las columnas constantes reciben escala 1 y peso 0. QR rechaza falta de rango entre columnas activas. Los pesos y el intercepto se convierten a las unidades originales. `data_scaling` reutiliza el escalador entrenado sin modificarlo. `score` retorna `Metrics {n,mse,rmse,mae,r2}`, no un único número. Para una respuesta constante, R² se devuelve como NaN porque no está definido.

## Resultados registrados

| Dataset | Entrenamiento / prueba | RMSE prueba | R² prueba |
|---|---:|---:|---:|
| Helados | 39 / 10 | 9.470006 | -0.105467 |
| Estudiantes | 160 / 40 | 2.335580 | 0.883097 |

El conjunto de helados presenta curvatura. Su resultado negativo se conserva: representa una limitación del modelo lineal solicitado. Los coeficientes son asociaciones condicionales; no prueban causalidad. Una sola partición no estima la incertidumbre de generalización.

## Verificación

Compilación ejecutada con GCC 14.2.0 en Linux de 64 bits, C++17 y advertencias activadas. Las 32 pruebas pasan. AddressSanitizer y UndefinedBehaviorSanitizer no reportaron errores en las pruebas ni en el ejecutable, con `detect_leaks=0`.

```bash
make sanitize ASAN_OPTIONS=detect_leaks=0
```

`make sanitize` activa detección de fugas por defecto. En el entorno de preparación, LeakSanitizer abortó porque no puede trabajar bajo `ptrace`; el registro original está en `results/sanitizers.txt`. Por ello **no se afirma haber validado ausencia de fugas con LeakSanitizer**. La liberación se diseña mediante RAII y regla de cero. Se incluye una comparación numérica independiente de desarrollo en `results/independent_validation.txt`: su herramienta externa no forma parte del programa ni se necesita para ejecutarlo.

Las direcciones de memoria cambian al repetir la ejecución. Los tamaños dependen del compilador y de la arquitectura. El total calculado excluye metadatos y redondeo del asignador. No es el RSS del proceso.

## Requisitos personales para entregar

Completar los nombres, identificaciones y correos institucionales de **ambos integrantes** en el informe. Subir este proyecto a un repositorio propio y reemplazar el campo del enlace. Grabar y publicar el video como **no listado**, con ambos rostros visibles, participación de ambos y duración máxima de 10 minutos; agregar su enlace al informe y a EAFIT Interactiva. El paquete no contiene un repositorio remoto publicado ni un video grabado. La fecha límite debe consultarse en EAFIT Interactiva.

## Límites del lector CSV

Se aceptan archivos numéricos delimitados por comas, punto decimal, cabecera, CRLF/LF, BOM UTF-8 y campos entre comillas. No se admiten campos multilínea, valores ausentes, líneas vacías, NaN, infinito ni columnas de texto. Los archivos adjuntos cumplen este formato. No se imputa ni se elimina ninguna fila de forma silenciosa.
