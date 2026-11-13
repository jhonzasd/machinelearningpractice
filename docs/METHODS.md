# Documentación de métodos y funciones

n: filas; p: atributos; m: filas evaluadas; q: columnas del diseño. Se usa solo C++ estándar.

## Modelo · `LinearRegression()`

Sin parámetros → modelo sin ajustar.

Inicializa bias=0, trained=false y tres vectores vacíos. Copia, movimiento y destructor son los generados por C++, mediante regla de cero. O(1).

## Modelo · `fit(X, y)`

Matriz n×p y vector n → void; actualiza el modelo.

Valida, calcula escalado de entrenamiento, excluye columnas constantes, resuelve QR y recupera pesos originales. Rechaza entradas inválidas, rango deficiente, sistema subdeterminado o desbordamiento. Conserva el ajuste anterior si falla. Tiempo O(np²+p³), memoria O(np+p²), cota amplia.

## Modelo · `predict(X) const`

Matriz m×p → vector de m predicciones.

Aplica bias y pesos originales. Rechaza modelo no entrenado, matriz vacía, dimensión incorrecta o valores no finitos; detecta desbordamiento de salida. No modifica el modelo. O(mp), espacio O(m).

## Modelo · `score(X, y) const`

X de evaluación e y → Metrics.

Llama predict y computeMetrics. Retorna n, MSE, RMSE, MAE y R² en unidades originales. Propaga sus excepciones. O(mp), espacio O(m).

## Modelo · `data_scaling(X) const`

Matriz m×p → nueva matriz escalada.

Reutiliza medias y escalas del entrenamiento; no vuelve a ajustarlas. Rechaza estado no entrenado, dimensiones incorrectas, entradas no finitas y desbordamiento. O(mp) de tiempo y espacio.

## Modelo · `getWeights() const noexcept`

Sin parámetros → const Vector&.

Consulta pesos originales sin copiarlos; vector vacío antes de fit. La referencia no debe conservarse después de destruir el modelo, y su contenido puede cambiar tras fit. O(1).

## Modelo · `getBias() const noexcept`

Sin parámetros → double.

Consulta intercepto original; devuelve 0 antes de fit. No lanza excepciones. O(1).

## Modelo · `getMeans() const noexcept`

Sin parámetros → const Vector&.

Consulta medias de entrenamiento. Vacío antes de fit; referencia ligada a la vida del modelo. O(1).

## Modelo · `getScales() const noexcept`

Sin parámetros → const Vector&.

Consulta desviaciones poblacionales; columnas constantes usan escala 1. Vacío antes de fit. O(1).

## Modelo · `isFitted() const noexcept`

Sin parámetros → bool.

Indica si existe un ajuste exitoso. O(1).

## Modelo · `dynamicBytes() const noexcept`

Sin parámetros → size_t.

Suma size de weights, means y scales y multiplica por sizeof(double). Mide bytes de elementos vivos, no el objeto ni metadatos. O(1).

## Modelo · `allocatedBytes() const noexcept`

Sin parámetros → size_t.

Usa capacity de los tres vectores para estimar su almacenamiento reservado conocido. No incluye metadatos, redondeo del asignador ni el objeto. O(1).

## Cálculo · `validateMatrix(X)`

Matriz → número de columnas.

Rechaza matriz sin filas/columnas, filas de diferente longitud y NaN/infinito. No modifica X. O(np) de tiempo y O(1) adicional.

## Cálculo · `leastSquaresQR(A, b)`

Copias de matriz m×q y vector m → vector q.

Reflectores Householder y sustitución hacia atrás propios. Requiere m≥q y rango completo. Tolerancia: 100·epsilon·max(m,q)·max(1,norma de Frobenius). Rechaza dimensiones, no finitos, rango deficiente o desbordamiento. O(mq²+q²), espacio O(mq+m).

## Cálculo · `computeMetrics(actual, predicted)`

Dos vectores de igual longitud → Metrics.

Acumula errores y media con long double. Rechaza vectores vacíos, tamaños distintos, no finitos y métricas fuera de rango. R²=NaN si y es constante. O(n), espacio O(1).

## Datos · `parseCSVLine(line)`

Una línea → vector de campos de texto.

Función interna. Acepta campos simples, comillas y comillas dobles escapadas. Rechaza comillas mal formadas; no admite campos multilínea. O(L), espacio O(L), con L caracteres.

## Datos · `parseNumber(field)`

Texto de celda → double.

Función interna. Usa locale clásico y exige consumir todo el campo excepto espacios. Rechaza vacío, texto extra, infinito y NaN. O(L), espacio auxiliar O(L).

## Datos · `boundedRandom(rng, bound)`

Generador mt19937 y límite → índice [0,bound).

Función interna. Muestreo por rechazo para evitar sesgo modular. Requiere 1≤bound≤2³²; actualiza rng. Tiempo esperado O(1), espacio O(1).

## Datos · `readCSV(path)`

Ruta → Dataset.

Última columna como y, anteriores como X; conserva cabeceras y rowIds desde 1. Acepta BOM UTF-8 y CRLF. Rechaza apertura/lectura fallida, cabecera vacía o duplicada, datos ausentes/inválidos y anchos distintos. O(L), espacio O(np).

## Datos · `trainTestSplit(data, fraction, seed)`

Dataset, fracción 0.8 y semilla 42 por defecto → Split.

Valida dimensiones y fracción; Fisher-Yates propio con mt19937. floor(n·fraction) filas de entrenamiento. Rechaza particiones vacías. Copia filas e IDs sin solapamiento. O(np), espacio O(np).

## Ejecutable · `printMetrics(label, metrics, out)`

Etiqueta, Metrics y stream → void.

Escribe n y las cuatro métricas en una línea. O(1) en cantidad de valores; propaga errores del stream si están habilitados.

## Ejecutable · `outputFile(path)`

Ruta → ofstream movible.

Abre un archivo de salida, comprueba apertura y activa excepciones de escritura. Configura 15 dígitos significativos. Puede lanzar runtime_error o ios_base::failure; costo dependiente de E/S.

## Ejecutable · `runExperiment(name, input, output)`

Nombre, CSV y carpeta de salida → void.

Lee, divide, entrena, evalúa y exporta cuatro CSV: métricas, coeficientes, predicciones y escalado. Compara con media de y_train. Imprime todas las predicciones de prueba. Propaga errores de datos/modelo/E/S; costo dominado por fit.

## Ejecutable · `memoryAnalysis(output)`

Carpeta de salida → void y memory.txt.

Ajusta modelos global, local y dinámico con y=1+2x1+3x2. Crea vector dinámico de tres copias; registra direcciones, sizeof, capacidad y buffers. Gestión por unique_ptr/vector. Tamaño fijo en este experimento; propaga errores de fit y E/S.

## Ejecutable · `main(argc, argv)`

Argumentos data_dir y results_dir opcionales → int.

Por defecto usa data y results. Crea carpeta, ejecuta ambos datasets y memoria. Devuelve 0 al completar o 1 con mensaje ERROR si captura std::exception. Rechaza argumentos extra.

## Pruebas · `require(condition, message)`

Condición y mensaje → void.

Falla con runtime_error si la condición es falsa. O(1), aparte del mensaje.

## Pruebas · `near(actual, expected, tolerance)`

Valores y tolerancia absoluta 1e-9 por defecto → void.

Exige resultado finito y diferencia dentro de la tolerancia; usa require. O(1).

## Pruebas · `expectThrow<Exception>(action)`

Función sin argumentos → void.

Pasa solo si action lanza el tipo esperado; otro tipo se propaga y ausencia de excepción falla. Costo de action.

## Pruebas · `test(name, action)`

Nombre y función → void.

Ejecuta action, imprime PASS/FAIL y actualiza contadores; captura std::exception. Costo de action; ejecución secuencial.

## Pruebas · `withCSV(content, action)`

Texto y función que recibe ruta → void.

Crea fixture temporal p2_ml_fixture.csv, llama action y elimina el archivo en éxito o excepción. Pruebas secuenciales; no ejecutar dos instancias simultáneas con el mismo directorio temporal. O(L) más action.

## Pruebas · `main(argc, argv)`

Ruta de datos opcional → int.

Ejecuta T01 a T32, imprime resumen y retorna 0 si no hay fallos; en caso contrario, 1. Las acciones anónimas de cada caso se documentan en la matriz de pruebas.
