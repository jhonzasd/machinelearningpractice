# Guía del video de sustentación

Duración planeada: 9 minutos y 30 segundos. Grabar con ambos rostros visibles y compartir pantalla. Publicar como **no listado** en YouTube. Esta guía no reemplaza el video exigido.

| Tiempo | Participante | Contenido que debe demostrar |
|---|---|---|
| 0:00-0:40 | Ambos | Nombres completos, práctica y objetivo. |
| 0:40-1:30 | Integrante 1 | CSV, respuesta y atributos, 49 y 200 filas, partición 80/20. |
| 1:30-2:40 | Integrante 1 | Encapsulamiento, `weights`, `bias`, `fit` y `predict`. |
| 2:40-3:40 | Integrante 2 | Escalado solo en entrenamiento, QR propia y recuperación de coeficientes originales. |
| 3:40-4:40 | Integrante 2 | Compilar, ejecutar las 32 pruebas y mostrar resultados de ambos conjuntos. |
| 4:40-5:30 | Integrante 1 | Interpretar RMSE, MAE y R²; explicar la curvatura de helados y su R² negativo. |
| 5:30-6:50 | Integrante 2 | Mostrar direcciones, 88 bytes por objeto, buffers y 432 bytes conocidos del vector dinámico. |
| 6:50-8:10 | Ambos | Heartbleed, Cloudbleed y WannaCry: falla de memoria, consecuencias y corrección. |
| 8:10-9:00 | Ambos | Dificultades, soluciones, al menos cinco conclusiones y limitación de LeakSanitizer. |
| 9:00-9:30 | Ambos | Mostrar estructura del repositorio, informe y enlace de entrega. |

## Preguntas para preparar la sustentación

1. ¿Por qué `sizeof(modelo)` no incluye los elementos de sus vectores? Porque mide la representación del objeto, no los bloques separados que administra.
2. ¿Por qué el objeto global también tiene memoria dinámica? Sus miembros `vector` reservan buffers independientemente de la duración del objeto que los contiene.
3. ¿Por qué R² puede ser negativo? Porque el error cuadrático supera el de predecir la media del conjunto evaluado.
4. ¿Qué evita la fuga de información? Dividir antes de calcular medias y escalas; el conjunto de prueba no participa en `fit`.
5. ¿Por qué QR? Resuelve mínimos cuadrados sin invertir la matriz de ecuaciones normales, que puede agravar el condicionamiento.
6. ¿Qué sucede con atributos constantes o colineales? Constantes: peso cero y efecto en el intercepto; dependencia entre activos: excepción explícita.
7. ¿Por qué no se usa herencia? No hay múltiples familias de modelos ni polimorfismo requerido; encapsulamiento y composición resuelven el objetivo.
8. ¿Puede afirmarse que estudiar una hora causa 1.56 puntos más? No; ese coeficiente describe una asociación condicional en los datos y el modelo.
9. ¿Los 432 bytes son todo lo que consume el programa? No; son el tamaño conocido del vector, su reserva de objetos y los buffers internos de esos modelos, sin sobrecarga del asignador.
10. ¿Pasar las pruebas garantiza ausencia de errores? No; respalda los casos ejecutados y sus contratos, con las limitaciones descritas en el informe.

## Antes de enviar

- Revisar que ambos integrantes puedan explicar cualquier parte.
- Completar nombres, IDs y correos en el informe.
- Reemplazar los dos campos de enlaces: repositorio y video.
- Comprobar que el enlace de YouTube abre sin permisos privados.
- Entregar nombres completos, repositorio y video en EAFIT Interactiva dentro del plazo indicado allí.
