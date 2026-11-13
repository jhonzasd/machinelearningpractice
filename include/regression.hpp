#pragma once

#include <cstddef>
#include <vector>

namespace ml {
using Vector = std::vector<double>;
using Matrix = std::vector<Vector>;

// Todas las metricas se calculan en las unidades originales de la respuesta.
struct Metrics {
    std::size_t n;
    double mse, rmse, mae, r2;
};

// Rechaza matrices vacias, irregulares o no finitas. Retorna columnas.
std::size_t validateMatrix(const Matrix& x);
// Minimos cuadrados mediante QR Householder propia, sin invertir A^T A.
Vector leastSquaresQR(Matrix a, Vector b);
Metrics computeMetrics(const Vector& actual, const Vector& predicted);

class LinearRegression {
private:
    Vector weights;
    Vector means;
    Vector scales;
    double bias = 0.0;
    bool trained = false;

public:
    // Regla de cero: vector administra copia, movimiento y liberacion.
    LinearRegression() = default;
    void fit(const Matrix& x, const Vector& y);
    Vector predict(const Matrix& x) const;
    Metrics score(const Matrix& x, const Vector& y) const;
    // Usa SOLO las medias/escalas aprendidas en fit. No modifica el modelo.
    Matrix data_scaling(const Matrix& x) const;
    const Vector& getWeights() const noexcept;
    double getBias() const noexcept;
    const Vector& getMeans() const noexcept;
    const Vector& getScales() const noexcept;
    bool isFitted() const noexcept;
    // Payload conocido; no incluye metadatos del asignador ni redondeo.
    std::size_t dynamicBytes() const noexcept;
    std::size_t allocatedBytes() const noexcept;
};
} // namespace ml
