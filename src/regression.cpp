#include "regression.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace ml {
std::size_t validateMatrix(const Matrix& x) {
    if (x.empty() || x.front().empty())
        throw std::invalid_argument("X debe tener filas y columnas");
    const auto p = x.front().size();
    for (const auto& row : x) {
        if (row.size() != p) throw std::invalid_argument("X irregular");
        for (double v : row)
            if (!std::isfinite(v)) throw std::invalid_argument("X no finita");
    }
    return p;
}

Vector leastSquaresQR(Matrix a, Vector b) {
    const std::size_t n = validateMatrix(a), m = a.size();
    if (m < n || b.size() != m)
        throw std::invalid_argument("Dimensiones incompatibles para QR");
    for (double v : b)
        if (!std::isfinite(v)) throw std::invalid_argument("y no finita");
    double norm = 0.0;
    for (const auto& row : a) for (double v : row) norm = std::hypot(norm, v);
    if (!std::isfinite(norm)) throw std::overflow_error("Norma fuera de rango");
    const double tolerance = 100.0 * std::numeric_limits<double>::epsilon()
        * static_cast<double>(std::max(m, n)) * std::max(1.0, norm);
    // Cada reflector transforma A y b; Q nunca se almacena explicitamente.
    for (std::size_t k = 0; k < n; ++k) {
        double length = 0.0;
        for (std::size_t i = k; i < m; ++i) length = std::hypot(length, a[i][k]);
        if (length <= tolerance)
            throw std::invalid_argument("Matriz sin rango completo: colinealidad");
        const double alpha = -std::copysign(length, a[k][k]);
        Vector v(m-k);
        for (std::size_t i=k; i<m; ++i) v[i-k]=a[i][k];
        v[0] -= alpha;
        double vnorm=0.0;
        for (double value:v) vnorm=std::hypot(vnorm,value);
        for (double& value:v) value/=vnorm;
        for (std::size_t j=k; j<n; ++j) {
            double dot=0.0;
            for (std::size_t i=k; i<m; ++i) dot+=v[i-k]*a[i][j];
            for (std::size_t i=k; i<m; ++i) a[i][j]-=2.0*v[i-k]*dot;
        }
        double dot=0.0;
        for (std::size_t i=k; i<m; ++i) dot+=v[i-k]*b[i];
        for (std::size_t i=k; i<m; ++i) b[i]-=2.0*v[i-k]*dot;
        a[k][k]=alpha;
        for (std::size_t i=k+1; i<m; ++i) a[i][k]=0.0;
    }
    Vector solution(n,0.0);
    for (std::size_t i=n; i-- > 0;) {
        double rhs=b[i];
        for (std::size_t j=i+1;j<n;++j) rhs-=a[i][j]*solution[j];
        solution[i]=rhs/a[i][i];
        if (!std::isfinite(solution[i])) throw std::overflow_error("QR fuera de rango");
    }
    return solution;
}

Metrics computeMetrics(const Vector& actual,const Vector& predicted) {
    if (actual.empty() || actual.size()!=predicted.size())
        throw std::invalid_argument("Respuestas vacias o incompatibles");
    long double total=0.0L, squared=0.0L, absolute=0.0L, centered=0.0L;
    for (std::size_t i=0;i<actual.size();++i) {
        if (!std::isfinite(actual[i]) || !std::isfinite(predicted[i]))
            throw std::invalid_argument("Respuestas no finitas");
        total+=actual[i];
        const long double e=static_cast<long double>(actual[i])-predicted[i];
        squared+=e*e; absolute+=std::abs(e);
    }
    const long double mean=total/static_cast<long double>(actual.size());
    for (double v:actual) {const long double d=v-mean;centered+=d*d;}
    const double mse=static_cast<double>(squared/actual.size());
    const double mae=static_cast<double>(absolute/actual.size());
    if (!std::isfinite(mse) || !std::isfinite(mae))
        throw std::overflow_error("Metricas fuera de rango");
    // Para y constante, R2 no esta definido (incluso con prediccion perfecta).
    const double r2=centered==0.0L ? std::numeric_limits<double>::quiet_NaN()
                                : static_cast<double>(1.0L-squared/centered);
    return {actual.size(),mse,std::sqrt(mse),mae,r2};
}

void LinearRegression::fit(const Matrix& x,const Vector& y) {
    const std::size_t p=validateMatrix(x), n=x.size();
    if (y.size()!=n) throw std::invalid_argument("X e y incompatibles");
    for (double v:y) if (!std::isfinite(v)) throw std::invalid_argument("y no finita");
    Vector newMeans(p,0.0), newScales(p,1.0), newWeights(p,0.0);
    std::vector<std::size_t> active;
    for (std::size_t j=0;j<p;++j) {
        // long double reduce cancelacion en las estadisticas auxiliares.
        long double sum=0.0L, variance=0.0L;
        bool constant=true;
        for (const auto& row:x) {sum+=row[j];constant=constant && row[j]==x[0][j];}
        newMeans[j]=static_cast<double>(sum/static_cast<long double>(n));
        for (const auto& row:x) {
            const long double d=static_cast<long double>(row[j])-newMeans[j];
            variance+=d*d;
        }
        if (!constant) {
            newScales[j]=static_cast<double>(std::sqrt(variance/n));
            if (!(newScales[j]>0.0) || !std::isfinite(newScales[j]))
                throw std::overflow_error("Escala fuera de rango");
            active.push_back(j);
        }
    }
    Matrix design(n,Vector(active.size()+1,1.0));
    for (std::size_t i=0;i<n;++i)
        for (std::size_t k=0;k<active.size();++k) {
            const auto j=active[k];
            design[i][k+1]=(x[i][j]-newMeans[j])/newScales[j];
        }
    const Vector beta=leastSquaresQR(std::move(design),y);
    double newBias=beta[0];
    for (std::size_t k=0;k<active.size();++k) {
        const auto j=active[k];
        newWeights[j]=beta[k+1]/newScales[j];
        newBias-=newWeights[j]*newMeans[j];
    }
    if (!std::isfinite(newBias)) throw std::overflow_error("Bias fuera de rango");
    for (double w:newWeights)
        if (!std::isfinite(w)) throw std::overflow_error("Peso fuera de rango");
    // Commit: si algo falla antes, el ajuste anterior permanece intacto.
    weights=std::move(newWeights);means=std::move(newMeans);scales=std::move(newScales);
    bias=newBias;trained=true;
}

Matrix LinearRegression::data_scaling(const Matrix& x) const {
    if (!trained) throw std::logic_error("Modelo sin entrenar");
    if (validateMatrix(x)!=weights.size()) throw std::invalid_argument("Numero de atributos incorrecto");
    Matrix result=x;
    for (auto& row:result) for (std::size_t j=0;j<row.size();++j) {
        row[j]=(row[j]-means[j])/scales[j];
        if (!std::isfinite(row[j])) throw std::overflow_error("Escalado fuera de rango");
    }
    return result;
}

Vector LinearRegression::predict(const Matrix& x) const {
    if (!trained) throw std::logic_error("Modelo sin entrenar");
    if (validateMatrix(x)!=weights.size()) throw std::invalid_argument("Numero de atributos incorrecto");
    Vector result;result.reserve(x.size());
    for (const auto& row:x) {
        long double value=bias;
        for (std::size_t j=0;j<weights.size();++j) value+=static_cast<long double>(weights[j])*row[j];
        const double prediction=static_cast<double>(value);
        if (!std::isfinite(prediction)) throw std::overflow_error("Prediccion fuera de rango");
        result.push_back(prediction);
    }
    return result;
}

Metrics LinearRegression::score(const Matrix& x,const Vector& y) const {
    return computeMetrics(y,predict(x));
}
const Vector& LinearRegression::getWeights() const noexcept {return weights;}
double LinearRegression::getBias() const noexcept {return bias;}
const Vector& LinearRegression::getMeans() const noexcept {return means;}
const Vector& LinearRegression::getScales() const noexcept {return scales;}
bool LinearRegression::isFitted() const noexcept {return trained;}
std::size_t LinearRegression::dynamicBytes() const noexcept {
    return (weights.size()+means.size()+scales.size())*sizeof(double);
}
std::size_t LinearRegression::allocatedBytes() const noexcept {
    return (weights.capacity()+means.capacity()+scales.capacity())*sizeof(double);
}
} // namespace ml
