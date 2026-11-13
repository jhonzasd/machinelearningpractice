#include "dataset.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>

namespace ml {
namespace {
// Parser de campos simples o entre comillas; admite comillas dobles escapadas.
std::vector<std::string> parseCSVLine(const std::string& line) {
    std::vector<std::string> fields;std::string field;
    bool quoted=false,closed=false;
    for (std::size_t i=0;i<line.size();++i) {
        const char c=line[i];
        if (quoted) {
            if (c=='"') {
                if (i+1<line.size() && line[i+1]=='"') {field+='"';++i;}
                else {quoted=false;closed=true;}
            } else field+=c;
        } else if (c==',') {fields.push_back(field);field.clear();closed=false;}
        else if (c=='"' && field.empty() && !closed) quoted=true;
        else if (closed || c=='"') throw std::invalid_argument("CSV: comillas mal formadas");
        else field+=c;
    }
    if (quoted) throw std::invalid_argument("CSV: comillas sin cerrar");
    fields.push_back(field);return fields;
}

double parseNumber(const std::string& field) {
    std::istringstream stream(field);stream.imbue(std::locale::classic());
    double value=0.0;
    if (!(stream>>value)) throw std::invalid_argument("CSV: numero ausente o invalido");
    stream>>std::ws;
    if (!stream.eof() || !std::isfinite(value)) throw std::invalid_argument("CSV: numero invalido o no finito");
    return value;
}

// Rechazo evita el sesgo modular. mt19937 y este muestreo fijan la particion.
std::size_t boundedRandom(std::mt19937& rng,std::size_t bound) {
    const std::uint64_t range=std::uint64_t{1}<<32;
    if (bound==0 || bound>range) throw std::invalid_argument("Limite aleatorio invalido");
    const std::uint64_t limit=range-range%bound;
    std::uint64_t value;
    do {value=rng();} while (value>=limit);
    return static_cast<std::size_t>(value%bound);
}
} // namespace

Dataset readCSV(const std::string& path) {
    std::ifstream file(path);
    if (!file) throw std::runtime_error("No se pudo abrir: "+path);
    std::string line;
    if (!std::getline(file,line)) throw std::invalid_argument("CSV vacio");
    if (!line.empty() && line.back()=='\r') line.pop_back();
    if (line.compare(0,3,"\xEF\xBB\xBF")==0) line.erase(0,3);
    const auto headers=parseCSVLine(line);
    if (headers.size()<2) throw std::invalid_argument("CSV requiere X e y");
    for (std::size_t i=0;i<headers.size();++i) {
        if (headers[i].empty()) throw std::invalid_argument("Cabecera vacia");
        if (std::find(headers.begin(),headers.begin()+static_cast<std::ptrdiff_t>(i),headers[i])
            !=headers.begin()+static_cast<std::ptrdiff_t>(i)) throw std::invalid_argument("Cabecera duplicada");
    }
    Dataset data;data.featureNames.assign(headers.begin(),headers.end()-1);data.targetName=headers.back();
    std::size_t row=0;
    while (std::getline(file,line)) {
        ++row;
        if (!line.empty() && line.back()=='\r') line.pop_back();
        try {
            const auto fields=parseCSVLine(line);
            if (fields.size()!=headers.size()) throw std::invalid_argument("Numero de columnas incorrecto");
            Vector values;values.reserve(fields.size());
            for (const auto& field:fields) values.push_back(parseNumber(field));
            data.y.push_back(values.back());values.pop_back();data.x.push_back(std::move(values));
            data.rowIds.push_back(row);
        } catch (const std::exception& e) {
            throw std::invalid_argument("CSV fila de datos "+std::to_string(row)+": "+e.what());
        }
    }
    if (!file.eof()) throw std::runtime_error("Error de lectura CSV");
    validateMatrix(data.x);return data;
}

Split trainTestSplit(const Dataset& data,double fraction,std::uint32_t seed) {
    const auto p=validateMatrix(data.x),n=data.x.size();
    if (data.y.size()!=n || data.rowIds.size()!=n || data.featureNames.size()!=p)
        throw std::invalid_argument("Dataset incompatible");
    if (!std::isfinite(fraction) || fraction<=0.0 || fraction>=1.0)
        throw std::invalid_argument("Fraccion fuera de (0,1)");
    const auto cut=static_cast<std::size_t>(std::floor(static_cast<double>(n)*fraction));
    if (cut==0 || cut==n) throw std::invalid_argument("Particion vacia");
    std::vector<std::size_t> indices(n);std::iota(indices.begin(),indices.end(),0);
    std::mt19937 rng(seed);
    for (std::size_t i=n;i>1;--i) std::swap(indices[i-1],indices[boundedRandom(rng,i)]);
    Split result;
    for (Dataset* part:{&result.train,&result.test}) {
        part->featureNames=data.featureNames;part->targetName=data.targetName;
    }
    for (std::size_t i=0;i<n;++i) {
        Dataset& part=i<cut?result.train:result.test;
        part.x.push_back(data.x[indices[i]]);part.y.push_back(data.y[indices[i]]);
        part.rowIds.push_back(data.rowIds[indices[i]]);
    }
    return result;
}
} // namespace ml
