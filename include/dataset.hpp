#pragma once
#include "regression.hpp"
#include <cstdint>
#include <string>

namespace ml {
struct Dataset {
    Matrix x;
    Vector y;
    std::vector<std::string> featureNames;
    std::string targetName;
    std::vector<std::size_t> rowIds; // 1 = primera fila de datos, excluye cabecera.
};
struct Split {Dataset train, test;};
Dataset readCSV(const std::string& path);
Split trainTestSplit(const Dataset& data,double trainFraction=0.8,std::uint32_t seed=42);
} // namespace ml
