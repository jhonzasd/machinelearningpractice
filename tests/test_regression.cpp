#include "dataset.hpp"
#include "regression.hpp"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <set>
#include <stdexcept>

namespace {
int passed=0,failed=0;
void require(bool condition,const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}
void near(double actual,double expected,double tolerance=1e-9) {
    require(std::isfinite(actual) && std::abs(actual-expected)<=tolerance,"Valor numerico inesperado");
}
template<class Exception> void expectThrow(const std::function<void()>& action) {
    try {action();} catch(const Exception&) {return;}
    throw std::runtime_error("No se lanzo la excepcion esperada");
}
void test(const std::string& name,const std::function<void()>& action) {
    try {action();++passed;std::cout<<"PASS "<<name<<'\n';}
    catch(const std::exception& e) {++failed;std::cout<<"FAIL "<<name<<": "<<e.what()<<'\n';}
}
// Cada prueba de CSV usa un archivo temporal y lo retira incluso con excepcion.
void withCSV(const std::string& content,const std::function<void(const std::string&)>& action) {
    const auto path=std::filesystem::temp_directory_path()/"p2_ml_fixture.csv";
    {std::ofstream out(path);out<<content;if(!out) throw std::runtime_error("Fixture no escribible");}
    try {action(path.string());} catch (...) {std::filesystem::remove(path);throw;}
    std::filesystem::remove(path);
}
} // namespace

int main(int argc,char* argv[]) {
    const std::filesystem::path data=argc>1?argv[1]:"data";
    const ml::Matrix x={{0},{1},{2},{3}};const ml::Vector y={1,3,5,7};
    test("T01 recta exacta: y=1+2x",[&]{ml::LinearRegression m;m.fit(x,y);near(m.getBias(),1);near(m.getWeights()[0],2);near(m.predict({{4}})[0],9);near(m.score(x,y).r2,1);});
    test("T02 multiple exacta: y=1+2x1+3x2",[]{ml::LinearRegression m;m.fit({{0,0},{1,0},{0,1},{1,1}},{1,3,4,6});near(m.getBias(),1);near(m.getWeights()[0],2);near(m.getWeights()[1],3);near(m.predict({{2,3}})[0],14);});
    test("T03 escalado: media 0 y varianza poblacional 1",[&]{ml::LinearRegression m;m.fit(x,y);auto z=m.data_scaling(x);double sum=0,sq=0;for(auto row:z){sum+=row[0];sq+=row[0]*row[0];}near(sum/4,0);near(sq/4,1);near(m.getMeans()[0],1.5);near(m.getScales()[0],std::sqrt(1.25));});
    test("T04 no fuga: test x=100 no cambia media=1.5",[&]{ml::LinearRegression m;m.fit(x,y);near(m.data_scaling({{100}})[0][0],98.5/std::sqrt(1.25));near(m.getMeans()[0],1.5);});
    test("T05 metricas conocidas MSE=5/3 MAE=1 R2=-1.5",[]{auto r=ml::computeMetrics({1,2,3},{1,3,5});near(r.mse,5.0/3);near(r.rmse,std::sqrt(5.0/3));near(r.mae,1);near(r.r2,-1.5);});
    test("T06 atributo constante se absorbe en intercepto",[]{ml::LinearRegression m;m.fit({{0,7},{1,7},{2,7}},{1,3,5});near(m.getWeights()[1],0);near(m.getScales()[1],1);near(m.getBias(),1);near(m.predict({{4,7}})[0],9);});
    test("T07 todos los atributos constantes: predice media=4",[]{ml::LinearRegression m;m.fit({{7},{7},{7}},{2,4,6});near(m.predict({{7}})[0],4);near(m.getWeights()[0],0);});
    test("T08 objetivo constante: RMSE=0 y R2 indefinido",[&]{ml::LinearRegression m;m.fit(x,{5,5,5,5});auto r=m.score(x,{5,5,5,5});near(r.rmse,0);require(std::isnan(r.r2),"R2 no es NaN");});
    test("T09 predict antes de fit rechaza",[&]{ml::LinearRegression m;expectThrow<std::logic_error>([&]{m.predict(x);});});
    test("T10 escalado antes de fit rechaza",[&]{ml::LinearRegression m;expectThrow<std::logic_error>([&]{m.data_scaling(x);});});
    test("T11 X vacia rechaza",[]{ml::LinearRegression m;expectThrow<std::invalid_argument>([&]{m.fit({},{});});});
    test("T12 matriz irregular rechaza",[]{ml::LinearRegression m;expectThrow<std::invalid_argument>([&]{m.fit({{1},{2,3}},{1,2});});});
    test("T13 X e y incompatibles rechaza",[&]{ml::LinearRegression m;expectThrow<std::invalid_argument>([&]{m.fit(x,{1});});});
    test("T14 NaN en X rechaza",[]{ml::LinearRegression m;expectThrow<std::invalid_argument>([&]{m.fit({{0},{std::numeric_limits<double>::quiet_NaN()}},{0,1});});});
    test("T15 infinito en y rechaza",[&]{ml::LinearRegression m;expectThrow<std::invalid_argument>([&]{m.fit(x,{1,2,3,std::numeric_limits<double>::infinity()});});});
    test("T16 dimension incorrecta en predict rechaza",[&]{ml::LinearRegression m;m.fit(x,y);expectThrow<std::invalid_argument>([&]{m.predict({{1,2}});});});
    test("T17 colinealidad exacta rechaza",[]{ml::LinearRegression m;expectThrow<std::invalid_argument>([&]{m.fit({{0,0},{1,2},{2,4},{3,6}},{0,1,2,3});});});
    test("T18 sistema subdeterminado rechaza",[]{ml::LinearRegression m;expectThrow<std::invalid_argument>([&]{m.fit({{0,0},{1,2}},{1,2});});});
    test("T19 fit fallido conserva ajuste anterior",[&]{ml::LinearRegression m;m.fit(x,y);expectThrow<std::invalid_argument>([&]{m.fit({},{});});near(m.predict({{4}})[0],9);});
    test("T20 copiar modelo conserva independencia",[&]{ml::LinearRegression a;a.fit(x,y);auto b=a;b.fit(x,{0,3,6,9});near(a.predict({{4}})[0],9);near(b.predict({{4}})[0],12);require(a.getWeights().data()!=b.getWeights().data(),"Buffers compartidos");});
    test("T21 cambiar numero de atributos al reentrenar",[&]{ml::LinearRegression m;m.fit(x,y);m.fit({{0,0},{1,0},{0,1},{1,1}},{1,3,4,6});require(m.getWeights().size()==2,"Dimension anterior");near(m.dynamicBytes(),6*sizeof(double),0);});
    test("T22 CSV BOM CRLF y campos entre comillas",[]{withCSV("\xEF\xBB\xBF\"x\",\"y\"\r\n\"1.5\",4\r\n2,5\r\n",[](const auto& p){auto d=ml::readCSV(p);near(d.x[0][0],1.5);near(d.y[1],5);require(d.rowIds[1]==2,"Id incorrecto");});});
    test("T23 CSV celda vacia rechaza",[]{withCSV("x,y\n1,\n",[](const auto& p){expectThrow<std::invalid_argument>([&]{ml::readCSV(p);});});});
    test("T24 CSV texto numerico invalido rechaza",[]{withCSV("x,y\n1oops,2\n",[](const auto& p){expectThrow<std::invalid_argument>([&]{ml::readCSV(p);});});});
    test("T25 CSV cantidad de columnas rechaza",[]{withCSV("x,y\n1,2,3\n",[](const auto& p){expectThrow<std::invalid_argument>([&]{ml::readCSV(p);});});});
    test("T26 particion repetible completa y disjunta",[&]{auto d=ml::readCSV((data/"student_exam_scores.csv").string());auto a=ml::trainTestSplit(d);auto b=ml::trainTestSplit(d);require(a.train.rowIds==b.train.rowIds && a.test.rowIds==b.test.rowIds,"No reproducible");std::set<std::size_t> seen(a.train.rowIds.begin(),a.train.rowIds.end());for(auto id:a.test.rowIds)require(seen.insert(id).second,"Fuga o repetido");require(seen.size()==200 && a.train.x.size()==160 && a.test.x.size()==40,"Tamano incorrecto");});
    test("T27 fraccion invalida rechaza",[&]{auto d=ml::readCSV((data/"student_exam_scores.csv").string());expectThrow<std::invalid_argument>([&]{ml::trainTestSplit(d,1.0);});});
    test("T28 datasets reales 49x1 y 200x4",[&]{for(auto item:std::vector<std::pair<std::string,std::size_t>>{{"Ice_cream_selling_data.csv",49},{"student_exam_scores.csv",200}}){auto d=ml::readCSV((data/item.first).string());require(d.x.size()==item.second,"Filas incorrectas");require(d.x[0].size()==(item.second==49?1u:4u),"Columnas incorrectas");auto s=ml::trainTestSplit(d);ml::LinearRegression m;m.fit(s.train.x,s.train.y);require(std::isfinite(m.score(s.test.x,s.test.y).rmse),"Metrica no finita");}});
    test("T29 memoria profunda aumenta con atributos",[&]{ml::LinearRegression m;require(m.dynamicBytes()==0 && !m.isFitted(),"Estado inicial incorrecto");m.fit(x,y);require(m.dynamicBytes()==3*sizeof(double),"Payload incorrecto");require(m.allocatedBytes()>=m.dynamicBytes(),"Capacidad menor que size");});
    test("T30 QR independiente conocida",[]{auto a=ml::leastSquaresQR({{1,0},{1,1},{1,2}},{1,3,5});near(a[0],1);near(a[1],2);});
    test("T31 metricas tamanos incompatibles rechaza",[]{expectThrow<std::invalid_argument>([]{ml::computeMetrics({1,2},{1});});});
    test("T32 CSV comillas sin cerrar rechaza",[]{withCSV("x,y\n\"1,2\n",[](const auto& p){expectThrow<std::invalid_argument>([&]{ml::readCSV(p);});});});
    std::cout<<"SUMMARY passed="<<passed<<" failed="<<failed<<'\n';
    return failed==0?0:1;
}
