#include "dataset.hpp"
#include "regression.hpp"
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>

namespace fs=std::filesystem;
// Duracion estatica. Sus buffers vectoriales se asignan durante fit.
ml::LinearRegression globalModel;

namespace {
void printMetrics(const std::string& label,const ml::Metrics& m,std::ostream& out) {
    out<<label<<": n="<<m.n<<", MSE="<<m.mse<<", RMSE="<<m.rmse
       <<", MAE="<<m.mae<<", R2="<<m.r2<<'\n';
}

std::ofstream outputFile(const fs::path& path) {
    std::ofstream file(path);
    if (!file) throw std::runtime_error("No se pudo crear: "+path.string());
    file.exceptions(std::ios::badbit|std::ios::failbit);
    file<<std::setprecision(15);return file;
}

void runExperiment(const std::string& name,const fs::path& input,const fs::path& output) {
    const ml::Dataset data=ml::readCSV(input.string());
    const ml::Split split=ml::trainTestSplit(data,0.8,42);
    ml::LinearRegression model;model.fit(split.train.x,split.train.y);
    const auto trainMetrics=model.score(split.train.x,split.train.y);
    const auto testMetrics=model.score(split.test.x,split.test.y);
    const double trainMean=std::accumulate(split.train.y.begin(),split.train.y.end(),0.0)
                           /static_cast<double>(split.train.y.size());
    const auto baseline=ml::computeMetrics(split.test.y,ml::Vector(split.test.y.size(),trainMean));
    std::cout<<"\nDATASET "<<name<<"\nfile="<<input.string()<<" rows="<<data.x.size()
             <<" features="<<data.featureNames.size()<<" seed=42 train_fraction=0.8\n";
    std::cout<<"bias="<<model.getBias()<<"\nweights:";
    for (double w:model.getWeights()) std::cout<<' '<<w;
    std::cout<<'\n';
    printMetrics("train",trainMetrics,std::cout);printMetrics("test",testMetrics,std::cout);
    printMetrics("test_baseline_train_mean",baseline,std::cout);
    auto metrics=outputFile(output/(name+"_metrics.csv"));
    metrics<<"partition,n,mse,rmse,mae,r2\n";
    const std::vector<std::pair<std::string,ml::Metrics>> records={
        {"train",trainMetrics},{"test",testMetrics},{"test_baseline_train_mean",baseline}};
    for (const auto& record:records) {
        const auto& m=record.second;
        metrics<<record.first<<','<<m.n<<','<<m.mse<<','<<m.rmse<<','<<m.mae<<','<<m.r2<<'\n';
    }
    auto coefficients=outputFile(output/(name+"_coefficients.csv"));
    coefficients<<"term,weight,training_mean,training_scale\n";
    coefficients<<"intercept,"<<model.getBias()<<",0,1\n";
    for (std::size_t j=0;j<data.featureNames.size();++j)
        coefficients<<'"'<<data.featureNames[j]<<"\","<<model.getWeights()[j]<<','
                    <<model.getMeans()[j]<<','<<model.getScales()[j]<<'\n';
    auto predictions=outputFile(output/(name+"_predictions.csv"));
    predictions<<"partition,row_id";
    for (const auto& feature:data.featureNames) predictions<<",\""<<feature<<'"';
    predictions<<",actual,predicted,residual\n";
    auto scaling=outputFile(output/(name+"_scaling.csv"));
    scaling<<"partition,row_id";
    for (std::size_t j=0;j<data.featureNames.size();++j) scaling<<",z"<<j+1;
    scaling<<'\n';
    for (const auto& part:std::vector<std::pair<std::string,const ml::Dataset*>>{
            {"train",&split.train},{"test",&split.test}}) {
        const auto yhat=model.predict(part.second->x);
        const auto z=model.data_scaling(part.second->x);
        for (std::size_t i=0;i<yhat.size();++i) {
            predictions<<part.first<<','<<part.second->rowIds[i];
            for (double x:part.second->x[i]) predictions<<','<<x;
            predictions<<','<<part.second->y[i]<<','<<yhat[i]<<','<<part.second->y[i]-yhat[i]<<'\n';
            scaling<<part.first<<','<<part.second->rowIds[i];
            for (double value:z[i]) scaling<<','<<value;
            scaling<<'\n';
            if (part.first=="test")
                std::cout<<"test row="<<part.second->rowIds[i]<<" actual="<<part.second->y[i]
                         <<" y_hat="<<yhat[i]<<" residual="<<part.second->y[i]-yhat[i]<<'\n';
        }
    }
    metrics.close();coefficients.close();predictions.close();scaling.close();
}

void memoryAnalysis(const fs::path& output) {
    const ml::Matrix x={{0,0},{1,0},{0,1},{1,1}};
    const ml::Vector y={1,3,4,6}; // y=1+2*x1+3*x2
    ml::LinearRegression localModel;
    auto dynamicModel=std::make_unique<ml::LinearRegression>();
    globalModel.fit(x,y);localModel.fit(x,y);dynamicModel->fit(x,y);
    auto models=std::make_unique<std::vector<ml::LinearRegression>>(3,localModel);
    auto out=outputFile(output/"memory.txt");
    out<<"MEMORY ANALYSIS\ncompiler="<<__VERSION__<<"\n"
       <<"sizeof(void*)="<<sizeof(void*)<<"\nsizeof(double)="<<sizeof(double)
       <<"\nsizeof(bool)="<<sizeof(bool)<<"\nsizeof(vector<double>)="<<sizeof(ml::Vector)
       <<"\nsizeof(LinearRegression)="<<sizeof(ml::LinearRegression)
       <<"\nalignof(LinearRegression)="<<alignof(ml::LinearRegression)<<'\n';
    for (const auto& item:std::vector<std::pair<std::string,const ml::LinearRegression*>>{
            {"global",&globalModel},{"local",&localModel},{"dynamic",dynamicModel.get()}}) {
        out<<item.first<<" object="<<static_cast<const void*>(item.second)
           <<" weights_buffer="<<static_cast<const void*>(item.second->getWeights().data())
           <<" sizeof="<<sizeof(*item.second)<<" live_dynamic_bytes="<<item.second->dynamicBytes()
           <<" allocated_dynamic_bytes="<<item.second->allocatedBytes()<<'\n';
    }
    out<<"dynamic_owner_on_stack="<<static_cast<const void*>(&dynamicModel)
       <<" sizeof_owner="<<sizeof(dynamicModel)<<'\n';
    std::size_t live=0,allocated=0;
    for (std::size_t i=0;i<models->size();++i) {
        const auto& model=(*models)[i];live+=model.dynamicBytes();allocated+=model.allocatedBytes();
        out<<"models["<<i<<"]="<<static_cast<const void*>(&model)
           <<" weights_buffer="<<static_cast<const void*>(model.getWeights().data())<<'\n';
    }
    out<<"vector_object="<<static_cast<const void*>(models.get())
       <<" vector_data="<<static_cast<const void*>(models->data())<<'\n'
       <<"vector_size="<<models->size()<<" vector_capacity="<<models->capacity()
       <<"\nsizeof_vector_object="<<sizeof(*models)
       <<"\nlive_elements_bytes="<<models->size()*sizeof(ml::LinearRegression)
       <<"\nreserved_elements_bytes="<<models->capacity()*sizeof(ml::LinearRegression)
       <<"\nnested_live_bytes="<<live<<"\nnested_allocated_bytes="<<allocated
       <<"\nknown_dynamic_total="<<sizeof(*models)+models->capacity()*sizeof(ml::LinearRegression)+allocated
       <<"\nNOTE: excludes allocator metadata, rounding, other allocations and local owners.\n"
       <<"Addresses and implementation-dependent sizes may change across runs/platforms.\n";
    out.close();
    std::ifstream read(output/"memory.txt");std::cout<<'\n'<<read.rdbuf();
}
} // namespace

int main(int argc,char* argv[]) {
    try {
        if (argc>3) throw std::invalid_argument("Uso: regression_demo [data_dir] [results_dir]");
        const fs::path data=argc>1?argv[1]:"data";
        const fs::path output=argc>2?argv[2]:"results";
        fs::create_directories(output);
        std::cout<<std::setprecision(12);
        runExperiment("ice_cream",data/"Ice_cream_selling_data.csv",output);
        runExperiment("students",data/"student_exam_scores.csv",output);
        memoryAnalysis(output);
        return 0;
    } catch (const std::exception& e) {std::cerr<<"ERROR: "<<e.what()<<'\n';return 1;}
}
