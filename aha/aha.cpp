/*
** Copyright 2025 Meng, Fanping. All rights reserved.
*/
#ifdef _MSC_VER
#pragma warning(disable : 4819 4805)
#endif

#include <aha.h>
#include <mvn.h>
#include <version.h>

namespace aha {

std::string Version() {
  return VERSION;
}

Model::Model(int rank, int dim) {
  p = new mix(rank, dim);
}

Model::~Model() {
  delete (mix*)p;
}

bool Model::Initialized() const {
  return ((mix*)p)->Initialized();
}

int Model::Rank() const {
  return ((mix*)p)->Rank();
}

int Model::Dim() const {
  return ((mix*)p)->Dim();
}

bool Model::IsIll() const {
  return ((mix*)p)->IsIll();
}

double Model::Predict(Ref<const VectorXd> x, VectorXd& y) const {
  return ((mix*)p)->Predict(x, y);
}

void Model::Sort() {
  return ((mix*)p)->Sort();
}

std::string Model::Export() const {
  return ((mix*)p)->Export();
}

bool Model::Import(const std::string& model) {
  return ((mix*)p)->Import(model);
}

VectorXd Model::BatchPredict(Ref<const MatrixXd> X, MatrixXd& Y) const {
  return ((mix*)p)->BatchPredict(X, Y);
}

VectorXd Model::FastPredict(Ref<const MatrixXf> X, MatrixXf& Y) const {
  return ((mix*)p)->FastPredict(X, Y);
}

std::vector<char> Model::Dump() const {
  return ((mix*)p)->Dump();
}

bool Model::Load(const std::vector<char>& input) {
  return ((mix*)p)->Load(input);
}

double Model::PredictEx(Ref<const VectorXd> x,
                        VectorXd& y,
                        VectorXd& vars) const {
  return ((mix*)p)->PredictEx(x, y, vars);
}

VectorXd Model::BatchPredictEx(Ref<const MatrixXd> X,
                               MatrixXd& Y,
                               MatrixXd& VARS) const {
  return ((mix*)p)->BatchPredictEx(X, Y, VARS);
}

VectorXd Model::FastPredictEx(Ref<const MatrixXf> X,
                              MatrixXf& Y,
                              MatrixXf& VARS) const {
  return ((mix*)p)->FastPredictEx(X, Y, VARS);
}

Trainer::Trainer(Model& m) {
  p = new trainer(*(mix*)*(void**)&m);
}

Trainer::~Trainer() {
  delete (trainer*)p;
}

int Trainer::Rank() const {
  return ((trainer*)p)->Rank();
}

int Trainer::Dim() const {
  return ((trainer*)p)->Dim();
}

void Trainer::Train(Ref<const VectorXd> sample) {
  ((trainer*)p)->Train(sample);
}

bool Trainer::Merge(const Trainer& t, double w) {
  return ((trainer*)p)->Merge(*(const trainer*)*(void**)&t, w);
}

std::string Trainer::Spit() const {
  return ((trainer*)p)->Spit();
}

bool Trainer::Swallow(const std::string& t, double w) {
  return ((trainer*)p)->Swallow(t, w);
}

double Trainer::Update(double noise_floor) {
  return ((trainer*)p)->Update(noise_floor);
}

void Trainer::Reset() {
  return ((trainer*)p)->Reset();
}

void Trainer::BatchTrain(Ref<const MatrixXd> samples) {
  ((trainer*)p)->BatchTrain(samples);
}

void Trainer::FastTrain(Ref<const MatrixXf> samples) {
  ((trainer*)p)->FastTrain(samples);
}

std::vector<char> Trainer::Dump() const {
  return ((trainer*)p)->Dump();
}

bool Trainer::Load(const std::vector<char>& input, double w) {
  return ((trainer*)p)->Load(input, w);
}

bool Trainer::Healthy() const {
  return ((trainer*)p)->Healthy();
}

void Trainer::SetInitMethod(int method) {
  static_cast<aha::trainer*>(p)->SetInitMethod(method);
}

}  // namespace aha
