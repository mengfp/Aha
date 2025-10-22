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

double Model::Predict(const VectorXd& x, VectorXd& y) const {
  return ((mix*)p)->Predict(x, y);
}

double Model::Predict(const std::vector<double>& x,
                      std::vector<double>& y) const {
  Map<const VectorXd> _x(x.data(), x.size());
  VectorXd _y;
  auto r = ((mix*)p)->Predict(_x, _y);
  y.assign(_y.begin(), _y.end());
  return r;
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

VectorXd Model::BatchPredict(const MatrixXd& X, MatrixXd& Y) const {
  return ((mix*)p)->BatchPredict(X, Y);
}

VectorXd Model::FastPredict(const MatrixXd& X, MatrixXd& Y) const {
  return ((mix*)p)->FastPredict(X, Y);
}

std::vector<char> Model::Dump() const {
  return ((mix*)p)->Dump();
}

bool Model::Load(const std::vector<char>& input) {
  return ((mix*)p)->Load(input);
}

double Model::PredictEx(const VectorXd& x, VectorXd& y, MatrixXd& cov) const {
  return ((mix*)p)->PredictEx(x, y, cov);
}

VectorXd Model::BatchPredictEx(const MatrixXd& X,
                             MatrixXd& Y,
                             MatrixXd& COV) const {
  return ((mix*)p)->BatchPredictEx(X, Y, COV);
}

VectorXd Model::FastPredictEx(const MatrixXd& X,
                               MatrixXd& Y,
                               MatrixXd& COV) const {
  return ((mix*)p)->FastPredictEx(X, Y, COV);
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

void Trainer::Train(const VectorXd& sample) {
  ((trainer*)p)->Train(sample);
}

void Trainer::Train(const std::vector<double>& sample) {
  Map<const VectorXd> s(sample.data(), sample.size());
  ((trainer*)p)->Train(s);
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

void Trainer::BatchTrain(const MatrixXd& samples) {
  ((trainer*)p)->BatchTrain(samples);
}

void Trainer::FastTrain(const MatrixXd& samples) {
  ((trainer*)p)->FastTrain(samples);
}

std::vector<char> Trainer::Dump() const {
  return ((trainer*)p)->Dump();
}

bool Trainer::Load(const std::vector<char>& input, double w) {
  return ((trainer*)p)->Load(input, w);
}

}  // namespace aha
