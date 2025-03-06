/*
** Copyright 2025 Meng, Fanping. All rights reserved.
*/
#ifdef _MSC_VER
#pragma warning(disable : 4819)
#endif

#include <aha.h>
#include <mvn.h>
#include <version.h>

namespace aha {

using namespace Eigen;

std::string Version() {
  return VERSION;
}

#define AHA_IMPLEMENT(Model, Trainer, V, T)                                 \
                                                                            \
  Model::Model(int rank, int dim) {                                         \
    p = new mix<T>(rank, dim);                                              \
  }                                                                         \
                                                                            \
  Model::~Model() {                                                         \
    delete (mix<T>*)p;                                                      \
  }                                                                         \
                                                                            \
  bool Model::Initialized() const {                                         \
    return ((mix<T>*)p)->Initialized();                                     \
  }                                                                         \
                                                                            \
  int Model::Rank() const {                                                 \
    return ((mix<T>*)p)->Rank();                                            \
  }                                                                         \
                                                                            \
  int Model::Dim() const {                                                  \
    return ((mix<T>*)p)->Dim();                                             \
  }                                                                         \
                                                                            \
  double Model::Predict(const V& x, V& y) const {                           \
    return ((mix<T>*)p)->Predict(x, y);                                     \
  }                                                                         \
                                                                            \
  double Model::Predict(const std::vector<T>& x, std::vector<T>& y) const { \
    Map<const V> _x(x.data(), x.size());                                    \
    V _y;                                                                   \
    auto r = ((mix<T>*)p)->Predict(_x, _y);                                 \
    y.assign(_y.begin(), _y.end());                                         \
    return r;                                                               \
  }                                                                         \
                                                                            \
  void Model::Sort() {                                                      \
    return ((mix<T>*)p)->Sort();                                            \
  }                                                                         \
                                                                            \
  std::string Model::Export() const {                                       \
    return ((mix<T>*)p)->Export();                                          \
  }                                                                         \
                                                                            \
  bool Model::Import(const std::string& model) {                            \
    return ((mix<T>*)p)->Import(model);                                     \
  }                                                                         \
                                                                            \
  Trainer::Trainer(Model& m) {                                              \
    p = new trainer<T>(*(mix<T>*)*(void**)&m);                              \
  }                                                                         \
                                                                            \
  Trainer::~Trainer() {                                                     \
    delete (trainer<T>*)p;                                                  \
  }                                                                         \
                                                                            \
  int Trainer::Rank() const {                                               \
    return ((trainer<T>*)p)->Rank();                                        \
  }                                                                         \
                                                                            \
  int Trainer::Dim() const {                                                \
    return ((trainer<T>*)p)->Dim();                                         \
  }                                                                         \
                                                                            \
  void Trainer::Train(const V& sample) {                                    \
    ((trainer<T>*)p)->Train(sample);                                        \
  }                                                                         \
                                                                            \
  void Trainer::Train(const std::vector<T>& sample) {                       \
    Map<const V> s(sample.data(), sample.size());                           \
    ((trainer<T>*)p)->Train(s);                                             \
  }                                                                         \
                                                                            \
  bool Trainer::Merge(const Trainer& t, double w) {                         \
    return ((trainer<T>*)p)->Merge(*(const trainer<T>*)*(void**)&t, w);     \
  }                                                                         \
                                                                            \
  std::string Trainer::Spit() {                                             \
    return ((trainer<T>*)p)->Spit();                                        \
  }                                                                         \
                                                                            \
  bool Trainer::Swallow(const std::string& t, double w) {                   \
    return ((trainer<T>*)p)->Swallow(t, w);                                 \
  }                                                                         \
                                                                            \
  double Trainer::Update(double noise_floor) {                              \
    return ((trainer<T>*)p)->Update(noise_floor);                           \
  }                                                                         \
                                                                            \
  void Trainer::Reset() {                                                   \
    ((trainer<T>*)p)->Reset();                                              \
  }

AHA_IMPLEMENT(Model64, Trainer64, VectorXd, double)
AHA_IMPLEMENT(Model32, Trainer32, VectorXf, float)

}  // namespace aha
