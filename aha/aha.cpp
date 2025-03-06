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

#define AHA_IMPLEMENT(Model, Trainer, mix, trainer, V, T)                                 \
                                                                            \
  Model::Model(int rank, int dim) {                                         \
    p = new mix(rank, dim);                                              \
  }                                                                         \
                                                                            \
  Model::~Model() {                                                         \
    delete (mix*)p;                                                      \
  }                                                                         \
                                                                            \
  bool Model::Initialized() const {                                         \
    return ((mix*)p)->Initialized();                                     \
  }                                                                         \
                                                                            \
  int Model::Rank() const {                                                 \
    return ((mix*)p)->Rank();                                            \
  }                                                                         \
                                                                            \
  int Model::Dim() const {                                                  \
    return ((mix*)p)->Dim();                                             \
  }                                                                         \
                                                                            \
  double Model::Predict(const V& x, V& y) const {                           \
    return ((mix*)p)->Predict(x, y);                                     \
  }                                                                         \
                                                                            \
  double Model::Predict(const std::vector<T>& x, std::vector<T>& y) const { \
    Map<const V> _x(x.data(), x.size());                                    \
    V _y;                                                                   \
    auto r = ((mix*)p)->Predict(_x, _y);                                 \
    y.assign(_y.begin(), _y.end());                                         \
    return r;                                                               \
  }                                                                         \
                                                                            \
  void Model::Sort() {                                                      \
    return ((mix*)p)->Sort();                                            \
  }                                                                         \
                                                                            \
  std::string Model::Export() const {                                       \
    return ((mix*)p)->Export();                                          \
  }                                                                         \
                                                                            \
  bool Model::Import(const std::string& model) {                            \
    return ((mix*)p)->Import(model);                                     \
  }                                                                         \
                                                                            \
  Trainer::Trainer(Model& m) {                                              \
    p = new trainer(*(mix*)*(void**)&m);                              \
  }                                                                         \
                                                                            \
  Trainer::~Trainer() {                                                     \
    delete (trainer*)p;                                                  \
  }                                                                         \
                                                                            \
  int Trainer::Rank() const {                                               \
    return ((trainer*)p)->Rank();                                        \
  }                                                                         \
                                                                            \
  int Trainer::Dim() const {                                                \
    return ((trainer*)p)->Dim();                                         \
  }                                                                         \
                                                                            \
  void Trainer::Train(const V& sample) {                                    \
    ((trainer*)p)->Train(sample);                                        \
  }                                                                         \
                                                                            \
  void Trainer::Train(const std::vector<T>& sample) {                       \
    Map<const V> s(sample.data(), sample.size());                           \
    ((trainer*)p)->Train(s);                                             \
  }                                                                         \
                                                                            \
  bool Trainer::Merge(const Trainer& t, double w) {                         \
    return ((trainer*)p)->Merge(*(const trainer*)*(void**)&t, w);     \
  }                                                                         \
                                                                            \
  std::string Trainer::Spit() {                                             \
    return ((trainer*)p)->Spit();                                        \
  }                                                                         \
                                                                            \
  bool Trainer::Swallow(const std::string& t, double w) {                   \
    return ((trainer*)p)->Swallow(t, w);                                 \
  }                                                                         \
                                                                            \
  double Trainer::Update(double noise_floor) {                              \
    return ((trainer*)p)->Update(noise_floor);                           \
  }                                                                         \
                                                                            \
  void Trainer::Reset() {                                                   \
    ((trainer*)p)->Reset();                                              \
  }

AHA_IMPLEMENT(Model64, Trainer64, mix64, trainer64, VectorXd, double)
AHA_IMPLEMENT(Model32, Trainer32, mix32, trainer32, VectorXf, float)

}  // namespace aha
