#include "aha.h"
#include "mvn.h"
#include "version.h"

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

double Model::Predict(const std::vector<double>& x,
                      std::vector<double>& y) const {
  Map<const Vector> _x(x.data(), x.size());
  Vector _y;
  auto r = ((mix*)p)->Predict(_x, _y);
  y = std::vector<double>(_y.begin(), _y.end());
  return r;
}

std::string Model::Export() const {
  return ((mix*)p)->Export();
}

bool Model::Import(const std::string& model) {
  return ((mix*)p)->Import(model);
}

Trainer::Trainer(Model& m, uint64_t seed) {
  p = new trainer(*(mix*)*(void**)&m, seed);
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

double Trainer::Entropy() const {
  return ((trainer*)p)->Entropy();
}

void Trainer::Initialize() {
  ((trainer*)p)->Initialize();
}

void Trainer::Train(const std::vector<double>& sample) {
  Map<const Vector> s(sample.data(), sample.size());
  ((trainer*)p)->Train(s);
}

void Trainer::Merge(const Trainer& t) {
  ((trainer*)p)->Merge(*(const trainer*)*(void**)&t);
}

void Trainer::Update() {
  ((trainer*)p)->Update();
}

}  // namespace aha
