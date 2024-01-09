#include "aha.h"

#include "mvn.h"
#include "version.h"

namespace aha {

std::string Version() {
  return std::to_string(MAJOR_VERSION) + "." + std::to_string(MINOR_VERSION) +
         "." + std::to_string(REVISION);
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

double Model::Predict(const std::vector<double>& x, std::vector<double>& y) const {
  Map<const Vector> _x(x.data(), x.size());
  Vector _y;
  auto r = ((mix*)p)->Predict(_x, _y);
  y = std::vector<double>(_y.begin(), _y.end());
  return r;
}

bool Model::Export(std::vector<char>& model) const {
  return false;
}

bool Model::Import(const std::vector<char>& model) {
  return false;
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

double Trainer::Score() const {
  return ((trainer*)p)->Score();
}

void Trainer::Initialize() {
  ((trainer*)p)->Initialize();
}

void Trainer::Merge(const Trainer& t) {
  ((trainer*)p)->Merge(*(const trainer*)*(void**)&t);
}

void Trainer::Update() {
  ((trainer*)p)->Update();
}

}  // namespace aha
