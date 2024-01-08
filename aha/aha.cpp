#include "aha.h"
#include "version.h"
#include "gaussian.h"

namespace aha {

std::string Version() {
  return std::to_string(MAJOR_VERSION) + "." + std::to_string(MINOR_VERSION) +
         "." + std::to_string(REVISION);
}

Model::Model(int rank, int dim) {
  p = new Mixture(rank, dim);
}

Model::~Model() {
  delete (Mixture*)p;
}

bool Model::Initialized() const {
  return ((Mixture*)p)->Initialized();
}

int Model::Rank() const {
  return ((Mixture*)p)->Rank();
}

int Model::Dim() const {
  return ((Mixture*)p)->Dim();
}

std::vector<double> Model::Predict(const std::vector<double>& x) const {
  Map<const Vector> mv_x(x.data(), x.size());
  Vector y;
  ((Mixture*)p)->Predict(mv_x, y);
  return std::vector<double>(y.begin(), y.end());
}

std::vector<double> Model::Export() const {
  return std::vector<double>();
}

bool Model::Import(const std::vector<double>& model) {
  return false;
}

}  // namespace aha
