#ifndef GAUSS_GAUSSION_H
#define GAUSS_GAUSSION_H

#include <cassert>
#include <iostream>
#include <vector>
#pragma warning(disable : 4819)
#include <Eigen>

#include "generator.h"

#ifndef M_PI
#define M_PI 3.1415926535897932384626
#endif

using namespace Eigen;
#define Vector VectorXd
#define Matrix MatrixXd

class mvn {
 public:
  mvn() {
  }

  mvn(const Vector& mu, const Matrix& sigma) {
    Initialize(mu, sigma);
  }

  int Dim() const {
    return (int)u.size();
  }

  void Initialize(const Vector& mu, const Matrix& sigma) {
    assert(mu.size() == sigma.rows());
    assert(sigma.rows() == sigma.cols());
    auto llt = LLT<Matrix>(sigma.selfadjointView<Lower>());
    assert(llt.info() == Success);
    u = mu;
    l = llt.matrixL();
    d = Vector(mu.size());
    double c = 0;
    for (int i = 0; i < (int)mu.size(); i++) {
      c += 2 * log(l(i, i));
      d(i) = c;
    }
  }

  double Evaluate(const Vector& x) const {
    assert(x.size() == u.size());
    auto n = u.size();
    return -0.5 * (l.triangularView<Lower>().solve(x - u).squaredNorm() +
                   n * log(2 * M_PI) + d(n - 1));
  }

  // Usually you don't need it
  double PartialEvaluate(const Vector& x) const {
    assert(x.size() <= u.size());
    auto k = x.size();
    return -0.5 * (l.topLeftCorner(k, k)
                     .triangularView<Lower>()
                     .solve(x - u.head(k))
                     .squaredNorm() +
                   k * log(2 * M_PI) + d(k - 1));
  }

  double Predict(const Vector& x, Vector& y) const {
    assert(x.size() <= u.size());
    auto n = u.size();
    auto k = x.size();
    Vector temp =
      l.topLeftCorner(k, k).triangularView<Lower>().solve(x - u.head(k));
    y = l.bottomLeftCorner(n - k, k) * temp + u.tail(n - k);
    return -0.5 * (temp.squaredNorm() + k * log(2 * M_PI) + d(k - 1));
  }

 public:
  const Vector& getu() const {
    return u;
  }

  const Matrix& getl() const {
    return l;
  }

 protected:
  Vector u;
  Matrix l;
  Vector d;
};

class mix {
 public:
  mix(int rank = 0, int dim = 0) : rank(rank), dim(dim) {
  }

  bool Initialized() {
    return cores.size() > 0;
  }

  int Rank() const {
    return rank;
  }

  int Dim() const {
    return dim;
  }

  void Initialize(const std::vector<double>& weights,
                  const std::vector<Vector>& means,
                  const std::vector<Matrix>& covs) {
    rank = (int)weights.size();
    dim = means.size() > 0 ? (int)means[0].size() : 0;
    this->weights = weights;
    cores.resize(rank);
    for (int i = 0; i < rank; i++) {
      cores[i].Initialize(means[i], covs[i]);
    }
  }

  double Evaluate(const Vector& x, std::vector<double>& w) const {
    assert(x.size() == dim);
    assert(w.size() == rank);
    double wmax = -DBL_MAX;
    for (int i = 0; i < rank; i++) {
      w[i] = cores[i].Evaluate(x);
      if (w[i] > wmax) {
        wmax = w[i];
      }
    }
    double sum = 0;
    for (int i = 0; i < rank; i++) {
      w[i] = weights[i] * exp(w[i] - wmax);
      sum += w[i];
    }
    for (int i = 0; i < rank; i++) {
      w[i] /= sum;
    }
    return log(sum) + wmax;
  }

  double Predict(const Vector& x, Vector& y) const {
    assert(x.size() <= dim);
    double wmax = -DBL_MAX;
    std::vector<Vector> v(rank);
    std::vector<double> w(rank);
    for (int i = 0; i < rank; i++) {
      w[i] = cores[i].Predict(x, v[i]);
      if (w[i] > wmax) {
        wmax = w[i];
      }
    }
    double sum = 0;
    for (int i = 0; i < rank; i++) {
      w[i] = weights[i] * exp(w[i] - wmax);
      sum += w[i];
    }
    y = Vector::Zero(dim - x.size());
    for (int i = 0; i < rank; i++) {
      y += (w[i] / sum) * v[i];
    }
    return log(sum) + wmax;
  }

  bool Export(std::vector<char>& model) const {
    uint32_t r = rank;
    uint32_t d = dim;
    model.clear();
    model.insert(model.end(), (char*)&r, (char*)(&r + 1));
    model.insert(model.end(), (char*)&d, (char*)(&d + 1));
    model.insert(model.end(),
                 (char*)weights.data(),
                 (char*)(weights.data() + weights.size()));
    for (int i = 0; i < rank; i++) {
      model.insert(model.end(),
                   (char*)cores[i].getu().data(),
                   (char*)(cores[i].getu().data() + cores[i].getu().size()));
      model.insert(model.end(),
                   (char*)cores[i].getl().data(),
                   (char*)(cores[i].getl().data() + cores[i].getl().size()));
    }
    return true;
  }

  bool Import(const std::vector<char>& model) {
    uint32_t r;
    uint32_t d;
    if (model.size() < sizeof(r) + sizeof(d)) {
      return false;
    }
    r = *(uint32_t*)model.data();
    d = *(((uint32_t*)model.data()) + 1);
    if (model.size() < sizeof(r) + sizeof(d) + r * sizeof(double) +
                         r * d * sizeof(double) + r * d * d * sizeof(double)) {
      return false;
    }
    rank = (int)r;
    dim = (int)d;
    auto p = (double*)(model.data() + sizeof(r) + sizeof(d));
    weights = std::vector<double>(p, p + rank);
    p += rank;
    cores.resize(rank);
    for (int i = 0; i < rank; i++) {
      auto u = Map<Vector>(p, dim);
      p += dim;
      auto l = Map<Matrix>(p, dim, dim);
      p += dim * dim;
      cores[i].Initialize(u, l * l.transpose());
    }
    return true;
  }

  void Print() {
    for (int i = 0; i < rank; i++) {
      std::cout << i << ": " << weights[i] << "\n";
      std::cout << "mean:\n" << cores[i].getu() << "\n";
      std::cout << "sigma:\n" << cores[i].getl() * cores[i].getl().transpose() << "\n\n";
    }
  }

 protected:
  int rank;
  int dim;
  std::vector<double> weights;
  std::vector<mvn> cores;
};

class trainer {
 public:
  trainer(mix& m, uint64_t seed = 0)
    : m(m),
      rank(m.Rank()),
      dim(m.Dim()),
      entropy(0),
      seed(seed),
      weights(m.Rank()),
      means(m.Rank()),
      covs(m.Rank()),
      temp(m.Rank()) {
  }

  void Initialize() {
    entropy = 0;
    for (int i = 0; i < rank; i++) {
      weights[i] = 0;
      means[i] = Vector::Zero(dim);
      covs[i] = Matrix::Zero(dim, dim);
      temp[i] = 0;
    }
  }

  void Train(const Vector& sample) {
    if (m.Initialized()) {
      entropy -= m.Evaluate(sample, temp);
      Matrix quadric = (sample * sample.transpose()).selfadjointView<Lower>();
      for (int i = 0; i < rank; i++) {
        weights[i] += temp[i];
        means[i] += sample * temp[i];
        covs[i] += (quadric * temp[i]).selfadjointView<Lower>();
      }
    } else {
      Matrix quadric = (sample * sample.transpose()).selfadjointView<Lower>();
      for (int i = 0; i < rank; i++) {
        weights[i] += 1.0;
        means[i] += sample;
        covs[i] += quadric.selfadjointView<Lower>();
      }
    }
  }

  void Merge(const trainer& trainer) {
    entropy += trainer.entropy;
    for (int i = 0; i < rank; i++) {
      weights[i] += trainer.weights[i];
      means[i] += trainer.means[i];
      covs[i] += (trainer.covs[i]).selfadjointView<Lower>();
    }
  }

  void Update() {
    double s = 0;
    for (auto& w : weights) {
      s += w;
    }
    entropy /= s;
    for (int i = 0; i < rank; i++) {
      means[i] /= weights[i];
      covs[i] = (covs[i] / weights[i] - means[i] * means[i].transpose())
                  .selfadjointView<Lower>();
      weights[i] /= s;
    }
    if (!m.Initialized() && rank > 0) {
      MVNGenerator gen(means[0], covs[0], seed);
      for (int i = 0; i < rank; i++) {
        means[i] = gen.Gen();
        Vector diagonal = covs[i].diagonal();
        covs[i] = diagonal.asDiagonal();
      }
    }
    m.Initialize(weights, means, covs);
  }

  int Rank() const {
    return rank;
  }

  int Dim() const {
    return dim;
  }

  double Entropy() const {
    return entropy;
  }

  void Print() {
    for (int i = 0; i < rank; i++) {
      std::cout << i << ": " << weights[i] << "\n";
      std::cout << "mean:\n" << means[i] << "\n";
      std::cout << "sigma:\n" << covs[i] << "\n";
    }
  }

 protected:
  mix& m;
  int rank;
  int dim;
  double entropy;
  uint64_t seed;
  std::vector<double> weights;
  std::vector<Vector> means;
  std::vector<Matrix> covs;
  std::vector<double> temp;
};

#endif
