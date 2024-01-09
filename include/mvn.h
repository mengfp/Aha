#ifndef GAUSS_GAUSSION_H
#define GAUSS_GAUSSION_H

#include <cassert>
#include <iostream>
#include <vector>

#pragma warning(disable : 4819)
#include <Eigen>

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
    L = llt.matrixL();
    d = Vector(mu.size());
    double c = 0;
    for (int i = 0; i < (int)mu.size(); i++) {
      c += 2 * log(L(i, i));
      d(i) = c;
    }
  }

  double Evaluate(const Vector& x) const {
    assert(x.size() == u.size());
    auto n = u.size();
    return -0.5 * (L.triangularView<Lower>().solve(x - u).squaredNorm() +
                   n * log(2 * M_PI) + d(n - 1));
  }

  // Usually you don't need it
  double PartialEvaluate(const Vector& x) const {
    assert(x.size() <= u.size());
    auto k = x.size();
    return -0.5 * (L.topLeftCorner(k, k)
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
      L.topLeftCorner(k, k).triangularView<Lower>().solve(x - u.head(k));
    y = L.bottomLeftCorner(n - k, k) * temp + u.tail(n - k);
    return -0.5 * (temp.squaredNorm() + k * log(2 * M_PI) + d(k - 1));
  }

 protected:
  Vector u;
  Matrix L;
  Vector d;
};

class mix {
 public:
  mix(int rank, int dim) : rank(rank), dim(dim) {
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
    assert(weights.size() == rank);
    assert(means.size() == rank);
    assert(covs.size() == rank);
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
    y = Vector::Zero(dim - x.size());
    for (int i = 0; i < rank; i++) {
      auto temp = weights[i] * exp(w[i] - wmax);
      y += temp * v[i];
      sum += temp;
    }
    return log(sum) + wmax;
  }

 protected:
  int rank;
  int dim;
  std::vector<double> weights;
  std::vector<mvn> cores;
};

class trainer {
 public:
  trainer(mix& m)
    : m(m),
      rank(m.Rank()),
      dim(m.Dim()),
      entropy(0),
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
      auto var = sample.array().square();
      for (int i = 0; i < rank; i++) {
        weights[i] += 1.0 / rank;
        means[i] += sample / rank;
        covs[i].diagonal() += var.matrix() * ((i + 1.0) / rank);
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
      covs[i] =
        (covs[i] / weights[i] - means[i] * means[i].transpose())
          .selfadjointView<Lower>();
      weights[i] /= s;
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
  std::vector<double> weights;
  std::vector<Vector> means;
  std::vector<Matrix> covs;
  std::vector<double> temp;
};

#endif
