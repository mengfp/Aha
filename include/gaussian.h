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

namespace gauss {

using namespace Eigen;
using Vector = VectorXd;
using Matrix = MatrixXd;

class Gaussian {
 public:
  Gaussian() {
  }

  Gaussian(const Vector& mu, const Matrix& sigma) {
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

class Mixture {
 public:
  Mixture() {
  }

  bool Initialized() {
    return gaussians.size() > 0;
  }

  int Rank() const {
    return (int)gaussians.size();
  }

  int Dim() const {
    if (gaussians.empty()) {
      return 0;
    } else {
      return gaussians[0].Dim();
    }
  }

  void Initialize(const std::vector<double>& weights,
                  const std::vector<Vector>& means,
                  const std::vector<Matrix>& covariances) {
    assert(weights.size() == means.size());
    assert(means.size() == covariances.size());
    this->weights = weights;
    gaussians.resize(weights.size());
    for (int i = 0; i < (int)gaussians.size(); i++) {
      gaussians[i].Initialize(means[i], covariances[i]);
    }
  }

  double Evaluate(const Vector& x, std::vector<double>& w) const {
    assert(w.size() == gaussians.size());
    double wmax = -DBL_MAX;
    for (int i = 0; i < (int)w.size(); i++) {
      w[i] = gaussians[i].Evaluate(x);
      if (w[i] > wmax) {
        wmax = w[i];
      }
    }
    double sum = 0;
    for (int i = 0; i < w.size(); i++) {
      w[i] = weights[i] * exp(w[i] - wmax);
      sum += w[i];
    }
    for (int i = 0; i < w.size(); i++) {
      w[i] /= sum;
    }
    return log(sum) + wmax;
  }

  double Predict(const Vector& x, Vector& y) const {
    double wmax = -DBL_MAX;
    std::vector<Vector> v(Rank());
    std::vector<double> w(Rank());
    for (int i = 0; i < Rank(); i++) {
      w[i] = gaussians[i].Predict(x, v[i]);
      if (w[i] > wmax) {
        wmax = w[i];
      }
    }
    double sum = 0;
    y = Vector::Zero(Dim() - x.size());
    for (int i = 0; i < Rank(); i++) {
      auto temp = weights[i] * exp(w[i] - wmax);
      y += temp * v[i];
      sum += temp;
    }
    return log(sum) + wmax;
  }

 protected:
  std::vector<double> weights;
  std::vector<Gaussian> gaussians;
};

class Trainer {
 public:
  Trainer(Mixture& mixture)
    : mixture(mixture),
      rank(mixture.Rank()),
      dim(mixture.Dim()),
      score(0),
      weights(mixture.Rank()),
      means(mixture.Rank()),
      covariances(mixture.Rank()),
      temp(mixture.Rank()) {
  }

  Trainer(Mixture& mixture, int rank, int dim)
    : mixture(mixture),
      rank(rank),
      dim(dim),
      score(0),
      weights(rank),
      means(rank),
      covariances(rank),
      temp(rank) {
  }

  void Initialize() {
    score = 0;
    for (int i = 0; i < rank; i++) {
      weights[i] = 0;
      means[i] = Vector::Zero(dim);
      covariances[i] = Matrix::Zero(dim, dim);
      temp[i] = 0;
    }
  }

  void Train(const Vector& sample) {
    if (mixture.Initialized()) {
      score += mixture.Evaluate(sample, temp);
      Matrix quadric = (sample * sample.transpose()).selfadjointView<Lower>();
      for (int i = 0; i < rank; i++) {
        weights[i] += temp[i];
        means[i] += sample * temp[i];
        covariances[i] += (quadric * temp[i]).selfadjointView<Lower>();
      }
    } else {
      auto var = sample.array().square();
      for (int i = 0; i < rank; i++) {
        weights[i] += 1.0 / rank;
        means[i] += sample / rank;
        covariances[i].diagonal() += var.matrix() * ((i + 1.0) / rank);
      }
    }
  }

  void Merge(const Trainer& trainer) {
    score += trainer.score;
    for (int i = 0; i < rank; i++) {
      weights[i] += trainer.weights[i];
      means[i] += trainer.means[i];
      covariances[i] += (trainer.covariances[i]).selfadjointView<Lower>();
    }
  }

  void Update() {
    double s = 0;
    for (auto& w : weights) {
      s += w;
    }
    score /= s;
    for (int i = 0; i < rank; i++) {
      means[i] /= weights[i];
      covariances[i] =
        (covariances[i] / weights[i] - means[i] * means[i].transpose())
          .selfadjointView<Lower>();
      weights[i] /= s;
    }
    mixture.Initialize(weights, means, covariances);
  }

  int Rank() const {
    return rank;
  }

  int Dim() const {
    return dim;
  }

  double Score() const {
    return score;
  }

  void Print() {
    for (int i = 0; i < rank; i++) {
      std::cout << i << ": " << weights[i] << "\n";
      std::cout << "mean:\n" << means[i] << "\n";
      std::cout << "sigma:\n" << covariances[i] << "\n";
    }
  }

 protected:
  Mixture& mixture;
  int rank;
  int dim;
  double score;
  std::vector<double> weights;
  std::vector<Vector> means;
  std::vector<Matrix> covariances;
  std::vector<double> temp;
};

}  // namespace gauss

#endif
