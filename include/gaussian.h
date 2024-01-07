#ifndef GAUSS_GAUSSION_H
#define GAUSS_GAUSSION_H

#include <cassert>

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

 protected:
  std::vector<double> weights;
  std::vector<Gaussian> gaussians;
};

class Trainer {
 public:
  void Initialize(int rank, int dim) {
    assert(rank > 0);
    assert(dim > 0);
    score = 0;
    weights.resize(rank);
    means.resize(rank);
    covariances.resize(rank);
    temp.resize(rank);
    for (int i = 0; i < rank; i++) {
      weights[i] = 0;
      means[i] = Vector::Zero(dim);
      covariances[i] = Matrix::Zero(dim, dim);
      temp[i] = 0;
    }
  }

  void Initialize(const Mixture& mix) {
    mixture = &mix;
    Initialize(mix.Rank(), mix.Dim());
  }

  void PreTrain(const Vector& sample) {
    auto var = sample.array().square();
    for (int i = 0; i < (int)weights.size(); i++) {
      weights[i] += 1.0 / weights.size();
      means[i] += sample / weights.size();
      covariances[i].diagonal() += var.matrix() * ((i + 1.0) / weights.size());
    }
  }

  void Train(const Vector& sample) {
    score += mixture->Evaluate(sample, temp);
    auto quadric = sample * sample.transpose();
    for (int i = 0; i < (int)weights.size(); i++) {
      weights[i] += temp[i];
      means[i] += sample * temp[i];
      covariances[i] += quadric * temp[i];
    }
  }

  void Merge(const Trainer& trainer) {
    assert(mixture == trainer.mixture);
    score += trainer.score;
    for (int i = 0; i < (int)weights.size(); i++) {
      weights[i] += trainer.weights[i];
      means[i] += trainer.means[i];
      covariances[i] += trainer.covariances[i];
    }
  }

  void Finalize() {
    double s = 0;
    for (auto& w : weights) {
      s += w;
    }
    score /= s;
    for (int i = 0; i < (int)weights.size(); i++) {
      means[i] /= weights[i];
      auto sigma =
        covariances[i] / weights[i] - means[i] * means[i].transpose();
      weights[i] /= s;
    }
  }

  double Score() const {
    return score;
  }

  const std::vector<double>& Weights() const {
    return weights;
  }

  const std::vector<Vector>& Means() const {
    return means;
  }

  const std::vector<Matrix>& Covariances() const {
    return covariances;
  }

 protected:
  const Mixture* mixture = nullptr;
  double score = 0;
  std::vector<double> weights;
  std::vector<Vector> means;
  std::vector<Matrix> covariances;
  std::vector<double> temp;
};

}  // namespace gauss

#endif
