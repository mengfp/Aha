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

  void Initialize(const Vector& mu, const Matrix& sigma) {
    assert(mu.size() == sigma.rows() && sigma.rows() == sigma.cols());
    auto llt = LLT<Matrix>(sigma);
    assert(llt.info() == Success);
    u = mu;
    L = llt.matrixL();
    d = Vector(mu.size());
    double c = 0;
    for (Index i = 0; i < mu.size(); i++) {
      c += 2 * log(L(i, i));
      d(i) = c;
    }
  }

  double Evaluate(const Vector& x) {
    assert(x.size() == u.size());
    auto n = u.size();
    return -0.5 * (TriangularView<Matrix, Lower>(L).solve(x - u).squaredNorm() +
                   n * log(2 * M_PI) + d(n - 1));
  }

  // Usually you don't need it
  double PartialEvaluate(const Vector& x) {
    assert(x.size() <= u.size());
    auto k = x.size();
    auto top = L.topLeftCorner(k, k);
    return -0.5 * (TriangularView<Block<Matrix>, Lower>(top)
                     .solve(x - u.head(k))
                     .squaredNorm() +
                   k * log(2 * M_PI) + d(k - 1));
  }

  double Predict(const Vector& x, Vector& y) {
    assert(x.size() <= u.size());
    auto n = u.size();
    auto k = x.size();
    auto top = L.topLeftCorner(k, k);
    Vector temp =
      TriangularView<Block<Matrix>, Lower>(top).solve(x - u.head(k));
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

  void Initialize(const std::vector<double>& weights,
                  const std::vector<Vector>& means,
                  const std::vector<Matrix>& covariances) {

  }

 double Evaluate(const Vector& x, std::vector<double>& w) {
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
  std::vector<double> values;
};

class Trainer {
 public:
  void Initialize() {
  }

  void Train() {
  }

 protected:
  Mixture mixture;
  std::vector<double> weights;
  std::vector<Vector> means;
  std::vector<Matrix> covariances;
};


}  // namespace gauss

#endif
