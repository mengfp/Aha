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

class Gaussian {
 public:
  Gaussian() {
  }

  Gaussian(const VectorXd& mu, const MatrixXd& sigma) {
    Initialize(mu, sigma);
  }

  void Initialize(const VectorXd& mu, const MatrixXd& sigma) {
    assert(mu.size() == sigma.rows() && sigma.rows() == sigma.cols());
    auto llt = LLT<MatrixXd>(sigma);
    assert(llt.info() == Success);
    u = mu;
    L = llt.matrixL();
    d = VectorXd(mu.size());
    double c = 0;
    for (Index i = 0; i < mu.size(); i++) {
      c += 2 * log(L(i, i));
      d(i) = c;
    }
  }

  double Evaluate(const VectorXd& x) {
    assert(x.size() == u.size());
    auto n = u.size();
    return -0.5 *
           (TriangularView<MatrixXd, Lower>(L).solve(x - u).squaredNorm() +
            n * log(2 * M_PI) + d(n - 1));
  }

  double PartialEvaluate(const VectorXd& x) {
    assert(x.size() <= u.size());
    auto k = x.size();
    auto top = L.topLeftCorner(k, k);
    return -0.5 * (TriangularView<Block<MatrixXd>, Lower>(top)
                     .solve(x - u.head(k))
                     .squaredNorm() +
                   k * log(2 * M_PI) + d(k - 1));
  }

  double Predict(const VectorXd& x, VectorXd& y) {
    assert(x.size() <= u.size());
    auto n = u.size();
    auto k = x.size();
    auto top = L.topLeftCorner(k, k);
    VectorXd temp =
      TriangularView<Block<MatrixXd>, Lower>(top).solve(x - u.head(k));
    y = L.bottomLeftCorner(n - k, k) * temp + u.tail(n - k);
    return -0.5 * (temp.squaredNorm() + k * log(2 * M_PI) + d(k - 1));
  }

 protected:
  VectorXd u;
  MatrixXd L;
  VectorXd d;
};

}  // namespace gauss

#endif
