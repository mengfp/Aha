#ifndef GAUSS_GAUSSION_H
#define GAUSS_GAUSSION_H

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
    m_mu = mu;
    m_llt = LLT<MatrixXd>(sigma);
    m_c = -0.5 * mu.size() * log(2 * M_PI) - log(m_llt.matrixL().determinant());
  }

  double Evaluate(const VectorXd& x) {
    return -0.5 * m_llt.matrixL().solve(x - m_mu).squaredNorm() + m_c;
  }

 private:
  VectorXd m_mu;
  LLT<MatrixXd> m_llt;
  double m_c = 0;
};

}  // namespace gauss

#endif
