#ifndef GAUSS_GENERATOR_H
#define GAUSS_GENERATOR_H

#include <vector>
#include "MersenneTwister.h"
#include "gaussian.h"

namespace gauss {

class Generator {
 public:
  void Initialize(int rank, int dim, uint32_t seed) {
    rand.seed(seed);
    weights.resize(rank);
    means.resize(rank);
    ls.resize(rank);
    double s = 0;
    for (int i = 0; i < rank; i++) {
      weights[i] = rand.randInt() % 3 + 1;
      s += weights[i];
    }
    for (int i = 0; i < rank; i++) {
      weights[i] /= s;
      means[i] = Vector::Zero(dim);
      for (int j = 0; j < dim; j++) {
        means[i][j] = rand.randNorm(0.0, 1.0);
      }
      ls[i] = Matrix::Identity(dim, dim);
      for (int j = 0; j < dim; j++) {
        auto r = rand.randNorm(0.0, 1.0);
        ls[i](j, j) *= (r * r + 0.5);
        for (int k = j + 1; k < dim; k++) {
          ls[i].row(k) += ls[i].row(j) * rand.randNorm(0.0, 1.0);
        }
      }
    }
  }

  void Gen(Vector& sample) {
    auto r = rand.rand();
    double s = 0;
    int i = 0;
    for (; i < (int)weights.size() - 1; i++) {
      s += weights[i];
      if (s > r) {
        break;
      }
    }
    for (int j = 0; j < (int)sample.size(); j++) {
      sample[j] = rand.randNorm(0.0, 1.0);
    }
    sample = ls[i] * sample + means[i];
  }

  void Print() {
    for (int i = 0; i < (int)weights.size(); i++) {
      std::cout << i << ": " << weights[i] << "\n";
      std::cout << "mean:\n" << means[i] << "\n";
      std::cout << "sigma:\n" << ls[i] * ls[i].transpose() << "\n";
    }
  }

 protected:
  std::vector<double> weights;
  std::vector<Vector> means;
  std::vector<Matrix> ls;
  MTRand rand;
};

}  // namespace gauss

#endif

