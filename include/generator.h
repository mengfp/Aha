/*
** Copyright 2025 Meng, Fanping. All rights reserved.
*/
#ifndef AHA_GENERATOR_H
#define AHA_GENERATOR_H

#include <chrono>
#include <vector>
#include <Eigen/Dense>

#define register
#include "MersenneTwister.h"
#undef register

namespace aha {

using namespace Eigen;

inline uint64_t nano() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
           std::chrono::steady_clock::now().time_since_epoch())
    .count();
}

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
      means[i] = VectorXd::Zero(dim);
      for (int j = 0; j < dim; j++) {
        means[i][j] = rand.randNorm(0.0, 1.0);
      }
      ls[i] = MatrixXd::Identity(dim, dim);
      for (int j = 0; j < dim; j++) {
        auto r = rand.randNorm(0.0, 1.0);
        ls[i](j, j) *= (r * r + 0.5);
        for (int k = j + 1; k < dim; k++) {
          ls[i].row(k) += ls[i].row(j) * rand.randNorm(0.0, 1.0);
        }
      }
    }
  }

  void Gen(VectorXd& sample) {
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
  std::vector<VectorXd> means;
  std::vector<MatrixXd> ls;
  MTRand rand;
};

class Gen2 {
 public:
  void Init(uint32_t seed) {
    rand.seed(seed);
  }

  void gen(std::vector<double>& sample) {
    auto x = rand.randNorm(0.0, 1.0);
    auto y = rand.randNorm(0.0, 1.0);
    auto z = rand.randNorm(0.0, 1.0);
    auto r = rand.randInt() % 4;
    if (r == 0) {
      sample[0] = x * 1000 + 1;
      sample[1] = y + 1;
      sample[2] = x * 1000 + z + 1;
    } else if (r == 1) {
      sample[0] = x - 1;
      sample[1] = y * 1000 - 1;
      sample[2] = -y * 1000 + z - 1;
    } else {
      sample[0] = x * 1000;
      sample[1] = y * 1000;
      sample[2] = z;
    }
  }

 protected:
  MTRand rand;
};

class GenNonLinear {
 public:
  void Init(uint32_t seed) {
    rand.seed(seed);
  }

  void gen(std::vector<double>& sample) {
    auto a = rand.randNorm(0.0, 1.0);
    auto b = rand.randNorm(0.0, 1.0);
    auto c = rand.randNorm(0.0, 1.0);
    sample[0] = a;
    sample[1] = b;
    sample[2] = a * b + c;
  }

 protected:
  MTRand rand;
};

class MVNGenerator {
 public:
  MVNGenerator() {
  }

  MVNGenerator(const VectorXd& mean, const MatrixXd& cov, uint64_t seed = 0) {
    Init(mean, cov, seed);
  }

  void Init(const VectorXd& mean, const MatrixXd& cov, uint64_t seed = 0) {
    this->mean = mean;
    this->L = LLT<MatrixXd>(cov.selfadjointView<Lower>()).matrixL();
    if (seed == 0) {
      seed = nano() ^ (uint64_t)this;
    }
    rand.seed((MTRand::uint32*)&seed, 2);
  }

  VectorXd Gen() {
    VectorXd v = VectorXd::Zero(mean.size());
    for (auto& x : v) {
      x = rand.randNorm(0, 1);
    }
    return L.triangularView<Lower>() * v + mean;
  }

 public:
  VectorXd mean;
  MatrixXd L;
  MTRand rand;
};

}  // namespace aha

#endif
