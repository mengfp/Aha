/*
** Copyright 2025 Meng, Fanping. All rights reserved.
*/
#ifndef AHA_GENERATOR_H
#define AHA_GENERATOR_H

#include <chrono>
#include <vector>

#include "eigen.h"

#define register
#include "MersenneTwister.h"
#undef register

namespace aha {

inline uint64_t nano() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
           std::chrono::steady_clock::now().time_since_epoch())
    .count();
}

template <typename T>
class Generator {
 public:
  void Initialize(int rank, int dim, uint32_t seed) {
    rand.seed(seed);
    weights.resize(rank);
    means.resize(rank);
    ls.resize(rank);
    double s = 0;
    for (int i = 0; i < rank; i++) {
      weights[i] = T(rand.randInt() % 3 + 1);
      s += weights[i];
    }
    for (int i = 0; i < rank; i++) {
      weights[i] /= T(s);
      means[i] = Vector<T>::Zero(dim);
      for (int j = 0; j < dim; j++) {
        means[i][j] = (T)rand.randNorm(0.0, 1.0);
      }
      ls[i] = Matrix<T>::Identity(dim, dim);
      for (int j = 0; j < dim; j++) {
        auto r = rand.randNorm(0.0, 1.0);
        ls[i](j, j) *= T(r * r + 0.5);
        for (int k = j + 1; k < dim; k++) {
          ls[i].row(k) += ls[i].row(j) * rand.randNorm(0.0, 1.0);
        }
      }
    }
  }

  void Gen(Vector<T>& sample) {
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
      sample[j] = (T)rand.randNorm(0.0, 1.0);
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
  std::vector<T> weights;
  std::vector<Vector<T>> means;
  std::vector<Matrix<T>> ls;
  MTRand rand;
};

template <typename T>
class Gen2 {
 public:
  void Init(uint32_t seed) {
    rand.seed(seed);
  }

  void gen(std::vector<T>& sample) {
    auto x = rand.randNorm(0.0, 1.0);
    auto y = rand.randNorm(0.0, 1.0);
    auto z = rand.randNorm(0.0, 1.0);
    auto r = rand.randInt() % 4;
    if (r == 0) {
      sample[0] = T(x * 1000 + 1);
      sample[1] = T(y + 1);
      sample[2] = T(x * 1000 + z + 1);
    } else if (r == 1) {
      sample[0] = T(x - 1);
      sample[1] = T(y * 1000 - 1);
      sample[2] = T(- y * 1000 + z - 1);
    } else {
      sample[0] = T(x * 1000);
      sample[1] = T(y * 1000);
      sample[2] = T(z);
    }
  }

 protected:
  MTRand rand;
};

template <typename T>
class GenNonLinear {
 public:
  void Init(uint32_t seed) {
    rand.seed(seed);
  }

  void gen(std::vector<T>& sample) {
    auto a = rand.randNorm(0.0, 1.0);
    auto b = rand.randNorm(0.0, 1.0);
    auto c = rand.randNorm(0.0, 1.0);
    sample[0] = T(a);
    sample[1] = T(b);
    sample[2] = T(a * b + c);
  }

 protected:
  MTRand rand;
};

template <typename T>
class MVNGenerator {
 public:
  MVNGenerator() {
  }

  MVNGenerator(const Vector<T>& mean, const Matrix<T>& cov, uint64_t seed = 0) {
    Init(mean, cov, seed);
  }

  void Init(const Vector<T>& mean, const Matrix<T>& cov, uint64_t seed = 0) {
    this->mean = mean;
    this->L = Eigen::LLT<Matrix<T>>(cov.selfadjointView<Eigen::Lower>()).matrixL();
    if (seed == 0) {
      seed = std::hash<long long>()(nano());
    } else {
      seed = std::hash<long long>()(seed);
    }
    rand.seed((MTRand::uint32*)&seed, 2);
  }

  Vector<T> Gen() {
    Vector<T> v = Vector<T>::Zero(mean.size());
    for (auto& x : v) {
      x = (T)rand.randNorm(0, 1);
    }
    return L.triangularView<Eigen::Lower>() * v + mean;
  }

 public:
  Vector<T> mean;
  Matrix<T> L;
  MTRand rand;
};

}  // namespace aha

#endif
