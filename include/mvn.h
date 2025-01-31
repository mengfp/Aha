#ifndef AHA_MVN_H
#define AHA_MVN_H

#include <Eigen/Dense>
#include <cassert>
#include <cfloat>
#include <iostream>
#include <nlohmann/json.hpp>
#include <vector>

#include "generator.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846264338327950288
#endif

using namespace Eigen;
#ifndef Vector
#define Vector VectorXd
#endif
#ifndef Matrix
#define Matrix MatrixXd
#endif

using json = nlohmann::ordered_json;

/*
** 多元正态分布
*/
class mvn {
 public:
  // 构造函数
  mvn() {
  }

  // 构造函数
  mvn(const Vector& mu, const Matrix& sigma) {
    Initialize(mu, sigma);
  }

  // 获取维数
  int Dim() const {
    return (int)u.size();
  }

  // 初始化，Cholesky分解
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

  // 计算对数概率密度
  double Evaluate(const Vector& x) const {
    assert(x.size() == u.size());
    auto n = u.size();
    return -0.5 * (l.triangularView<Lower>().solve(x - u).squaredNorm() +
                   n * log(2 * M_PI) + d(n - 1));
  }

  // 计算对数边缘概率密度
  double PartialEvaluate(const Vector& x) const {
    assert(x.size() <= u.size());
    auto k = x.size();
    return -0.5 * (l.topLeftCorner(k, k)
                     .triangularView<Lower>()
                     .solve(x - u.head(k))
                     .squaredNorm() +
                   k * log(2 * M_PI) + d(k - 1));
  }

  // 计算对数边缘概率密度和条件期望
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

/*
** 混合正态分布
*/
class mix {
 public:
  // 构造函数
  mix(int rank = 0, int dim = 0) : rank(rank), dim(dim) {
  }

  // 获取初始化状态
  bool Initialized() const {
    return cores.size() > 0;
  }

  // 获取阶数
  int Rank() const {
    return rank;
  }

  // 获取维数
  int Dim() const {
    return dim;
  }

  // 获取权重
  std::vector<double> GetWeights() const {
    return weights;
  }

  // 获取内核
  std::vector<mvn> GetCores() const {
    return cores;
  }

  // 初始化
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

  // 计算对数概率密度和分类权重
  double Evaluate(const Vector& x, std::vector<double>& w) const {
    assert((int)x.size() == dim);
    assert((int)w.size() == rank);
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

  // 计算对数边缘概率密度和条件期望
  double Predict(const Vector& x, Vector& y) const {
    assert((int)x.size() <= dim);
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

  // 导出Json字符串
  std::string Export() const {
    json j;
    if (!Initialized()) {
      return "*** not initialized ***";
    }
    j["r"] = rank;
    j["d"] = dim;
    j["w"] = weights;
    j["c"] = {};
    for (int i = 0; i < rank; i++) {
      auto& u = cores[i].getu();
      auto& l = cores[i].getl();
      Matrix s = l * l.transpose();
      std::vector<double> mu(u.data(), u.data() + u.size());
      std::vector<double> sigma(s.data(), s.data() + s.size());
      j["c"].push_back({{"u", mu}, {"s", sigma}});
    }
    return j.dump();
  }

  // 导入Json字符串
  bool Import(const std::string& model) {
    try {
      auto j = nlohmann::json::parse(model);
      int r = j["r"];
      int d = j["d"];
      if ((int)j["w"].size() != r) {
        return false;
      }
      if ((int)j["c"].size() != r) {
        return false;
      }
      std::vector<double> w = j["w"];
      std::vector<mvn> c(r);
      for (int i = 0; i < r; i++) {
        std::vector<double> mu = j["c"][i]["u"];
        std::vector<double> sigma = j["c"][i]["s"];
        if ((int)mu.size() != d || (int)sigma.size() != d * d) {
          return false;
        }
        auto u = Map<Vector>(mu.data(), d);
        auto s = Map<Matrix>(sigma.data(), d, d);
        c[i].Initialize(u, s);
      }
      rank = r;
      dim = d;
      weights = w;
      cores = c;
      return true;
    } catch (...) {
      return false;
    }
  }

  // 按权重从大到小笨拙排序
  void Sort() {
    for (int i = 0; i < rank - 1; i++) {
      for (int j = i + 1; j < rank; j++) {
        if (weights[i] < weights[j]) {
          // 交换位置
          std::swap(weights[i], weights[j]);
          std::swap(cores[i], cores[j]);
        }
      }
    }
  }

  void Print() {
    for (int i = 0; i < rank; i++) {
      std::cout << i << ": " << weights[i] << "\n";
      std::cout << "u:\n" << cores[i].getu() << "\n";
      std::cout << "s:\n"
                << cores[i].getl() * cores[i].getl().transpose() << "\n\n";
    }
  }

 protected:
  int rank;
  int dim;
  std::vector<double> weights;
  std::vector<mvn> cores;
};

/*
** 模型训练器
*/
class trainer {
 public:
  // 构造函数
  trainer(mix& m)
    : m(m),
      rank(m.Rank()),
      dim(m.Dim()),
      entropy(0),
      weights(m.Rank()),
      means(m.Rank()),
      covs(m.Rank()),
      temp(m.Rank()) {
    Reset();
  }

  // 添加一个样本
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

  // 合并两个训练器
  bool Merge(const trainer& t) {
    if (t.rank != rank) {
      return false;
    }
    if (t.dim != dim) {
      return false;
    }
    entropy += t.entropy;
    for (int i = 0; i < rank; i++) {
      weights[i] += t.weights[i];
      means[i] += t.means[i];
      covs[i] += (t.covs[i]).selfadjointView<Lower>();
    }
    return true;
  }

  // 导出训练结果
  std::string Spit() const {
    json j;
    j["r"] = rank;
    j["d"] = dim;
    j["e"] = entropy;
    j["w"] = weights;
    j["m"] = {};
    j["c"] = {};
    for (int i = 0; i < rank; i++) {
      std::vector<double> m(means[i].data(), means[i].data() + means[i].size());
      j["m"].push_back(m);
      std::vector<double> c(covs[i].data(), covs[i].data() + covs[i].size());
      j["c"].push_back(c);
    }
    return j.dump();
  }

  // 合并训练结果
  bool Swallow(const std::string& s) {
    try {
      auto j = nlohmann::json::parse(s);
      if ((int)j["r"] != rank) {
        return false;
      }
      if ((int)j["d"] != dim) {
        return false;
      }
      if ((int)j["w"].size() != rank) {
        return false;
      }
      if ((int)j["m"].size() != rank) {
        return false;
      }
      if ((int)j["c"].size() != rank) {
        return false;
      }
      entropy += (double)j["e"];
      for (int i = 0; i < rank; i++) {
        weights[i] += (double)j["w"][i];
        std::vector<double> m = j["m"][i];
        if ((int)m.size() != dim) {
          return false;
        }
        means[i] += Map<Vector>(m.data(), dim);
        std::vector<double> c = j["c"][i];
        if ((int)c.size() != dim * dim) {
          return false;
        }
        covs[i] += Map<Matrix>(c.data(), dim, dim);
      }
      return true;
    } catch (...) {
      return false;
    }
  }

  // 更新模型
  double Update() {
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
      // 随机初始化
      MVNGenerator gen(means[0], covs[0]);
      for (int i = 0; i < rank; i++) {
        means[i] = gen.Gen();
        Vector diagonal = covs[i].diagonal();
        covs[i] = diagonal.asDiagonal();
      }
    }
    m.Initialize(weights, means, covs);
    auto ret = entropy;
    Reset();
    return ret;
  }

  // 获取阶数
  int Rank() const {
    return rank;
  }

  // 获取维数
  int Dim() const {
    return dim;
  }

  // 输出
  void Print() {
    for (int i = 0; i < rank; i++) {
      std::cout << i << ": " << weights[i] << "\n";
      std::cout << "m:\n" << means[i] << "\n";
      std::cout << "s:\n" << covs[i] << "\n";
    }
  }

 protected:
  // 清空记忆
  void Reset() {
    entropy = 0;
    for (int i = 0; i < rank; i++) {
      weights[i] = 0;
      means[i] = Vector::Zero(dim);
      covs[i] = Matrix::Zero(dim, dim);
      temp[i] = 0;
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
