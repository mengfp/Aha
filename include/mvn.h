/*
** Copyright 2025 Meng, Fanping. All rights reserved.
*/
#ifndef AHA_MVN_H
#define AHA_MVN_H

#include <cassert>
#include <cfloat>
#include <iostream>
#include <nlohmann/json.hpp>
#include <vector>
#include <Eigen/Dense>

#include "generator.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846264338327950288
#endif

namespace aha {

using namespace Eigen;

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
  mvn(const VectorXd& mu, const MatrixXd& sigma) {
    Initialize(mu, sigma);
  }

  // 获取维数
  int Dim() const {
    return (int)u.size();
  }

  // 初始化，Cholesky分解
  void Initialize(const VectorXd& mu, const MatrixXd& sigma) {
    assert(mu.size() == sigma.rows());
    assert(sigma.rows() == sigma.cols());
    auto llt = LLT<MatrixXd>(sigma.selfadjointView<Lower>());
    assert(llt.info() == Success);
    u = mu;
    l = llt.matrixL();
    d = VectorXd(mu.size());
    double c = 0;
    for (int i = 0; i < (int)mu.size(); i++) {
      c += 2 * log(l(i, i));
      d(i) = c;
    }
  }

  // 计算对数概率密度
  double Evaluate(const VectorXd& x) const {
    assert(x.size() == u.size());
    auto n = u.size();
    return -0.5 * (l.triangularView<Lower>().solve(x - u).squaredNorm() +
                   n * log(2 * M_PI) + d(n - 1));
  }

  // 计算对数边缘概率密度
  double PartialEvaluate(const VectorXd& x) const {
    assert(x.size() <= u.size());
    auto k = x.size();
    return -0.5 * (l.topLeftCorner(k, k)
                     .triangularView<Lower>()
                     .solve(x - u.head(k))
                     .squaredNorm() +
                   k * log(2 * M_PI) + d(k - 1));
  }

  // 计算对数边缘概率密度和条件期望
  double Predict(const VectorXd& x, VectorXd& y) const {
    assert(x.size() <= u.size());
    auto n = u.size();
    auto k = x.size();
    VectorXd temp =
      l.topLeftCorner(k, k).triangularView<Lower>().solve(x - u.head(k));
    y = l.bottomLeftCorner(n - k, k) * temp + u.tail(n - k);
    return -0.5 * (temp.squaredNorm() + k * log(2 * M_PI) + d(k - 1));
  }

  // 批量计算对数边缘概率密度和条件期望
  VectorXd BatchPredict(const MatrixXd& X, MatrixXd& Y) const {
    assert(X.rows() <= u.size());
    auto n = u.size();
    auto k = X.rows();
    MatrixXd temp = l.topLeftCorner(k, k).triangularView<Lower>().solve(
      X.colwise() - u.head(k));
    Y = (l.bottomLeftCorner(n - k, k) * temp).colwise() + u.tail(n - k);
    return -0.5 * (temp.colwise().squaredNorm().array() + k * log(2 * M_PI) +
                   d(k - 1));
  }

  // 快速计算对数边缘概率密度和条件期望
  VectorXd FastPredict(const MatrixXd& X, MatrixXd& Y) const {
    assert(X.rows() <= u.size());
    auto n = u.size();
    auto k = X.rows();
    MatrixXf temp =
      l.cast<float>().topLeftCorner(k, k).triangularView<Lower>().solve(
        X.cast<float>().colwise() - u.cast<float>().head(k));
    Y = ((l.cast<float>().bottomLeftCorner(n - k, k) * temp).colwise() +
         u.cast<float>().tail(n - k))
          .cast<double>();
    return -0.5 * (temp.cast<double>().colwise().squaredNorm().array() +
                   k * log(2 * M_PI) + d(k - 1));
  }

 public:
  const VectorXd& getu() const {
    return u;
  }

  const MatrixXd& getl() const {
    return l;
  }

 protected:
  VectorXd u;
  MatrixXd l;
  VectorXd d;
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
                  const std::vector<VectorXd>& means,
                  const std::vector<MatrixXd>& covs) {
    rank = (int)weights.size();
    dim = means.size() > 0 ? (int)means[0].size() : 0;
    this->weights = weights;
    cores.resize(rank);
    for (int i = 0; i < rank; i++) {
      cores[i].Initialize(means[i], covs[i]);
    }
  }

  // 计算对数概率密度和分类权重
  double _Evaluate(const VectorXd& x, std::vector<double>& w) const {
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

  // 计算对数概率密度和分类权重
  double Evaluate(const VectorXd& x, std::vector<double>& w) const {
    assert((int)x.size() == dim);
    assert((int)w.size() == rank);
    double sum = 0;
    for (int i = 0; i < rank; i++) {
      w[i] = weights[i] * exp(cores[i].Evaluate(x));
      sum += w[i];
    }
    for (int i = 0; i < rank; i++) {
      w[i] /= sum;
    }
    return log(sum);
  }

  // 计算对数边缘概率密度和条件期望
  double _Predict(const VectorXd& x, VectorXd& y) const {
    assert((int)x.size() <= dim);
    double wmax = -DBL_MAX;
    std::vector<VectorXd> v(rank);
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
    y = VectorXd::Zero(dim - x.size());
    for (int i = 0; i < rank; i++) {
      y += (w[i] / sum) * v[i];
    }
    return log(sum) + wmax;
  }

  // 计算对数边缘概率密度和条件期望
  double Predict(const VectorXd& x, VectorXd& y) const {
    assert((int)x.size() <= dim);
    std::vector<VectorXd> v(rank);
    std::vector<double> w(rank);
    double sum = 0;
    for (int i = 0; i < rank; i++) {
      w[i] = weights[i] * exp(cores[i].Predict(x, v[i]));
      sum += w[i];
    }
    y = VectorXd::Zero(dim - x.size());
    for (int i = 0; i < rank; i++) {
      y += (w[i] / sum) * v[i];
    }
    return log(sum);
  }

  // 批量计算对数边缘概率密度和条件期望
  VectorXd BatchPredict(const MatrixXd& X, MatrixXd& Y) const {
    assert((int)X.rows() <= dim);
    std::vector<MatrixXd> V(rank);
    MatrixXd W(X.cols(), rank);
    for (int i = 0; i < rank; i++) {
      W.col(i) = weights[i] * cores[i].BatchPredict(X, V[i]).array().exp();
    }
    VectorXd sum = W.rowwise().sum();
    W = W.array().colwise() / sum.array();
    Y = MatrixXd::Zero(dim - X.rows(), X.cols());
    for (int i = 0; i < rank; i++) {
      Y += V[i] * DiagonalMatrix<double, Dynamic>(W.col(i));
    }
    return sum.array().log();
  }

  // 快速计算对数边缘概率密度和条件期望
  VectorXd FastPredict(const MatrixXd& X, MatrixXd& Y) const {
    assert((int)X.rows() <= dim);
    std::vector<MatrixXd> V(rank);
    MatrixXd W(X.cols(), rank);
    for (int i = 0; i < rank; i++) {
      W.col(i) = weights[i] * cores[i].FastPredict(X, V[i]).array().exp();
    }
    VectorXd sum = W.rowwise().sum();
    W = W.array().colwise() / sum.array();
    Y = MatrixXd::Zero(dim - X.rows(), X.cols());
    for (int i = 0; i < rank; i++) {
      Y += V[i] * DiagonalMatrix<double, Dynamic>(W.col(i));
    }
    return sum.array().log();
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
      MatrixXd s = l * l.transpose();
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
        auto u = Map<VectorXd>(mu.data(), d);
        auto s = Map<MatrixXd>(sigma.data(), d, d);
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
  }

  // 添加一个样本
  void Train(const VectorXd& sample) {
    if (m.Initialized()) {
      entropy -= m.Evaluate(sample, temp);
      MatrixXd quadric = (sample * sample.transpose()).selfadjointView<Lower>();
      for (int i = 0; i < rank; i++) {
        weights[i] += temp[i];
        means[i] += sample * temp[i];
        covs[i] += (quadric * temp[i]).selfadjointView<Lower>();
      }
    } else {
      MatrixXd quadric = (sample * sample.transpose()).selfadjointView<Lower>();
      for (int i = 0; i < rank; i++) {
        weights[i] += 1.0;
        means[i] += sample;
        covs[i] += quadric.selfadjointView<Lower>();
      }
    }
  }

  // 合并两个训练器（w为样本权重）
  bool Merge(const trainer& t, double w = 1.0) {
    if (t.rank != rank) {
      return false;
    }
    if (t.dim != dim) {
      return false;
    }
    entropy += t.entropy * w;
    for (int i = 0; i < rank; i++) {
      weights[i] += t.weights[i] * w;
      means[i] += t.means[i] * w;
      covs[i] += (t.covs[i] * w).selfadjointView<Lower>();
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

  // 合并训练结果（w为样本权重）
  bool Swallow(const std::string& s, double w = 1.0) {
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
      entropy += (double)j["e"] * w;
      for (int i = 0; i < rank; i++) {
        weights[i] += (double)j["w"][i] * w;
        std::vector<double> m = j["m"][i];
        if ((int)m.size() != dim) {
          return false;
        }
        means[i] += Map<VectorXd>(m.data(), dim) * w;
        std::vector<double> c = j["c"][i];
        if ((int)c.size() != dim * dim) {
          return false;
        }
        covs[i] += Map<MatrixXd>(c.data(), dim, dim) * w;
      }
      return true;
    } catch (...) {
      return false;
    }
  }

  // 更新模型（对角线加载为可选项）
  double Update(double noise_floor = 0.0) {
    double s = 0;
    for (auto& w : weights) {
      s += w;
    }
    entropy /= s;
    for (int i = 0; i < rank; i++) {
      means[i] /= weights[i];
      covs[i] = (covs[i] / weights[i] - means[i] * means[i].transpose())
                  .selfadjointView<Lower>();
      covs[i] += MatrixXd::Identity(dim, dim) * noise_floor * noise_floor;
      weights[i] /= s;
    }
    if (!m.Initialized() && rank > 0) {
      // 随机初始化
      MVNGenerator gen(means[0], covs[0]);
      for (int i = 0; i < rank; i++) {
        means[i] = gen.Gen();
      }
      entropy = std::numeric_limits<double>::infinity();
    }
    m.Initialize(weights, means, covs);
    return entropy;
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

  // 清空记忆
  void Reset() {
    entropy = 0;
    for (int i = 0; i < rank; i++) {
      weights[i] = 0;
      means[i] = VectorXd::Zero(dim);
      covs[i] = MatrixXd::Zero(dim, dim);
      temp[i] = 0;
    }
  }

 protected:
  mix& m;
  int rank;
  int dim;
  double entropy;
  std::vector<double> weights;
  std::vector<VectorXd> means;
  std::vector<MatrixXd> covs;
  std::vector<double> temp;
};

}  // namespace aha

#endif
