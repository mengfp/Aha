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
#include <string>
#include <Eigen/Dense>

#include "version.h"
#include "generator.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846264338327950288
#endif

namespace aha {

using Eigen::DiagonalMatrix;
using Eigen::Dynamic;
using Eigen::Infinity;
using Eigen::LLT;
using Eigen::Lower;
using Eigen::Map;
using Eigen::MatrixXd;
using Eigen::MatrixXf;
using Eigen::Success;
using Eigen::VectorXd;
using Eigen::VectorXf;

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
  mvn(const VectorXdRef& mu, const MatrixXdRef& sigma) {
    Initialize(mu, sigma);
  }

  // 获取维数
  int Dim() const {
    return (int)u.size();
  }

  // 初始化，Cholesky分解
  void Initialize(const VectorXdRef& mu, const MatrixXdRef& sigma) {
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
  double Evaluate(const VectorXdRef& x) const {
    assert(x.size() == u.size());
    auto n = u.size();
    return -0.5 * (l.triangularView<Lower>().solve(x - u).squaredNorm() +
                   n * log(2 * M_PI) + d(n - 1));
  }

  // 批量计算对数概率密度
  VectorXd BatchEvaluate(const MatrixXdRef& X) const {
    assert(X.rows() == u.size());
    auto n = u.size();
    return -0.5 * (l.triangularView<Lower>()
                     .solve(X.colwise() - u)
                     .colwise()
                     .squaredNorm()
                     .array() +
                   n * log(2 * M_PI) + d(n - 1));
  }

  // 快速批量计算对数概率密度
  VectorXd FastEvaluate(const MatrixXdRef& X) const {
    assert(X.rows() == u.size());
    auto n = u.size();
    auto _u = u.cast<float>();
    auto _l = l.cast<float>();
    return -0.5 * (_l.triangularView<Lower>()
                     .solve(X.cast<float>().colwise() - _u)
                     .colwise()
                     .squaredNorm()
                     .array()
                     .cast<double>() +
                   n * log(2 * M_PI) + d(n - 1));
  }

  // 计算对数边缘概率密度
  double PartialEvaluate(const VectorXdRef& x) const {
    assert(x.size() <= u.size());
    auto k = x.size();
    return -0.5 * (l.topLeftCorner(k, k)
                     .triangularView<Lower>()
                     .solve(x - u.head(k))
                     .squaredNorm() +
                   k * log(2 * M_PI) + d(k - 1));
  }

  // 计算对数边缘概率密度和条件期望
  double Predict(const VectorXdRef& x, VectorXd& y) const {
    assert(x.size() <= u.size());
    auto n = u.size();
    auto k = x.size();
    y.resize(n - k);
    VectorXd temp =
      l.topLeftCorner(k, k).triangularView<Lower>().solve(x - u.head(k));
    y = l.bottomLeftCorner(n - k, k) * temp + u.tail(n - k);
    return -0.5 * (temp.squaredNorm() + k * log(2 * M_PI) + d(k - 1));
  }

  // 批量计算对数边缘概率密度和条件期望
  VectorXd BatchPredict(const MatrixXdRef& X, MatrixXd& Y) const {
    assert(X.rows() <= u.size());
    auto n = u.size();
    auto k = X.rows();
    Y.resize(n - k, X.cols());
    MatrixXd temp = l.topLeftCorner(k, k).triangularView<Lower>().solve(
      X.colwise() - u.head(k));
    Y = (l.bottomLeftCorner(n - k, k) * temp).colwise() + u.tail(n - k);
    return -0.5 * (temp.colwise().squaredNorm().array() + k * log(2 * M_PI) +
                   d(k - 1));
  }

  // 快速计算对数边缘概率密度和条件期望
  VectorXd FastPredict(const MatrixXdRef& X, MatrixXd& Y) const {
    assert(X.rows() <= u.size());
    auto n = u.size();
    auto k = X.rows();
    Y.resize(n - k, X.cols());
    auto _u = u.cast<float>();
    auto _l = l.cast<float>();
    MatrixXf temp = _l.topLeftCorner(k, k).triangularView<Lower>().solve(
      X.cast<float>().colwise() - _u.head(k));
    Y = ((_l.bottomLeftCorner(n - k, k) * temp).colwise() + _u.tail(n - k))
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
  double Evaluate(const VectorXdRef& x, VectorXd& w) const {
    assert((int)x.size() == dim);
    assert((int)w.size() == rank);
    for (int i = 0; i < rank; i++) {
      w[i] = cores[i].Evaluate(x);
    }
    double wmax = w.maxCoeff();
    for (int i = 0; i < rank; i++) {
      w[i] = weights[i] * exp(w[i] - wmax);
    }
    double sum = w.sum();
    w.array() /= sum;
    return log(sum) + wmax;
  }

  // 批量计算对数概率密度和分类权重
  VectorXd BatchEvaluate(const MatrixXdRef& X, Ref<MatrixXd> W) const {
    assert((int)X.rows() == dim);
    assert(W.rows() == X.cols());
    assert((int)W.cols() == rank);
    for (int i = 0; i < rank; i++) {
      W.col(i).noalias() = cores[i].BatchEvaluate(X);
      W.col(i).array() += log(weights[i]);
    }
    VectorXd wmax = W.rowwise().maxCoeff();
    W = (W.colwise() - wmax).array().exp();
    VectorXd sum = W.rowwise().sum();
    W.array().colwise() /= sum.array();
    return sum.array().log() + wmax.array();
  }

  // 快速批量计算对数概率密度和分类权重
  VectorXd FastEvaluate(const MatrixXdRef& X, MatrixXd& W) const {
    assert((int)X.rows() == dim);
    assert(W.rows() == X.cols());
    assert((int)W.cols() == rank);
    for (int i = 0; i < rank; i++) {
      W.col(i) = cores[i].FastEvaluate(X);
    }
    VectorXd wmax = W.rowwise().maxCoeff();
    for (int i = 0; i < rank; i++) {
      W.col(i) = weights[i] * (W.col(i).array() - wmax.array()).exp();
    }
    VectorXd sum = W.rowwise().sum();
    W.array().colwise() /= sum.array();
    return sum.array().log() + wmax.array();
  }

  // 计算对数边缘概率密度和条件期望
  double Predict(const VectorXdRef& x, VectorXd& y) const {
    assert((int)x.size() <= dim);
    std::vector<VectorXd> v(rank);
    VectorXd w(rank);
    for (int i = 0; i < rank; i++) {
      w[i] = cores[i].Predict(x, v[i]);
    }
    double wmax = w.maxCoeff();
    for (int i = 0; i < rank; i++) {
      w[i] = weights[i] * exp(w[i] - wmax);
    }
    double sum = w.sum();
    y.resize(dim - x.size());
    y.setZero();
    for (int i = 0; i < rank; i++) {
      y += (w[i] / sum) * v[i];
    }
    return log(sum) + wmax;
  }

  // 批量计算对数边缘概率密度和条件期望
  VectorXd BatchPredict(const MatrixXdRef& X, MatrixXd& Y) const {
    assert((int)X.rows() <= dim);
    std::vector<MatrixXd> V(rank);
    MatrixXd W(X.cols(), rank);
    for (int i = 0; i < rank; i++) {
      W.col(i) = cores[i].BatchPredict(X, V[i]);
    }
    VectorXd wmax = W.rowwise().maxCoeff();
    for (int i = 0; i < rank; i++) {
      W.col(i) = weights[i] * (W.col(i) - wmax).array().exp();
    }
    VectorXd sum = W.rowwise().sum();
    W = W.array().colwise() / sum.array();
    Y.resize(dim - X.rows(), X.cols());
    Y.setZero();
    for (int i = 0; i < rank; i++) {
      Y += V[i] * DiagonalMatrix<double, Dynamic>(W.col(i));
    }
    return sum.array().log() + wmax.array();
  }

  // 快速计算对数边缘概率密度和条件期望
  VectorXd FastPredict(const MatrixXdRef& X, MatrixXd& Y) const {
    assert((int)X.rows() <= dim);
    std::vector<MatrixXd> V(rank);
    MatrixXd W(X.cols(), rank);
    for (int i = 0; i < rank; i++) {
      W.col(i) = cores[i].FastPredict(X, V[i]);
    }
    VectorXd wmax = W.rowwise().maxCoeff();
    for (int i = 0; i < rank; i++) {
      W.col(i) = weights[i] * (W.col(i) - wmax).array().exp();
    }
    VectorXd sum = W.rowwise().sum();
    W = W.array().colwise() / sum.array();
    Y.resize(dim - X.rows(), X.cols());
    Y.setZero();
    for (int i = 0; i < rank; i++) {
      Y += V[i] * DiagonalMatrix<double, Dynamic>(W.col(i));
    }
    return sum.array().log() + wmax.array();
  }

  // 计算条件期望和条件协方差
  double PredictEx(const VectorXdRef& x, VectorXd& y, MatrixXd& cov) const {
    assert((int)x.size() <= dim);
    std::vector<VectorXd> v(rank);
    VectorXd w(rank);
    for (int i = 0; i < rank; i++) {
      w[i] = cores[i].Predict(x, v[i]);
    }
    double wmax = w.maxCoeff();
    for (int i = 0; i < rank; i++) {
      w[i] = weights[i] * exp(w[i] - wmax);
    }
    double sum = w.sum();
    y.resize(dim - x.size());
    y.setZero();
    for (int i = 0; i < rank; i++) {
      y += (w[i] / sum) * v[i];
    }
    // 计算条件协方差
    cov.resize(y.size(), y.size());
    cov.setZero();
    for (int i = 0; i < rank; i++) {
      auto L = cores[i].getl().bottomRightCorner(y.size(), y.size());
      auto _v = v[i] - y;
      cov += (w[i] / sum) * (L * L.transpose());
      cov += (w[i] / sum) * (_v * _v.transpose());
    }
    return log(sum) + wmax;
  }

  // 批量计算条件期望和条件协方差
  VectorXd BatchPredictEx(const MatrixXdRef& X,
                          MatrixXd& Y,
                          MatrixXd& COV) const {
    assert((int)X.rows() <= dim);
    std::vector<MatrixXd> V(rank);
    MatrixXd W(X.cols(), rank);
    for (int i = 0; i < rank; i++) {
      W.col(i) = cores[i].BatchPredict(X, V[i]);
    }
    VectorXd wmax = W.rowwise().maxCoeff();
    for (int i = 0; i < rank; i++) {
      W.col(i) = weights[i] * (W.col(i) - wmax).array().exp();
    }
    VectorXd sum = W.rowwise().sum();
    W = W.array().colwise() / sum.array();
    Y.resize(dim - X.rows(), X.cols());
    Y.setZero();
    for (int i = 0; i < rank; i++) {
      Y += V[i] * DiagonalMatrix<double, Dynamic>(W.col(i));
    }
    // 计算条件协方差
    COV.resize(Y.rows(), Y.rows() * Y.cols());
    COV.setZero();
    for (int i = 0; i < rank; i++) {
      auto L = cores[i].getl().bottomRightCorner(Y.rows(), Y.rows());
      auto C = L * L.transpose();
      for (int j = 0; j < Y.cols(); j++) {
        auto _COV = COV.middleCols(COV.rows() * j, COV.rows());
        auto _V = V[i].col(j) - Y.col(j);
        _COV += W(j, i) * C;
        _COV += W(j, i) * (_V * _V.transpose());
      }
    }
    return sum.array().log() + wmax.array();
  }

  // 批量快速计算条件期望和条件协方差
  VectorXd FastPredictEx(const MatrixXdRef& X,
                         MatrixXd& Y,
                         MatrixXd& COV) const {
    assert((int)X.rows() <= dim);
    std::vector<MatrixXd> V(rank);
    MatrixXd W(X.cols(), rank);
    for (int i = 0; i < rank; i++) {
      W.col(i) = cores[i].FastPredict(X, V[i]);
    }
    VectorXd wmax = W.rowwise().maxCoeff();
    for (int i = 0; i < rank; i++) {
      W.col(i) = weights[i] * (W.col(i) - wmax).array().exp();
    }
    VectorXd sum = W.rowwise().sum();
    W = W.array().colwise() / sum.array();
    Y.resize(dim - X.rows(), X.cols());
    Y.setZero();
    for (int i = 0; i < rank; i++) {
      Y += V[i] * DiagonalMatrix<double, Dynamic>(W.col(i));
    }
    // 计算条件协方差
    COV.resize(Y.rows(), Y.rows() * Y.cols());
    COV.setZero();
    for (int i = 0; i < rank; i++) {
      auto L = cores[i].getl().bottomRightCorner(Y.rows(), Y.rows());
      auto C = L * L.transpose();
      for (int j = 0; j < Y.cols(); j++) {
        auto _COV = COV.middleCols(COV.rows() * j, COV.rows());
        auto _V = V[i].col(j) - Y.col(j);
        _COV += W(j, i) * C;
        _COV += W(j, i) * (_V * _V.transpose());
      }
    }
    return sum.array().log() + wmax.array();
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
      std::vector<double> mu(dim);
      Map<VectorXd>(mu.data(), dim) = cores[i].getu();
      std::vector<double> sigma(dim * dim);
      Map<MatrixXd>(sigma.data(), dim, dim) =
        cores[i].getl() * cores[i].getl().transpose();
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

  // 以二进制导出
  std::vector<char> Dump() const {
    if (!Initialized()) {
      return {};
    }
    const int magic = MAGIC;
    const int version = parse_version(VERSION);
    int size = sizeof(int) * 4 + sizeof(double) * rank * (1 + dim + dim * dim);
    std::vector<char> output(size);
    char* p = output.data();
    *(int*)p = magic;
    p += sizeof(int);
    *(int*)p = version;
    p += sizeof(int);
    *(int*)p = rank;
    p += sizeof(int);
    *(int*)p = dim;
    p += sizeof(int);
    memcpy(p, weights.data(), sizeof(double) * rank);
    p += sizeof(double) * rank;
    for (int i = 0; i < rank; i++) {
      Map<VectorXd>((double*)p, dim) = cores[i].getu();
      p += sizeof(double) * dim;
      Map<MatrixXd>((double*)p, dim, dim) =
        cores[i].getl() * cores[i].getl().transpose();
      p += sizeof(double) * dim * dim;
    }
    return output;
  }

  // 以二进制导入
  bool Load(const std::vector<char>& input) {
    if (input.size() < sizeof(int) * 4) {
      return false;
    }
    // 检查数据头
    const char* p = input.data();
    int magic = *(int*)p;
    if (magic != MAGIC) {
      return false;
    }
    p += sizeof(int) * 2;
    // 读取阶数和维数
    int r = *(int*)p;
    p += sizeof(int);
    int d = *(int*)p;
    p += sizeof(int);
    // 检查字节数
    if (input.size() !=
        sizeof(double) * (r + (d + d * d) * r) + sizeof(int) * 4) {
      return false;
    }
    // 加载模型
    rank = r;
    dim = d;
    weights.assign((double*)p, (double*)p + rank);
    p += sizeof(double) * rank;
    cores.resize(rank);
    for (int i = 0; i < rank; i++) {
      auto u = Map<VectorXd>((double*)p, dim);
      p += sizeof(double) * dim;
      auto s = Map<MatrixXd>((double*)p, dim, dim);
      p += sizeof(double) * dim * dim;
      cores[i].Initialize(u, s);
    }
    return true;
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
      covs(m.Rank()) {
  }

  // 添加一个样本
  void Train(const VectorXdRef& sample) {
    if (m.Initialized()) {
      VectorXd temp = VectorXd::Zero(rank);
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

  // 批量添加样本
  void __BatchTrain(const MatrixXdRef& samples) {
    if (m.Initialized()) {
      MatrixXd W = MatrixXd::Zero(samples.cols(), rank);
      entropy -= m.BatchEvaluate(samples, W).sum();
      for (int i = 0; i < rank; i++) {
        weights[i] += W.col(i).sum();
        means[i] += samples * W.col(i);
        MatrixXd m =
          samples.array().rowwise() * W.col(i).transpose().array().sqrt();
        covs[i].selfadjointView<Lower>().rankUpdate(m);
      }
    } else {
      MatrixXd quadric = MatrixXd::Zero(samples.rows(), samples.rows());
      quadric.selfadjointView<Lower>().rankUpdate(samples);
      for (int i = 0; i < rank; i++) {
        weights[i] += samples.cols();
        means[i] += samples.rowwise().sum();
        covs[i] += quadric.selfadjointView<Lower>();
      }
    }
  }

  void BatchTrain(const MatrixXdRef& samples) {
    const Eigen::Index N = samples.cols();
    const Eigen::Index D = samples.rows();
    const Eigen::Index blocksize = 16;  // 你验证出的 L1 缓存黄金分割点

    // --- 1. 预分配工作空间 (Workspace)，彻底避免循环内 malloc ---
    // W_buffer 用于存储 BatchEvaluate 的似然权重
    MatrixXd W_buffer(blocksize, rank);
    MatrixXd W_sqrt_buffer(blocksize, rank);
    // tmp_buffer 用于 M-Step 的矩阵投影计算
    MatrixXd tmp_buffer(D, blocksize);
    // quadric_buffer 用于初始化分支的协方差累加
    MatrixXd quadric_buffer(D, D);

    for (int i = 0; i < N; i += blocksize) {
      // 自动处理 N 不是 blocksize 整数倍的情况（尾部样本）
      auto cur_B = std::min(blocksize, N - i);
      auto block = samples.middleCols(i, cur_B);

      if (m.Initialized()) {
        auto W = W_buffer.topRows(cur_B);
        entropy -= m.BatchEvaluate(block, W).sum();

        // --- 绝杀 1: 预计算 sqrt，Workspace 复用 (定义在循环外) ---
        auto W_sqrt = W_sqrt_buffer.topRows(cur_B);
        W_sqrt.array() = W.array().sqrt();

        for (int k = 0; k < rank; k++) {
          weights[k] += W.col(k).sum();

          // --- 绝杀 2: 显式优化均值更新 ---
          means[k].noalias() += block.lazyProduct(W.col(k));

          auto tmp = tmp_buffer.leftCols(cur_B);
          // 直接乘预算好的 sqrt，彻底消灭循环内 sqrt 调用
          tmp.noalias() =
            (block.array().rowwise() * W_sqrt.col(k).transpose().array())
              .matrix();

          covs[k].selfadjointView<Lower>().rankUpdate(tmp);
        }
      }

      // if (m.Initialized()) {
      //   // --- 分支 A: 增量训练 (EM 步) ---
      //   auto W = W_buffer.topRows(cur_B);

      //  // 似然计算直接覆盖 W，无需 setZero
      //  entropy -= m.BatchEvaluate(block, W).sum();

      //  for (int k = 0; k < rank; k++) {
      //    // 1. 更新权重 (Scalar)
      //    weights[k] += W.col(k).sum();

      //    // 2. 更新均值 (Matrix-Vector)
      //    means[k].noalias() += block * W.col(k);

      //    // 3. 更新协方差 (Rank-1 Update)
      //    // 先计算加权投影，结果仅 25KB，稳在 L1 Cache
      //    auto tmp = tmp_buffer.leftCols(cur_B);
      //    tmp = (block.array().rowwise() *
      //    W.col(k).transpose().array().sqrt())
      //            .matrix();
      //    covs[k].selfadjointView<Lower>().rankUpdate(tmp);
      //  }
      //}
      //

      else {
        // --- 分支 B: 初始化阶段 (极致优化版) ---
        // 1. 预计算当前块的协方差贡献 (只算一次)
        quadric_buffer.setZero();
        quadric_buffer.selfadjointView<Lower>().rankUpdate(block);

        // 2. 预计算当前块的均值贡献 (只算一次)
        VectorXd block_sum = block.rowwise().sum();

        for (int k = 0; k < rank; k++) {
          weights[k] += cur_B;
          means[k].noalias() += block_sum;  // 直接累加预计算的结果
          covs[k] += quadric_buffer.selfadjointView<Lower>();
        }
      }
    }
  }

  // 快速批量添加样本
  void FastTrain(const MatrixXdRef& samples) {
    if (m.Initialized()) {
      MatrixXd W = MatrixXd::Zero(samples.cols(), rank);
      entropy -= m.FastEvaluate(samples, W).sum();
      for (int i = 0; i < rank; i++) {
        weights[i] += W.col(i).sum();
        means[i] += samples * W.col(i);
        MatrixXf quadric = MatrixXf::Zero(samples.rows(), samples.rows());
        MatrixXf m =
          (samples.array().rowwise() * W.col(i).transpose().array().sqrt())
            .cast<float>();
        quadric.selfadjointView<Lower>().rankUpdate(m);
        covs[i] += quadric.cast<double>().selfadjointView<Lower>();
      }
    } else {
      MatrixXd quadric = MatrixXd::Zero(samples.rows(), samples.rows());
      quadric.selfadjointView<Lower>().rankUpdate(samples);
      for (int i = 0; i < rank; i++) {
        weights[i] += samples.cols();
        means[i] += samples.rowwise().sum();
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
      std::vector<double> m(dim);
      Map<VectorXd>(m.data(), dim) = means[i];
      j["m"].push_back(m);
      std::vector<double> c(dim * dim);
      Map<MatrixXd>(c.data(), dim, dim) = covs[i];
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
        covs[i] +=
          (Map<MatrixXd>(c.data(), dim, dim) * w).selfadjointView<Lower>();
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
      means[i] *= (1.0 / weights[i]);
      covs[i] *= (1.0 / weights[i]);
      covs[i].selfadjointView<Lower>().rankUpdate(means[i], -1.0);
      covs[i].diagonal().array() += noise_floor * noise_floor;
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
    }
  }

  // 以二进制导出训练结果
  std::vector<char> Dump() const {
    int size = sizeof(int) * 2 + sizeof(double) +
               sizeof(double) * rank * (1 + dim + dim * dim);
    std::vector<char> output(size);
    char* p = output.data();
    *(int*)p = rank;
    p += sizeof(int);
    *(int*)p = dim;
    p += sizeof(int);
    *(double*)p = entropy;
    p += sizeof(double);
    memcpy(p, weights.data(), sizeof(double) * rank);
    p += sizeof(double) * rank;
    for (int i = 0; i < rank; i++) {
      Map<VectorXd>((double*)p, dim) = means[i];
      p += sizeof(double) * dim;
      Map<MatrixXd>((double*)p, dim, dim) = covs[i];
      p += sizeof(double) * dim * dim;
    }
    return output;
  }

  // 以二进制合并训练结果
  bool Load(const std::vector<char> input, double w = 1.0) {
    if (input.size() < sizeof(int) * 2) {
      return false;
    }
    // 检查阶数和维数
    const char* p = input.data();
    int r = *(int*)p;
    p += sizeof(int);
    int d = *(int*)p;
    p += sizeof(int);
    if (r != rank || d != dim) {
      return false;
    }
    // 检查字节数
    if (input.size() !=
        sizeof(double) * (1 + r + (d + d * d) * r) + sizeof(int) * 2) {
      return false;
    }
    // 合并结果
    entropy += *(double*)p * w;
    p += sizeof(double);
    for (int i = 0; i < rank; i++) {
      weights[i] += *(double*)p * w;
      p += sizeof(double);
    }
    for (int i = 0; i < rank; i++) {
      auto u = Map<VectorXd>((double*)p, dim);
      means[i] += u * w;
      p += sizeof(double) * dim;
      auto s = Map<MatrixXd>((double*)p, dim, dim);
      covs[i] += (s * w).selfadjointView<Lower>();
      p += sizeof(double) * dim * dim;
    }
    return true;
  }

 protected:
  mix& m;
  int rank;
  int dim;
  double entropy;
  std::vector<double> weights;
  std::vector<VectorXd> means;
  std::vector<MatrixXd> covs;
};

}  // namespace aha

#endif
