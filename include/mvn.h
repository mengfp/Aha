/*
** Copyright 2025 Meng, Fanping. All rights reserved.
*/
#ifndef AHA_MVN_H
#define AHA_MVN_H

#include <cassert>
#include <cfloat>
#include <iostream>
#include <nlohmann/json.hpp>
#include <numeric>
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
using Eigen::Ref;
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
  mvn(Ref<const VectorXd> mu, Ref<const MatrixXd> sigma) {
    Initialize(mu, sigma);
  }

  // 获取维数
  int Dim() const {
    return (int)u.size();
  }

  // 初始化，Cholesky分解
  bool Initialize(Ref<const VectorXd> mu, Ref<const MatrixXd> sigma) {
    assert(mu.size() > 0);
    assert(mu.size() == sigma.rows());
    assert(sigma.rows() == sigma.cols());
    auto llt = LLT<MatrixXd>(sigma.selfadjointView<Lower>());
    if (llt.info() == Success) {
      ill = false;
      u = mu;
      l = llt.matrixL();
      int dim = (int)mu.size();
      c.resize(dim);
      double d = 0.0;
      for (int i = 0; i < dim; i++) {
        d += 2 * std::log(l(i, i));
        c(i) = -0.5 * ((i + 1) * std::log(2 * M_PI) + d);
      }
      return true;
    } else {
      ill = true;
      return false;
    }
  }

  // 计算对数概率密度
  double Evaluate(Ref<const VectorXd> x) const {
    assert(x.size() == u.size());
    return -0.5 * l.triangularView<Lower>().solve(x - u).squaredNorm() +
           c(c.size() - 1);
  }

  // 批量计算对数概率密度
  VectorXd BatchEvaluate(Ref<const MatrixXd> X) const {
    assert(X.rows() == u.size());
    return -0.5 * l.triangularView<Lower>()
                    .solve(X.colwise() - u)
                    .colwise()
                    .squaredNorm()
                    .array() +
           c(c.size() - 1);
  }

  // 批量计算对数概率密度（单精度）
  VectorXd FastEvaluate(Ref<const MatrixXf> X) const {
    assert(X.rows() == u.size());
    return -0.5 * l.cast<float>()
                    .triangularView<Lower>()
                    .solve(X.colwise() - u.cast<float>())
                    .colwise()
                    .squaredNorm()
                    .array()
                    .cast<double>() +
           c(c.size() - 1);
  }

  // 计算对数边缘概率密度
  double PartialEvaluate(Ref<const VectorXd> x) const {
    assert(x.size() <= u.size());
    auto k = x.size();
    return -0.5 * l.topLeftCorner(k, k)
                    .triangularView<Lower>()
                    .solve(x - u.head(k))
                    .squaredNorm() +
           c(k - 1);
  }

  // 计算对数边缘概率密度和条件期望
  double Predict(Ref<const VectorXd> x, Ref<VectorXd> y) const {
    assert(x.size + y.size() == u.size());
    auto n = u.size();
    auto k = x.size();
    VectorXd temp =
      l.topLeftCorner(k, k).triangularView<Lower>().solve(x - u.head(k));
    y.noalias() = l.bottomLeftCorner(n - k, k) * temp + u.tail(n - k);
    return -0.5 * temp.squaredNorm() + c(k - 1);
  }

  // 批量计算对数边缘概率密度和条件期望
  VectorXd BatchPredict(Ref<const MatrixXd> X, Ref<MatrixXd> Y) const {
    assert(X.cols() == Y.cols());
    assert(X.rows() + Y.rows() == u.size());
    auto n = u.size();
    auto k = X.rows();
    MatrixXd temp = l.topLeftCorner(k, k).triangularView<Lower>().solve(
      X.colwise() - u.head(k));
    Y.noalias() =
      (l.bottomLeftCorner(n - k, k) * temp).colwise() + u.tail(n - k);
    return -0.5 * temp.colwise().squaredNorm().array() + c(k - 1);
  }

  // 批量计算对数边缘概率密度和条件期望（单精度）
  VectorXd FastPredict(Ref<const MatrixXd> X, MatrixXd& Y) const {
    assert(X.rows() <= u.size());
    auto n = u.size();
    auto k = X.rows();
    MatrixXd temp = l.topLeftCorner(k, k).triangularView<Lower>().solve(
      X.colwise() - u.head(k));
    Y.noalias() =
      (l.bottomLeftCorner(n - k, k) * temp).colwise() + u.tail(n - k);
    return -0.5 * temp.colwise().squaredNorm().array() + c(k - 1);
  }

  // 批量计算对数边缘概率密度和条件期望（单精度）
  VectorXd _FastPredict(Ref<const MatrixXf> X, MatrixXf& Y) const {
    assert(X.rows() <= u.size());
    auto n = u.size();
    auto k = X.rows();
    MatrixXf temp =
      l.topLeftCorner(k, k).cast<float>().triangularView<Lower>().solve(
        X.colwise() - u.head(k).cast<float>());
    Y.noalias() =
      ((l.bottomLeftCorner(n - k, k).cast<float>() * temp).colwise() +
       u.tail(n - k).cast<float>());
    return -0.5 * temp.cast<double>().colwise().squaredNorm().array() +
           c(k - 1);
  }

 public:
  const VectorXd& get_u() const {
    return u;
  }

  const MatrixXd& get_l() const {
    return l;
  }

  bool is_ill() const {
    return ill;
  }

 protected:
  VectorXd u;
  MatrixXd l;
  VectorXd c;
  bool ill;
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
    return weights.size() > 0;
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
  const VectorXd& GetWeights() const {
    return weights;
  }

  // 获取内核
  const std::vector<mvn>& GetCores() const {
    return cores;
  }

  // 初始化
  void Initialize(Ref<const VectorXd> weights,
                  Ref<const MatrixXd> means,
                  Ref<const MatrixXd> covs) {
    rank = (int)weights.size();
    dim = (int)means.rows();
    assert(rank > 0 && dim > 0);
    assert(means.cols() == rank);
    assert(covs.rows() == dim);
    assert(covs.cols() == dim * rank);
    this->weights = weights;
    cores.resize(rank);
    for (int i = 0; i < rank; i++) {
      cores[i].Initialize(means.col(i), covs.middleCols(dim * i, dim));
    }
  }

  // 计算对数概率密度和分类权重
  double Evaluate(Ref<const VectorXd> x, VectorXd& w) const {
    assert((int)x.size() == dim);
    assert((int)w.size() == rank);
    for (int i = 0; i < rank; i++) {
      w(i) = cores[i].Evaluate(x);
    }
    w.array() += weights.array().log();
    double wmax = w.maxCoeff();
    w = (w.array() - wmax).exp();
    double sum = w.sum();
    w /= sum;
    return std::log(sum) + wmax;
  }

  // 批量计算对数概率密度和分类权重
  VectorXd BatchEvaluate(Ref<const MatrixXd> X, Ref<MatrixXd> W) const {
    assert((int)X.rows() == dim);
    assert(W.rows() == X.cols());
    assert((int)W.cols() == rank);
    for (int i = 0; i < rank; i++) {
      W.col(i) = cores[i].BatchEvaluate(X);
    }
    W.array().rowwise() += weights.transpose().array().log();
    VectorXd wmax = W.rowwise().maxCoeff();
    W = (W.colwise() - wmax).array().exp();
    VectorXd sum = W.rowwise().sum();
    W.array().colwise() /= sum.array();
    return sum.array().log() + wmax.array();
  }

  // 批量计算对数概率密度和分类权重（单精度）
  VectorXd FastEvaluate(Ref<const MatrixXf> X, MatrixXd& W) const {
    assert((int)X.rows() == dim);
    assert(W.rows() == X.cols());
    assert((int)W.cols() == rank);
    for (int i = 0; i < rank; i++) {
      W.col(i) = cores[i].FastEvaluate(X);
    }
    W.array().rowwise() += weights.transpose().array().log();
    VectorXd wmax = W.rowwise().maxCoeff();
    W = (W.colwise() - wmax).array().exp();
    VectorXd sum = W.rowwise().sum();
    W.array().colwise() /= sum.array();
    return sum.array().log() + wmax.array();
  }

  // 计算对数边缘概率密度和条件期望
  double Predict(Ref<const VectorXd> x, Ref<VectorXd> y) const {
    assert((int)x.size() + (int)y.size() == dim);
    VectorXd w = VectorXd::Zero(rank);
    MatrixXd Y = MatrixXd::Zero(y.size(), rank);
    for (int i = 0; i < rank; i++) {
      w(i) = cores[i].Predict(x, Y.col(i));
    }
    w.array() += weights.array().log();
    double wmax = w.maxCoeff();
    w = (w.array() - wmax).exp();
    double sum = w.sum();
    y = Y * (w / sum);
    return std::log(sum) + wmax;
  }

  // TODO: 批量计算对数边缘概率密度和条件期望
  VectorXd BatchPredict(Ref<const MatrixXd> X, MatrixXd& Y) const {
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

  // TODO: 快速计算对数边缘概率密度和条件期望
  VectorXd FastPredict(Ref<const MatrixXd> X, MatrixXd& Y) const {
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

  // TODO: 计算条件期望和条件协方差
  double PredictEx(Ref<const VectorXd> x, VectorXd& y, MatrixXd& cov) const {
    assert((int)x.size() <= dim);
    std::vector<VectorXd> v(rank);
    VectorXd w(rank);
    for (int i = 0; i < rank; i++) {
      w[i] = cores[i].Predict(x, v[i]);
    }
    double wmax = w.maxCoeff();
    w = weights.array() * (w.array() - wmax).exp();
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
      auto L = cores[i].get_l().bottomRightCorner(y.size(), y.size());
      auto _v = v[i] - y;
      cov += (w[i] / sum) * (L * L.transpose());
      cov += (w[i] / sum) * (_v * _v.transpose());
    }
    return std::log(sum) + wmax;
  }

  // TODO: 批量计算条件期望和条件协方差
  VectorXd BatchPredictEx(Ref<const MatrixXd> X,
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
      auto L = cores[i].get_l().bottomRightCorner(Y.rows(), Y.rows());
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

  // TODO: 批量快速计算条件期望和条件协方差
  VectorXd FastPredictEx(Ref<const MatrixXd> X,
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
      auto L = cores[i].get_l().bottomRightCorner(Y.rows(), Y.rows());
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
    j["w"] = std::vector<double>(weights.begin(), weights.end());
    j["c"] = {};
    for (int i = 0; i < rank; i++) {
      auto& u = cores[i].get_u();
      auto& l = cores[i].get_l();
      std::vector<double> mu(u.begin(), u.end());
      std::vector<double> sigma(dim * dim);
      Map<MatrixXd>(sigma.data(), dim, dim) = l * l.transpose();
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
      weights = Map<const VectorXd>(w.data(), w.size());
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
      std::cout << "u:\n" << cores[i].get_u() << "\n";
      std::cout << "s:\n"
                << cores[i].get_l() * cores[i].get_l().transpose() << "\n\n";
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
      memcpy(p, cores[i].get_u().data(), sizeof(double) * dim);
      p += sizeof(double) * dim;
      Map<MatrixXd>((double*)p, dim, dim) =
        cores[i].get_l() * cores[i].get_l().transpose();
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
    if (*(int*)p != MAGIC) {
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
    weights = Map<const VectorXd>((double*)p, rank);
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
  VectorXd weights;
  std::vector<mvn> cores;
};

/*
** 模型训练器
*/
class trainer {
 public:
  // 构造函数
  trainer(mix& m) : m(m), rank(m.Rank()), dim(m.Dim()) {
    assert(rank > 0 && dim > 0);
    Reset();
  }

  // 添加一个样本
  void Train(Ref<const VectorXd> sample) {
    assert(sample.size() == dim);
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
    if (m.Initialized()) {
      VectorXd temp = VectorXd::Zero(rank);
      entropy -= m.Evaluate(sample, temp);
      weights += temp;
      means += sample * temp.transpose();
      for (int i = 0; i < rank; i++) {
        covs.middleCols(dim * i, dim)
          .selfadjointView<Lower>()
          .rankUpdate(sample, temp(i));
      }
    } else {
      weights(0) += 1.0;
      means.col(0) += sample;
      covs.leftCols(dim).selfadjointView<Lower>().rankUpdate(sample);
    }
  }

  // 批量添加样本
  void BatchTrain(Ref<const MatrixXd> samples) {
    assert(samples.rows() == dim);
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
    if (m.Initialized()) {
      MatrixXd W = MatrixXd::Zero(samples.cols(), rank);
      entropy -= m.BatchEvaluate(samples, W).sum();
      weights += W.colwise().sum();
      means += samples * W;
      MatrixXd temp = MatrixXd::Zero(samples.rows(), samples.cols());
      for (int i = 0; i < rank; i++) {
        temp = samples.array().rowwise() * W.col(i).transpose().array().sqrt();
        covs.middleCols(dim * i, dim).selfadjointView<Lower>().rankUpdate(temp);
      }
    } else {
      weights(0) += samples.cols();
      means.col(0) += samples.rowwise().sum();
      covs.leftCols(dim).selfadjointView<Lower>().rankUpdate(samples);
    }
  }

  // 批量添加样本（单精度）
  void FastTrain(Ref<const MatrixXf> samples) {
    assert(samples.rows() == dim);
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
    if (m.Initialized()) {
      MatrixXd W = MatrixXd::Zero(samples.cols(), rank);
      entropy -= m.FastEvaluate(samples, W).sum();
      weights += W.colwise().sum();
      means += (samples * W.cast<float>()).cast<double>();
      for (int i = 0; i < rank; i++) {
        MatrixXf temp = samples.array().rowwise() *
                        W.col(i).transpose().array().sqrt().cast<float>();
        covs.middleCols(dim * i, dim).triangularView<Lower>() +=
          (temp * temp.transpose()).cast<double>();
      }
    } else {
      weights(0) += samples.cols();
      means.col(0) += samples.cast<double>().rowwise().sum();
      covs.leftCols(dim).selfadjointView<Lower>().rankUpdate(
        samples.cast<double>());
    }
  }

  // 合并两个训练器（w为样本权重）
  bool Merge(const trainer& t, double w = 1.0) {
    if (t.rank != rank || t.dim != dim) {
      return false;
    }
    entropy += w * t.entropy;
    weights += w * t.weights;
    means += w * t.means;
    covs += w * t.covs;
    return true;
  }

  // 导出训练结果
  std::string Spit() const {
    json j;
    j["r"] = rank;
    j["d"] = dim;
    j["e"] = entropy;
    j["w"] = std::vector<double>(weights.begin(), weights.end());
    j["m"] = {};
    j["c"] = {};
    for (int i = 0; i < rank; i++) {
      std::vector<double> m(means.col(i).begin(), means.col(i).end());
      j["m"].push_back(m);
      std::vector<double> c(dim * dim);
      Map<MatrixXd>(c.data(), dim, dim) =
        covs.middleCols(dim * i, dim).selfadjointView<Lower>();
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
      entropy += w * (double)j["e"];
      for (int i = 0; i < rank; i++) {
        weights(i) += w * (double)j["w"][i];
        std::vector<double> m = j["m"][i];
        if ((int)m.size() != dim) {
          return false;
        }
        means.col(i) += w * Map<VectorXd>(m.data(), dim);
        std::vector<double> c = j["c"][i];
        if ((int)c.size() != dim * dim) {
          return false;
        }
        covs.middleCols(dim * i, dim) += w * Map<MatrixXd>(c.data(), dim, dim);
      }
      return true;
    } catch (...) {
      return false;
    }
  }

  // 更新模型（对角线加载为可选项）
  double Update(double noise_floor = 0.0) {
    if (m.Initialized()) {
      double s = weights.sum();
      entropy /= s;
      means.array().rowwise() /= weights.transpose().array();
      for (int i = 0; i < rank; i++) {
        Ref<MatrixXd> c = covs.middleCols(dim * i, dim);
        c /= weights(i);
        c.selfadjointView<Lower>().rankUpdate(means.col(i), -1.0);
        c.diagonal().array() += noise_floor * noise_floor;
      }
      weights /= s;
      m.Initialize(weights, means, covs);
      return entropy;
    } else {
      // 随机初始化
      const double s = weights(0);
      VectorXd u = means.col(0) / s;
      MatrixXd c = covs.leftCols(dim) / s;
      c.selfadjointView<Lower>().rankUpdate(u, -1.0);
      c.diagonal().array() += noise_floor * noise_floor;
      MVNGenerator gen(u, c);
      for (int i = 0; i < rank; i++) {
        weights(i) = 1.0 / rank;
        means.col(i) = gen.Gen();
        covs.middleCols(dim * i, dim) = c;
      }
      m.Initialize(weights, means, covs);
      return std::numeric_limits<double>::infinity();
    }
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
      std::cout << i << ": " << weights(i) << "\n";
      std::cout << "m:\n" << means.col(i) << "\n";
      std::cout << "s:\n" << covs.middleCols(dim * i, dim) << "\n";
    }
  }

  // 清空记忆
  void Reset() {
    entropy = 0;
    weights.setZero(rank);
    means.setZero(dim, rank);
    covs.setZero(dim, dim * rank);
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
      memcpy(p, means.col(i).data(), sizeof(double) * dim);
      p += sizeof(double) * dim;
      Map<MatrixXd>((double*)p, dim, dim) =
        covs.middleCols(dim * i, dim).selfadjointView<Lower>();
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
    entropy += w * (*(double*)p);
    p += sizeof(double);
    weights += w * Map<VectorXd>((double*)p, rank);
    p += sizeof(double) * rank;
    for (int i = 0; i < rank; i++) {
      means.col(i) += w * Map<VectorXd>((double*)p, dim);
      p += sizeof(double) * dim;
      covs.middleCols(dim * i, dim) += w * Map<MatrixXd>((double*)p, dim, dim);
      p += sizeof(double) * dim * dim;
    }
    return true;
  }

 protected:
  mix& m;
  int rank;
  int dim;
  double entropy;
  VectorXd weights;
  MatrixXd means;
  MatrixXd covs;
};

}  // namespace aha

#endif
