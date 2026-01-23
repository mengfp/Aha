/*
** Copyright 2025 Meng, Fanping. All rights reserved.
*/
#ifndef AHA_AHA_H
#define AHA_AHA_H

#include <string>
#include <vector>
#include <Eigen/Dense>

namespace aha {

using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::Map;
using Eigen::Ref;
using MatrixXdRef = Eigen::Ref<const Eigen::MatrixXd>;
using VectorXdRef = Eigen::Ref<const Eigen::VectorXd>;
using MatrixXfRef = Eigen::Ref<const Eigen::MatrixXf>;
using VectorXfRef = Eigen::Ref<const Eigen::VectorXf>;

std::string Version();

class Model {
 public:
  Model(int rank = 0, int dim = 0);
  ~Model();
  bool Initialized() const;
  int Rank() const;
  int Dim() const;
  double Predict(const VectorXdRef& x, VectorXd& y) const;
  double Predict(const std::vector<double>& x, std::vector<double>& y) const;
  void Sort();
  std::string Export() const;
  bool Import(const std::string& model);
  VectorXd BatchPredict(const MatrixXdRef& X, MatrixXd& Y) const;
  VectorXd FastPredict(const MatrixXdRef& X, MatrixXd& Y) const;
  std::vector<char> Dump() const;
  bool Load(const std::vector<char>& input);
  double PredictEx(const VectorXdRef& x, VectorXd& y, MatrixXd& cov) const;
  VectorXd BatchPredictEx(const MatrixXdRef& X, MatrixXd& Y, MatrixXd& COV) const;
  VectorXd FastPredictEx(const MatrixXdRef& X, MatrixXd& Y, MatrixXd& COV) const;

 private:
  void* p;
};

class Trainer {
 public:
  Trainer(Model& m);
  ~Trainer();
  int Rank() const;
  int Dim() const;
  void Train(const VectorXdRef& sample);
  void Train(const std::vector<double>& sample);
  bool Merge(const Trainer& t, double w = 1.0);
  std::string Spit() const;
  bool Swallow(const std::string& t, double w = 1.0);
  double Update(double noise_floor = 0.0);
  void Reset();
  void BatchTrain(const MatrixXdRef& samples);
  void FastTrain(const MatrixXdRef& samples);
  std::vector<char> Dump() const;
  bool Load(const std::vector<char>& input, double w = 1.0);

 private:
  void* p;
};

}  // namespace aha

#endif
