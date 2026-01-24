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
using Eigen::MatrixXf;
using Eigen::VectorXf;
using Eigen::Map;
using Eigen::Ref;

std::string Version();

class Model {
 public:
  Model(int rank = 0, int dim = 0);
  ~Model();
  bool Initialized() const;
  int Rank() const;
  int Dim() const;
  double Predict(Ref<const VectorXd> x, VectorXd& y) const;
  void Sort();
  std::string Export() const;
  bool Import(const std::string& model);
  VectorXd BatchPredict(Ref<const MatrixXd> X, MatrixXd& Y) const;
  VectorXd FastPredict(Ref<const MatrixXf> X, MatrixXf& Y) const;
  std::vector<char> Dump() const;
  bool Load(const std::vector<char>& input);
  double PredictEx(Ref<const VectorXd> x, VectorXd& y, MatrixXd& cov) const;
  VectorXd BatchPredictEx(Ref<const MatrixXd> X, MatrixXd& Y, MatrixXd& COV) const;
  VectorXd FastPredictEx(Ref<const MatrixXd> X, MatrixXd& Y, MatrixXd& COV) const;

 private:
  void* p;
};

class Trainer {
 public:
  Trainer(Model& m);
  ~Trainer();
  int Rank() const;
  int Dim() const;
  void Train(Ref<const VectorXd> sample);
  bool Merge(const Trainer& t, double w = 1.0);
  std::string Spit() const;
  bool Swallow(const std::string& t, double w = 1.0);
  double Update(double noise_floor = 0.0);
  void Reset();
  void BatchTrain(Ref<const MatrixXd> samples);
  void FastTrain(Ref<const MatrixXf> samples);
  std::vector<char> Dump() const;
  bool Load(const std::vector<char>& input, double w = 1.0);

 private:
  void* p;
};

}  // namespace aha

#endif
