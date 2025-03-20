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

std::string Version();

class Model {
 public:
  Model(int rank = 0, int dim = 0);
  ~Model();
  bool Initialized() const;
  int Rank() const;
  int Dim() const;
  double Predict(const VectorXd& x, VectorXd& y) const;
  double Predict(const std::vector<double>& x, std::vector<double>& y) const;
  void Sort();
  std::string Export() const;
  bool Import(const std::string& model);
  VectorXd BatchPredict(const MatrixXd& X, MatrixXd& Y) const;
  VectorXd FastPredict(const MatrixXd& X, MatrixXd& Y) const;
  std::vector<char> Dump() const;
  bool Load(const std::vector<char>& input);

 private:
  void* p;
};

class Trainer {
 public:
  Trainer(Model& m);
  ~Trainer();
  int Rank() const;
  int Dim() const;
  void Train(const VectorXd& sample);
  void Train(const std::vector<double>& sample);
  bool Merge(const Trainer& t, double w = 1.0);
  std::string Spit() const;
  bool Swallow(const std::string& t, double w = 1.0);
  double Update(double noise_floor = 0.0);
  void Reset();
  void BatchTrain(const MatrixXd& samples);
  void FastTrain(const MatrixXd& samples);
  std::vector<char> Dump() const;
  bool Load(const std::vector<char>& input, double w = 1.0);

 private:
  void* p;
};

}  // namespace aha

#endif
