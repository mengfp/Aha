/*
** Copyright 2025 Meng, Fanping. All rights reserved.
*/
#ifndef AHA_AHA_H
#define AHA_AHA_H

#include <string>
#include <vector>
#include <Eigen/Dense>

using namespace Eigen;
#ifndef Vector
#define Vector VectorXd
#endif
#ifndef Matrix
#define Matrix MatrixXd
#endif

namespace aha {

std::string Version();

class Model {
 public:
  Model(int rank, int dim);
  ~Model();
  bool Initialized() const;
  int Rank() const;
  int Dim() const;
  double Predict(const Vector& x, Vector& y) const;
  double Predict(const std::vector<double>& x, std::vector<double>& y) const;
  std::string Export() const;
  bool Import(const std::string& model);

 private:
  void* p;
};

class Trainer {
 public:
  Trainer(Model& m);
  ~Trainer();
  int Rank() const;
  int Dim() const;
  void Train(const Vector& sample);
  void Train(const std::vector<double>& sample);
  void Merge(const Trainer& t, double w = 1.0);
  std::string Spit();
  bool Swallow(const std::string& t, double w = 1.0);
  double Update();

 private:
  void* p;
};

}  // namespace aha

#endif
