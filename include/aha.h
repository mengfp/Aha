/*
** Copyright 2025 Meng, Fanping. All rights reserved.
*/
#ifndef AHA_AHA_H
#define AHA_AHA_H

#include <string>
#include <vector>

#include "eigen.h"

namespace aha {

std::string Version();

template <typename T>
class Model {
 public:
  Model(int rank = 0, int dim = 0);
  ~Model();
  bool Initialized() const;
  int Rank() const;
  int Dim() const;
  double Predict(const Vector<T>& x, Vector<T>& y) const;
  double Predict(const std::vector<T>& x, std::vector<T>& y) const;
  void Sort();
  std::string Export() const;
  bool Import(const std::string& model);

 private:
  void* p;
};

template <typename T>
class Trainer {
 public:
  Trainer(Model<T>& m);
  ~Trainer();
  int Rank() const;
  int Dim() const;
  void Train(const Vector<T>& sample);
  void Train(const std::vector<T>& sample);
  bool Merge(const Trainer& t, double w = 1.0);
  std::string Spit();
  bool Swallow(const std::string& t, double w = 1.0);
  double Update(double noise_floor = 0.0);
  void Reset();

 private:
  void* p;
};

}  // namespace aha

#endif
