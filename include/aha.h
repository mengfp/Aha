#ifndef AHA_AHA_H
#define AHA_AHA_H

#include <string>
#include <vector>

namespace aha {

std::string Version();

class Model {
 public:
  Model(int rank, int dim);
  ~Model();
  bool Initialized() const;
  int Rank() const;
  int Dim() const;
  double Predict(const std::vector<double>& x, std::vector<double>& y) const;
  std::string Export() const;
  bool Import(const std::string& model);

 private:
  void* p;
};

class Trainer {
 public:
  Trainer(Model& m, uint64_t seed = 0);
  ~Trainer();
  int Rank() const;
  int Dim() const;
  double Entropy() const;
  void Reset();
  void Train(const std::vector<double>& sample);
  void Merge(const Trainer& t);
  void Update();

 private:
  void* p;
};

}  // namespace aha

#endif
