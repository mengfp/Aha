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
  std::vector<double> Predict(const std::vector<double>& x) const;
  std::vector<double> Export() const;
  bool Import(const std::vector<double>& model);

 private:
  void* p;
};

class Trainer {
 public:
  Trainer(Model& model);
  ~Trainer();
  int Rank() const;
  int Dim() const;
  double Score() const;
  void Initialize();
  void Merge(const Trainer& trainer);
  void Update();

 private:
  void* p;
};

}  // namespace aha

#endif
