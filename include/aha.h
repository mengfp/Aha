#ifndef AHA_AHA_H
#define AHA_AHA_H

#include <vector>
#include <string>

namespace aha {

std::string Version();

class Model {
 public:
  Model();
  ~Model();
  bool Initialized() const;
  int Rank() const;
  int Dimension() const;
  std::vector<double> Predict(const std::vector<double>& x) const;
  std::vector<double> Export() const;
  bool Import(const std::vector<double>& model);

 private:
  void* p;
};

class Trainer {

private:
  void* p;
};

}  // namespace aha

#endif
