/*
** Copyright 2025 Meng, Fanping. All rights reserved.
*/
#ifdef _MSC_VER
#pragma warning(disable : 4819)
#endif

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <version.h>
#include <aha.h>

namespace py = pybind11;
using namespace aha;
using namespace Eigen;

PYBIND11_MODULE(aha, m) {
  m.attr("__version__") = VERSION;

  // Version
  m.def("Version", Version);

#define DECL(Model, ModelStr, Trainer, TrainerStr, M, V, T)                    \
  py::class_<Model>(m, ModelStr)                                               \
    .def(py::init<int, int>(), py::arg("rank") = 0, py::arg("dim") = 0)        \
    .def("Initialized", &Model::Initialized)                                   \
    .def("Rank", &Model::Rank)                                                 \
    .def("Dim", &Model::Dim)                                                   \
    .def(                                                                      \
      "Predict",                                                               \
      [](const Model& self, const std::vector<T>& x) {                         \
        std::vector<T> y;                                                      \
        auto r = self.Predict(x, y);                                           \
        return py::make_tuple(r, y);                                           \
      },                                                                       \
      py::arg("x"))                                                            \
    .def(                                                                      \
      "BatchPredict",                                                          \
      [](const Model& self, const M& x) {                                      \
        V r = V::Zero(x.rows());                                               \
        M y = M::Zero(x.rows(), self.Dim() - x.cols());                        \
        V temp;                                                                \
        for (int i = 0; i < (int)x.rows(); i++) {                              \
          r[i] = T(self.Predict(x.row(i), temp));                              \
          y.row(i) = temp;                                                     \
        }                                                                      \
        return py::make_tuple(r, y);                                           \
      },                                                                       \
      py::arg("x"))                                                            \
    .def("Sort", &Model::Sort)                                                 \
    .def("Export", &Model::Export)                                             \
    .def("Import", &Model::Import, py::arg("model"));                          \
                                                                               \
  py::class_<Trainer>(m, TrainerStr)                                           \
    .def(py::init<Model&>(), py::arg("model"))                                 \
    .def("Rank", &Trainer::Rank)                                               \
    .def("Dim", &Trainer::Dim)                                                 \
    .def(                                                                      \
      "Train",                                                                 \
      [](Trainer& self, const std::vector<T>& sample) {                        \
        return self.Train(sample);                                             \
      },                                                                       \
      py::arg("sample"))                                                       \
    .def(                                                                      \
      "BatchTrain",                                                            \
      [](Trainer& self, const M& samples) {                                    \
        for (auto row : samples.rowwise()) {                                   \
          self.Train(row);                                                     \
        }                                                                      \
      },                                                                       \
      py::arg("samples"))                                                      \
    .def("Merge", &Trainer::Merge, py::arg("trainer"), py::arg("w") = 1.0)     \
    .def("Spit", &Trainer::Spit)                                               \
    .def("Swallow", &Trainer::Swallow, py::arg("trainer"), py::arg("w") = 1.0) \
    .def("Update", &Trainer::Update, py::arg("noise_floor") = 0.0)             \
    .def("Reset", &Trainer::Reset);

  DECL(Model64, "Model64", Trainer64, "Trainer64", MatrixXd, VectorXd, double)
  DECL(Model32, "Model32", Trainer32, "Trainer32", MatrixXf, VectorXf, float)
}
