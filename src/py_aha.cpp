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

  // class Model
  py::class_<Model>(m, "Model")
    .def(py::init<int, int>(), py::arg("rank") = 0, py::arg("dim") = 0)
    .def("Initialized", &Model::Initialized)
    .def("Rank", &Model::Rank)
    .def("Dim", &Model::Dim)
    .def(
      "Predict",
      [](const Model& self, const std::vector<double>& x) {
        std::vector<double> y;
        auto r = self.Predict(x, y);
        return py::make_tuple(r, y);
      },
      py::arg("x"))
    .def(
      "BatchPredict",
      [](const Model& self, const MatrixXd& x) {
        MatrixXd y;
        auto r = self.BatchPredict(x.transpose(), y);
        return py::make_tuple(r, y.transpose());
      },
      py::arg("x"))
    .def(
      "FastPredict",
      [](const Model& self, const MatrixXd& x) {
        MatrixXd y;
        auto r = self.FastPredict(x.transpose(), y);
        return py::make_tuple(r, y.transpose());
      },
      py::arg("x"))
    .def("Sort", &Model::Sort)
    .def("Export", &Model::Export)
    .def("Import", &Model::Import, py::arg("model"));

  // class Trainer
  py::class_<Trainer>(m, "Trainer")
    .def(py::init<Model&>(), py::arg("model"))
    .def("Rank", &Trainer::Rank)
    .def("Dim", &Trainer::Dim)
    .def(
      "Train",
      [](Trainer& self, const std::vector<double>& sample) {
        return self.Train(sample);
      },
      py::arg("sample"))
    .def(
      "BatchTrain",
      [](Trainer& self, const MatrixXd& samples) {
        self.BatchTrain(samples.transpose());
      },
      py::arg("samples"))
    .def(
      "FastTrain",
      [](Trainer& self, const MatrixXd& samples) {
        self.FastTrain(samples.transpose());
      },
      py::arg("samples"))
    .def("Merge", &Trainer::Merge, py::arg("trainer"), py::arg("w") = 1.0)
    .def("Spit", &Trainer::Spit)
    .def("Swallow", &Trainer::Swallow, py::arg("trainer"), py::arg("w") = 1.0)
    .def("Update", &Trainer::Update, py::arg("noise_floor") = 0.0)
    .def("Reset", &Trainer::Reset);
}
