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

PYBIND11_MODULE(aha, m) {
  m.attr("__version__") = VERSION;

  // Version
  m.def("Version", Version);

  // class Model
  py::class_<Model>(m, "Model")
    .def(py::init<int, int>(), py::arg("rank"), py::arg("dim"))
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
      [](const Model& self, const Matrix& x) {
        Vector r = Vector::Zero(x.rows());
        Matrix y = Matrix::Zero(x.rows(), self.Dim() - x.cols());
        Vector temp;
        for (int i = 0; i < (int)x.rows(); i++) {
          r[i] = self.Predict(x.row(i), temp);
          y.row(i) = temp;
        }
        return py::make_tuple(r, y);
      },
      py::arg("x"))
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
      [](Trainer& self, const Matrix& samples) {
        for (auto row : samples.rowwise()) {
          self.Train(row);
        }
      },
      py::arg("samples"))
    .def("Merge", &Trainer::Merge, py::arg("trainer"), py::arg("w") = 1.0)
    .def("Spit", &Trainer::Spit)
    .def("Swallow", &Trainer::Swallow, py::arg("trainer"), py::arg("w") = 1.0)
    .def("Update", &Trainer::Update);
}
