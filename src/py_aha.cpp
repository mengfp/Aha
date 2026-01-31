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
using RowMatrixXdRef = Ref<const Matrix<double, -1, -1, RowMajor>>;
using RowMatrixXfRef = Ref<const Matrix<float, -1, -1, RowMajor>>;

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
      [](const Model& self, Ref<const VectorXd> x) {
        VectorXd y;
        auto r = self.Predict(x, y);
        return py::make_tuple(r, y);
      },
      py::arg("x"))
    .def(
      "PredictEx",
      [](const Model& self, Ref<const VectorXd> x) {
        VectorXd y;
        VectorXd vars;
        auto r = self.PredictEx(x, y, vars);
        return py::make_tuple(r, y, vars);
      },
      py::arg("x"))
    .def(
      "BatchPredict",
      [](const Model& self, RowMatrixXdRef x) {
        Map<const MatrixXd> x_view(x.data(), x.cols(), x.rows());
        MatrixXd y;
        auto r = self.BatchPredict(x_view, y);
        return py::make_tuple(r, y.transpose());
      },
      py::arg("x").noconvert())
    .def(
      "BatchPredictEx",
      [](const Model& self, RowMatrixXdRef x) {
        Map<const MatrixXd> x_view(x.data(), x.cols(), x.rows());
        MatrixXd y;
        MatrixXd vars;
        auto r = self.BatchPredictEx(x_view, y, vars);
        return py::make_tuple(r, y.transpose(), vars.transpose());
      },
      py::arg("x").noconvert())
    .def(
      "FastPredict",
      [](const Model& self, RowMatrixXfRef x) {
        Map<const MatrixXf> x_view(x.data(), x.cols(), x.rows());
        MatrixXf y;
        auto r = self.FastPredict(x_view, y);
        return py::make_tuple(r, y.transpose());
      },
      py::arg("x").noconvert())
    .def(
      "FastPredictEx",
      [](const Model& self, RowMatrixXfRef x) {
        Map<const MatrixXf> x_view(x.data(), x.cols(), x.rows());
        MatrixXf y;
        MatrixXf vars;
        auto r = self.FastPredictEx(x_view, y, vars);
        return py::make_tuple(r, y.transpose(), vars.transpose());
      },
      py::arg("x").noconvert())
    .def("Sort", &Model::Sort)
    .def("Export", &Model::Export)
    .def("Import", &Model::Import, py::arg("model"))
    .def("Dump",
         [](const Model& self) {
           auto data = self.Dump();
           return py::bytes(data.data(), data.size());
         })
    .def(
      "Load",
      [](Model& self, py::bytes model) {
        py::buffer_info info(py::buffer(model).request());
        std::vector<char> data((char*)info.ptr, (char*)info.ptr + info.size);
        return self.Load(data);
      },
      py::arg("model"));

  // class Trainer
  py::class_<Trainer>(m, "Trainer")
    .def(py::init<Model&>(), py::arg("model"))
    .def("Rank", &Trainer::Rank)
    .def("Dim", &Trainer::Dim)
    .def(
      "Train",
      [](Trainer& self, Ref<const VectorXd> sample) {
        return self.Train(sample);
      },
      py::arg("sample"))
    .def(
      "BatchTrain",
      [](Trainer& self, RowMatrixXdRef samples) {
        Map<const MatrixXd> samples_view(
          samples.data(), samples.cols(), samples.rows());
        self.BatchTrain(samples_view);
      },
      py::arg("samples").noconvert())
    .def(
      "FastTrain",
      [](Trainer& self, RowMatrixXfRef samples) {
        Map<const MatrixXf> samples_view(
          samples.data(), samples.cols(), samples.rows());
        self.FastTrain(samples_view);
      },
      py::arg("samples").noconvert())
    .def("Merge", &Trainer::Merge, py::arg("trainer"), py::arg("w") = 1.0)
    .def("Spit", &Trainer::Spit)
    .def("Swallow", &Trainer::Swallow, py::arg("trainer"), py::arg("w") = 1.0)
    .def("Update", &Trainer::Update, py::arg("noise_floor") = 0.0)
    .def("Reset", &Trainer::Reset)
    .def("Dump",
         [](const Trainer& self) {
           auto data = self.Dump();
           return py::bytes(data.data(), data.size());
         })
    .def(
      "Load",
      [](Trainer& self, py::bytes trainer, double w) {
        py::buffer_info info(py::buffer(trainer).request());
        std::vector<char> data((char*)info.ptr, (char*)info.ptr + info.size);
        return self.Load(data, w);
      },
      py::arg("trainer"),
      py::arg("w") = 1.0);
}
