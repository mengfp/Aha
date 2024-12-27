#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <aha.h>
#include <version.h>

namespace pb = pybind11;
using namespace aha;

PYBIND11_MODULE(aha, m) {
  m.attr("__version__") = VERSION;
    
  // Version
  m.def("Version", Version);

  // class Model
  pb::class_<Model>(m, "Model")
    .def(pb::init<int, int>(), pb::arg("rank"), pb::arg("dim"))
    .def("Initialized", &Model::Initialized)
    .def("Rank", &Model::Rank)
    .def("Dim", &Model::Dim)
    .def("Predict", &Model::Predict, pb::arg("x"), pb::arg("y"))
    .def("Export", &Model::Export)
    .def("Import", &Model::Import);

  // class Trainer
  pb::class_<Trainer>(m, "Trainer")
    .def(pb::init<Model&, uint64_t>(), pb::arg("model"), pb::arg("seed") = 0)
    .def("Rank", &Trainer::Rank)
    .def("Dim", &Trainer::Dim)
    .def("Entropy", &Trainer::Entropy)
    .def("Reset", &Trainer::Reset)
    .def("Train", &Trainer::Train)
    .def("Merge", &Trainer::Merge)
    .def("Update", &Trainer::Update);
}
