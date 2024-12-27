#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include "aha.h"

namespace pb = pybind11;
using namespace aha;

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

PYBIND11_MODULE(aha, m) {
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
    .def("Initialize", &Trainer::Initialize)
    .def("Train", &Trainer::Train)
    .def("Merge", &Trainer::Merge)
    .def("Update", &Trainer::Update);

#ifdef VERSION
  m.attr("__version__") = MACRO_STRINGIFY(VERSION);
#else
  m.attr("__version__") = "dev";
#endif
}
