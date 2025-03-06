/*
** Copyright 2025 Meng, Fanping. All rights reserved.
*/
#ifndef AHA_MVN_H
#define AHA_MVN_H

#include <cassert>
#include <cfloat>
#include <iostream>
#include <nlohmann/json.hpp>
#include <vector>
#include <Eigen/Dense>

#include "generator.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846264338327950288
#endif

namespace aha {

using namespace Eigen;
using json = nlohmann::ordered_json;

#undef T
#undef mvn
#undef mix
#undef trainer
#define T double
#define mvn mvn64
#define mix mix64
#define trainer trainer64
#include "mvn_impl.h"

#undef T
#undef mvn
#undef mix
#undef trainer
#define T float
#define mvn mvn32
#define mix mix32
#define trainer trainer32
#include "mvn_impl.h"

#undef T
#undef mvn
#undef mix
#undef trainer

}  // namespace aha

#endif
