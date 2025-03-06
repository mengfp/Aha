/*
** Copyright 2025 Meng, Fanping. All rights reserved.
*/
#ifdef _MSC_VER
#pragma warning(disable : 4819)
#endif

#include <aha.h>
#include <mvn.h>
#include <version.h>

namespace aha {

std::string Version() {
  return VERSION;
}

#define T double
#include "aha.template"
#undef T
#define T float
#include "aha.template"

}  // namespace aha
