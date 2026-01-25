/*
** Copyright 2025 Meng, Fanping. All rights reserved.
*/
#ifndef AHA_VERSION_H
#define AHA_VERSION_H

#define VERSION "1.3.1"

#define MAGIC 0x0ebf76da

#include <string>

inline int parse_version(const char* ver) {
  int a = 0, b = 0, c = 0;
  try {
    std::size_t pos;
    a = std::stoi(ver, &pos);
    ver += pos;
    if (*ver == '.') {
      b = std::stoi(++ver, &pos);
      ver += pos;
      if (*ver == '.') {
        c = std::stoi(++ver, &pos);
      }
    }
  } catch (...) {
  }
  return (a << 16) + (b << 8) + c;
}

#endif
