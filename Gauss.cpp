// Gauss.cpp : This file contains the 'main' function. Program execution begins
// and ends there.
//

#include <iostream>

#include "matrix.h"

using namespace gauss;

int main() {
  std::cout << "Hello World!\n";

  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
  _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);

  Matrix m(3, 3);
  m << 1, 2, 3, 4, 5, 6, 7, 8, 9;
  std::cout << m << std::endl;

  Vector v(3);
  v << 1, 2, 3;
  std::cout << v << std::endl;

  std::cout << m * v << std::endl;
}
