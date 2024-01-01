// Gauss.cpp : This file contains the 'main' function. Program execution begins
// and ends there.
//
#include <iostream>
#include <iomanip>

#include "gaussian.h"

using namespace gauss;

bool TestGaussian() {
  VectorXd mu(4);
  mu << 1, 2, 3, 4;
  MatrixXd sigma(4, 4);
  sigma << 1, -1.63146317635113, 0.48415374753413, 0.612554313174915,
    -1.63146317635113, 3.66167209578971, -1.99150882300832, -2.98246059182851,
    0.48415374753413, -1.99150882300832, 2.67831905685275, 2.62385056110414,
    0.612554313174915, -2.98246059182851, 2.62385056110414, 5.31101099069907;
  VectorXd x(4);
  x << -1.30535215719319, 1.74210276800367, -1.27370521036104,
    0.985230072213369;
  x += mu;
  Gaussian g(mu, sigma);
  auto error = g.Evaluate(x) - (-5.671657285065);
  if (error < 1.0e-12 && error > -1.0e-12) {
    std::cout << "TestGaussian OK" << std::endl;
    return true;
  } else {
    std::cout << "*** TestGaussian failed" << std::endl;
    return false;
  }
}

int main() {
  TestGaussian();
}
