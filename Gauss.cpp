// Gauss.cpp : This file contains the 'main' function. Program execution begins
// and ends there.
//
#include <iomanip>
#include <iostream>

#include "gaussian.h"

using namespace gauss;

bool TestGaussian() {
  VectorXd mu(4);
  mu << 1, 2, 3, 4;
  MatrixXd sigma(4, 4);
  sigma << 100, 32.4796258609215869, 31.6838227860951349, 141.409621752763684,
    32.4796258609215869, 110.549260960654465, -152.033658539600196,
    237.757814080695653, 31.6838227860951349, -152.033658539600196,
    373.530902783367878, -140.279703673223594, 141.409621752763684,
    237.757814080695653, -140.279703673223594, 827.467631118572399;
  VectorXd x(4);
  x << -52.8138247836419055, 167.036008837659296, -254.908653564947315,
    437.285521520668226;
  x += mu;
  Gaussian g(mu, sigma);
  auto e = 1.0e-12;

  auto error = g.Evaluate(x) - (-248.438063922770);
  if (error < -e || error > e) {
    std::cout << "*** TestGaussian failed" << std::endl;
    return false;
  }

  error = g.PartialEvaluate(x.head(3)) - (-211.729511486218);
  if (error < -e || error > e) {
    std::cout << "*** TestGaussian failed" << std::endl;
    return false;
  }

  VectorXd y;
  error = g.Predict(x.head(2), y) - (-190.01885211864618);
  if (error < -e || error > e) {
    std::cout << "*** TestGaussian failed" << std::endl;
    return false;
  }

  y -= mu.tail(2);

  error = y(0) - (-315.71841551378759);
  if (error < -e || error > e) {
    std::cout << "*** TestGaussian failed" << std::endl;
    return false;
  }

  error = y(1) - (278.644584795271271);
  if (error < -e || error > e) {
    std::cout << "*** TestGaussian failed" << std::endl;
    return false;
  }
  
  std::cout << "TestGaussian OK" << std::endl;
  return true;
}

int main() {
  TestGaussian();
}
