// Test.cpp : This file contains the 'main' function. Program execution begins
// and ends there.
//
#include <iomanip>
#include <iostream>
#include "mvn.h"
#include "generator.h"
#include "aha.h"

bool TestGaussian() {
  Vector mu(4);
  mu << 1, 2, 3, 4;
  Matrix sigma(4, 4);
  sigma << 100, 32.4796258609215869, 31.6838227860951349, 141.409621752763684,
    32.4796258609215869, 110.549260960654465, -152.033658539600196,
    237.757814080695653, 31.6838227860951349, -152.033658539600196,
    373.530902783367878, -140.279703673223594, 141.409621752763684,
    237.757814080695653, -140.279703673223594, 827.467631118572399;
  Vector x(4);
  x << -52.8138247836419055, 167.036008837659296, -254.908653564947315,
    437.285521520668226;
  x += mu;
  mvn g(mu, sigma);
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

  Vector y;
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

bool TestRand() {
  const int N = 1000000;
  double mean = 0;
  double var = 0;
  MTRand rand(uint32_t(1));
  for (int i = 0; i < N; i++) {
    auto x = rand.randNorm(0.0, 1.0);
    mean += x;
    var += x * x;
  }
  mean /= N;
  var = var / N - mean * mean;
  std::cout << "TestRand: "
            << "mean = " << mean << ", var = " << var << std::endl;
  return true;
}

bool TestTrain() {
  const int N = 1000000;
  const int seed = 1;
  const int rank = 3;
  const int dim = 3;

  mix mix(rank, dim);
  trainer train(mix);
  Generator gen;
  Vector sample = Vector::Zero(dim);

  for (int k = 0; k < 10; k++) {
    gen.Initialize(rank, dim, seed);
    train.Initialize();
    for (int i = 0; i < N; i++) {
      gen.Gen(sample);
      train.Train(sample);    
    }
    train.Update();
    std::cout << "Score = " << train.Score() << std::endl;
  }

  std::cout << "Generator:" << std::endl;
  gen.Print();
  std::cout << std::endl;
  std::cout << "Trainer:" << std::endl;
  train.Print();
  std::cout << std::endl;

  return true;
}

bool TestAha() {
  std::cout << "Version: " << aha::Version() << std::endl;
  aha::Model model(3, 3);
  aha::Trainer trainer(model);
  return true;
}

int main() {
  TestGaussian();
  TestRand();
  TestTrain();
  TestAha();
  return 0;
}
