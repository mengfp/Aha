// Test.cpp : This file contains the 'main' function. Program execution begins
// and ends there.
//
#include <iomanip>
#include <iostream>
#include <fstream>

#include "aha.h"
#include "generator.h"
#include "mvn.h"

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

  for (int k = 0; k < 20; k++) {
    gen.Initialize(rank, dim, seed);
    train.Initialize();
    for (int i = 0; i < N; i++) {
      gen.Gen(sample);
      train.Train(sample);
    }
    train.Update();
    std::cout << "Entropy = " << train.Entropy() << std::endl;
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
  const int N = 1000000;
  const int seed = 3;

  Gen2 gen;
  gen.Init(seed);

  std::cout << "Version: " << aha::Version() << std::endl;
  aha::Model model(5, 3);
  aha::Trainer trainer(model);

  for (int k = 0; k < 30; k++) {
    trainer.Initialize();
    std::vector<double> sample(3);
    for (int i = 0; i < N; i++) {
      gen.gen(sample);
      trainer.Train(sample);
    }
    trainer.Update();
    std::cout << "Entropy = " << trainer.Entropy() << std::endl;
  }

  // Test predict
  std::cout << "Test prediction ..." << std::endl;
  for (int k = 0; k < 10; k++) {
    std::vector<double> sample(3);
    gen.gen(sample);
    std::cout << "sample: " << sample[0] << " " << sample[1] << " " << sample[2]
              << std::endl;
    std::vector<double> y;
    sample.resize(2);
    model.Predict(sample, y);
    std::cout << "prediction: " << y[0] << std::endl;
  }

  (**(::trainer**)(&trainer)).Print();

  return true;
}

bool TestNonLinear() {
  GenNonLinear gen;
  gen.Init(1);
  std::vector<double> sample(3);
  int N = 1000000;

  aha::Model m1(1, 3);
  aha::Model m3(3, 3);
  aha::Model m5(5, 3);

  aha::Trainer t1(m1);
  aha::Trainer t3(m3);
  aha::Trainer t5(m5);

  for (int k = 0; k < 50; k++) {
    t1.Initialize();
    t3.Initialize();
    t5.Initialize();

    for (int i = 0; i < N; i++) {
      gen.gen(sample);
      t1.Train(sample);
      t3.Train(sample);
      t5.Train(sample);
    }

    t1.Update();
    t3.Update();
    t5.Update();

    std::cout << "Entropy = " << t1.Entropy() << " " << t3.Entropy() << " "
              << t5.Entropy() << std::endl;
  }

  // Predict
  std::ofstream ofs("nonlinear.csv");
  ofs << "X, Y, Z, Z1, Z3, Z5" << std::endl; 
  for (int i = 0; i < 100; i++) {
    std::vector<double> sample(3);
    gen.gen(sample);
    ofs << sample[0] << "," << sample[1] << "," << sample[2];
    std::vector<double> y;
    sample.resize(2);
    m1.Predict(sample, y);
    ofs << "," << y[0];
    m3.Predict(sample, y);
    ofs << "," << y[0];
    m5.Predict(sample, y);
    ofs << "," << y[0] << std::endl;
  }

  return true;
}

bool TestMVNGenerator() {
  int N = 1000000;
  Vector mean(4);
  mean << 1, 2, 3, 4;
  Matrix cov(4, 4);
  cov << 100, 32.4796258609215869, 31.6838227860951349, 141.409621752763684,
    32.4796258609215869, 110.549260960654465, -152.033658539600196,
    237.757814080695653, 31.6838227860951349, -152.033658539600196,
    373.530902783367878, -140.279703673223594, 141.409621752763684,
    237.757814080695653, -140.279703673223594, 827.467631118572399;
  MVNGenerator gen(mean, cov, 1);

  mix m(1, 4);
  trainer t(m);

  for (int k = 0; k < 10; k++) {
    t.Initialize();
    for (int i = 0; i < N; i++) {
      auto sample = gen.Gen();
      t.Train(sample);
      t.Train(sample);
    }
    t.Update();
    std::cout << "Entropy = " << t.Entropy() << std::endl;
  }
  t.Print();
  return true;
}

bool TestImportExport() {
  std::vector<double> weights(2);
  std::vector<Vector> means(2);
  std::vector<Matrix> covs(2);

  weights[0] = 0.1;
  weights[1] = 0.9;
   
  means[0] = Vector::Zero(4);
  means[0] << 1, 2, 3, 4;
  means[1] = Vector::Zero(4);
  means[1] << 4, 3, 2, 1;

  covs[0] = Matrix::Identity(4, 4);
  covs[1] = Matrix::Zero(4, 4);
  covs[1] << 100, 32.4796258609215869, 31.6838227860951349, 141.409621752763684,
            32.4796258609215869, 110.549260960654465, -152.033658539600196, 237.757814080695653,
            31.6838227860951349, -152.033658539600196, 373.530902783367878, -140.279703673223594,
            141.409621752763684, 237.757814080695653, -140.279703673223594, 827.467631118572399;
  mix m;
  m.Initialize(weights, means, covs);
  std::cout << "Original:" << std::endl;
  m.Print();
  std::string model = m.Export();
  std::cout << "Exported:" << std::endl;
  std::cout << model << std::endl;
  mix m2;
  m2.Import(model);
  std::cout << "Imported:" << std::endl;
  m2.Print();
  return true;
}

int main() {
  TestGaussian();
  TestRand();
  TestTrain();
  TestAha();
  TestNonLinear();
  TestMVNGenerator();
  TestImportExport();
  return 0;
}
