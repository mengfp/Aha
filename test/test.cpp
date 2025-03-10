/*
** Copyright 2025 Meng, Fanping. All rights reserved.
*/
#ifdef _MSC_VER
#pragma warning(disable : 4819)
#endif

#include <aha.h>
#include <generator.h>
#include <mvn.h>

#define USING_TIMER
#include <timer.h>

#include <fstream>
#include <iomanip>
#include <iostream>
#include <chrono>

using namespace std::chrono;
using namespace aha;
using namespace Eigen;

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

  mix m(rank, dim);
  trainer train(m);
  Generator gen;
  VectorXd sample = VectorXd::Zero(dim);

  for (int k = 0; k < 20; k++) {
    gen.Initialize(rank, dim, seed);
    train.Reset();
    for (int i = 0; i < N; i++) {
      gen.Gen(sample);
      train.Train(sample);
    }
    auto entropy = train.Update();
    std::cout << "Entropy = " << entropy << std::endl;
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

  std::cout << "Version: " << Version() << std::endl;
  Model model(5, 3);
  Trainer trainer(model);

  for (int k = 0; k < 30; k++) {
    std::vector<double> sample(3);
    trainer.Reset();
    for (int i = 0; i < N; i++) {
      gen.gen(sample);
      trainer.Train(sample);
    }
    auto e = trainer.Update();
    std::cout << "Entropy = " << e << std::endl;
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
  int N = 1000000;
  GenNonLinear gen;

  std::vector<std::vector<double>> samples;
  std::vector<double> sample(3);
  for (int i = 0; i < N; i++) {
    gen.gen(sample);
    samples.push_back(sample);
  }

  // Train
  Model model(5, 3);
  Trainer trainer(model);

  auto now = steady_clock::now();
  for (int k = 0; k < 30; k++) {
    trainer.Reset();
    for (auto& s : samples) {
      trainer.Train(s);
    }
    auto e = trainer.Update();
    std::cout << k << ": Entropy = " << e << std::endl;
  }
  std::cout << "Train time = "
            << duration<double>(steady_clock::now() - now).count() << std::endl;

  // Predict
  now = steady_clock::now();
  double d = 0.0;
  for (auto& s : samples) {
    std::vector<double> x(&s[0], &s[2]);
    std::vector<double> y;
    model.Predict(x, y);
    d += pow(y[0] - s[2], 2);
  }
  d /= N;
  std::cout << "MSE = " << d << std::endl;
  std::cout << "Predict time = "
            << duration<double>(steady_clock::now() - now).count() << std::endl;

  return true;
}

bool TestMVNGenerator() {
  int N = 1000000;
  VectorXd mean(4);
  mean << 1, 2, 3, 4;
  MatrixXd cov(4, 4);
  cov << 100, 32.4796258609215869, 31.6838227860951349, 141.409621752763684,
    32.4796258609215869, 110.549260960654465, -152.033658539600196,
    237.757814080695653, 31.6838227860951349, -152.033658539600196,
    373.530902783367878, -140.279703673223594, 141.409621752763684,
    237.757814080695653, -140.279703673223594, 827.467631118572399;
  MVNGenerator gen(mean, cov, 1);

  mix m(1, 4);
  trainer t(m);

  for (int k = 0; k < 10; k++) {
    t.Reset();
    for (int i = 0; i < N; i++) {
      auto sample = gen.Gen();
      t.Train(sample);
      t.Train(sample);
    }
    auto e = t.Update();
    std::cout << "Entropy = " << e << std::endl;
  }
  t.Print();
  return true;
}

bool TestImportExport() {
  std::vector<double> weights(2);
  std::vector<VectorXd> means(2);
  std::vector<MatrixXd> covs(2);

  weights[0] = 0.1;
  weights[1] = 0.9;

  means[0] = VectorXd::Zero(3);
  means[0] << 1, 2, 3;
  means[1] = VectorXd::Zero(3);
  means[1] << 3, 2, 1;

  covs[0] = MatrixXd::Identity(3, 3);
  covs[1] = MatrixXd::Zero(3, 3);
  covs[1] << 100, 32.4796258609215869, 31.6838227860951349, 32.4796258609215869,
    110.549260960654465, -152.033658539600196, 31.6838227860951349,
    -152.033658539600196, 373.530902783367878;

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

bool TestSpitSwallow() {
  mix m(2, 3);
  trainer t(m);

  std::string j = R"({
        "r": 2,
        "d": 3,
        "e": 0.1,
        "w": [1, 2],
        "m": [[100, 200, 300], [400, 500, 600]],
        "c": [[1,2,3,4,5,6,7,8,9], [9,8,7,6,5,4,3,2,1]]
          })";
  std::cout << t.Swallow(j) << std::endl;
  std::cout << t.Spit() << std::endl;

  return true;
}

#define EPS 1.0e-2

inline bool eq(const std::vector<double>& a, const std::vector<double>& b) {
  if (a.size() != b.size()) {
    return false;
  }
  for (int i = 0; i < (int)a.size(); i++) {
    if (fabs(a[i] - b[i]) > EPS) {
      return false;
    }
  }
  return true;
}

inline bool eq(const VectorXd& a, const VectorXd& b) {
  if (a.size() != b.size()) {
    return false;
  }
  for (int i = 0; i < (int)a.size(); i++) {
    if (fabs(a[i] - b[i]) > EPS) {
      return false;
    }
  }
  return true;
}

inline bool eq(const MatrixXd& a, const MatrixXd& b) {
  if (a.size() != b.size()) {
    return false;
  }
  for (int i = 0; i < (int)a.size(); i++) {
    if (fabs(a.data()[i] - b.data()[i]) > EPS) {
      return false;
    }
  }
  return true;
}

// Functional Verification Test
bool FVTest() {
  Timer timer("FVTTest");
  TimerGuard guard(timer);

  const int RANK = 3;
  const int DIM = 3;
  const int N = 100000;
  const int LOOP = 30;

  std::vector<MVNGenerator> generators(RANK);
  for (int i = 0; i < RANK; i++) {
    VectorXd mean = VectorXd::Ones(DIM) * i * 10;
    MatrixXd cov = MatrixXd::Identity(DIM, DIM);
    generators[i].Init(mean, cov);
  }

  Model m(RANK, DIM);
  mix *p = *(mix**)&m;
  Trainer t(m);
  // Single trainer
  for (int loop = 0; loop < LOOP; loop++) {
    t.Reset();
    for (int i = 0; i < N; i++) {
      t.Train(generators[0].Gen());
      t.Train(generators[0].Gen());
      t.Train(generators[0].Gen());
      t.Train(generators[1].Gen());
      t.Train(generators[1].Gen());
      t.Train(generators[2].Gen());
    }
    auto e = t.Update(1.0e-4);
    std::cout << loop << ": " << e << std::endl;
  }

  // To and from json
  auto json = m.Export();
  std::cout << "model: " << json << std::endl;
  if (json.empty()) {
    return false;
  }
  if (!m.Import(json)) {
    return false;
  }

  p->Sort();
  p->Print();

  // Verify
  if (m.Rank() != RANK) {
    return false;
  }
  if (m.Dim() != DIM) {
    return false;
  }
  if (!eq(p->GetWeights(), {0.5, 0.3333, 0.1667})) {
    return false;
  }
  for (int i = 0; i < RANK; i++) {
    if (!eq(p->GetCores()[i].getu(), generators[i].mean)) {
      return false;
    }
    if (!eq(p->GetCores()[i].getl(), generators[i].L)) {
      return false;
    }
  }

  // Multiple trainer
  Trainer t0(m);
  Trainer t1(m);
  Trainer t2(m);
  for (int loop = 0; loop < LOOP; loop++) {
    t.Reset();
    t0.Reset();
    t1.Reset();
    t2.Reset();
    for (int i = 0; i < N; i++) {
      t0.Train(generators[0].Gen());
      t0.Train(generators[0].Gen());
      t1.Train(generators[0].Gen());
      t1.Train(generators[1].Gen());
      t2.Train(generators[1].Gen());
      t2.Train(generators[2].Gen());
    }
#if 1
    t.Swallow(t0.Spit());
    t.Swallow(t1.Spit());
    t.Swallow(t2.Spit());
#else
    t.Merge(t0);
    t.Merge(t1);
    t.Merge(t2);
#endif
    auto e = t.Update(1.0e-4);
    std::cout << loop << ": " << e << std::endl;
  }
  p->Print();

  // Verify
  if (m.Rank() != RANK) {
    return false;
  }
  if (m.Dim() != DIM) {
    return false;
  }
  if (!eq(p->GetWeights(), {0.5, 0.3333, 0.1667})) {
    return false;
  }
  for (int i = 0; i < RANK; i++) {
    if (!eq(p->GetCores()[i].getu(), generators[i].mean)) {
      return false;
    }
    if (!eq(p->GetCores()[i].getl(), generators[i].L)) {
      return false;
    }
  }

  return true;
}

int main() {
  // TestGaussian();
  // TestRand();
  // TestTrain();
  // TestAha();
  // TestNonLinear();
  // TestMVNGenerator();
  // TestImportExport();
  // TestSpitSwallow();

  if (FVTest()) {
    std::cout << "### OK ###" << std::endl;
  } else {
    std::cout << "*** Failed ***" << std::endl;
  }

  return 0;
}
