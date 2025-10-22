/*
** Copyright 2025 Meng, Fanping. All rights reserved.
*/
#ifdef _MSC_VER
#pragma warning(disable : 4819 4805)
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
#include <string>
#include <random>

using namespace std::chrono;
using namespace aha;

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
  Random rand(uint32_t(1));
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
  std::cout << "model size = " << model.size() << std::endl;
  std::cout << "Exported:" << std::endl;
  std::cout << model << std::endl;
  mix m2;
  m2.Import(model);
  std::cout << "Imported:" << std::endl;
  m2.Print();
  return true;
}

bool TestDumpLoad() {
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
  auto model = m.Dump();
  std::cout << "model size = " << model.size() << std::endl;
  mix m2;
  if (m2.Load(model)) {
    std::cout << "Reloaded:" << std::endl;
    m2.Print();
  } else {
    std::cout << "Error" << std::endl;
  }
  return true;
}

bool TestSpitSwallow() {
  mix m(2, 3);
  trainer t(m);
  t.Reset();

  std::string j = R"({
        "r": 2,
        "d": 3,
        "e": 0.1,
        "w": [1, 2],
        "m": [[100, 200, 300], [400, 500, 600]],
        "c": [[1,2,3,4,5,6,7,8,9], [9,8,7,6,5,4,3,2,1]]
          })";
  std::cout << "Swallow: " << t.Swallow(j) << std::endl;
  std::cout << "Spit: " << t.Spit() << std::endl;

  // test Dump/Load
  std::cout << "Dump/Load: " << t.Load(t.Dump()) << std::endl;
  std::cout << "Spit again: " << t.Spit() << std::endl;

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
  mix* p = *(mix**)&m;
  Trainer t(m);
  // Single trainer
  for (int loop = 0; loop < LOOP; loop++) {
    t.Reset();
    auto samples = MatrixXd(DIM, 6);
    for (int i = 0; i < N; i++) {
      samples.col(0) = generators[0].Gen();
      samples.col(1) = generators[0].Gen();
      samples.col(2) = generators[0].Gen();
      samples.col(3) = generators[1].Gen();
      samples.col(4) = generators[1].Gen();
      samples.col(5) = generators[2].Gen();
      // for (int j = 0; j < 6; j++) {
      //   t.Train(samples.col(j));
      // }
      t.BatchTrain(samples);
    }
    auto e = t.Update(1.0e-4);
    std::cout << loop << ": " << e << std::endl;
  }

  // To and from json
  auto json = m.Export();
  std::cout << "json size: " << json.size() << std::endl;
  std::cout << "json: " << json << std::endl;
  if (json.empty()) {
    return false;
  }
  if (!m.Import(json)) {
    return false;
  }

  // Dump and load binary
  auto model = m.Dump();
  std::cout << "model size: " << model.size() << std::endl;
  if (model.empty()) {
    return false;
  }
  if (!m.Load(model)) {
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

template <typename T>
double distance(const T& a, const T& b) {
  return double((a - b).template lpNorm<Infinity>());
}

bool TestBatchPredict() {
  const int RANK = 8;
  const int DIM = 256;
  const int N = 10000;
  const int K = 16;

  std::vector<double> weights(RANK);
  std::vector<VectorXd> means(RANK);
  std::vector<MatrixXd> covs(RANK);
  std::vector<MVNGenerator> generators(RANK);

  for (int i = 0; i < RANK; i++) {
    weights[i] = 1.0 / RANK;
    means[i] = VectorXd::Ones(DIM) * i * 10;
    covs[i] = MatrixXd::Identity(DIM, DIM);
    generators[i].Init(means[i], covs[i]);
  }

  Model m(RANK, DIM);
  mix* p = *(mix**)&m;
  p->Initialize(weights, means, covs);

  MatrixXd data(DIM, N);
  for (int i = 0; i < N; i++) {
    data.col(i) = generators[i % RANK].Gen();
  }

  // Single predict
  VectorXd r(N);
  MatrixXd Y(K, N);
  Timer t_single("SinglePredict");
  t_single.start();
  for (int i = 0; i < N; i++) {
    VectorXd y;
    r[i] = m.Predict(data.col(i).head(DIM - K), y);
    Y.col(i) = y;
  }
  t_single.stop();

  // Batch predict
  VectorXd _r;
  MatrixXd _Y;
  Timer t_batch("BatchPredict");
  t_batch.start();
  _r = m.BatchPredict(data.topRows(DIM - K), _Y);
  t_batch.stop();

  // Fast predict
  VectorXd __r;
  MatrixXd __Y;
  Timer t_fast("FastPredict");
  t_fast.start();
  __r = m.FastPredict(data.topRows(DIM - K), __Y);
  t_fast.stop();

  std::cout << distance(r, _r) << ", " << distance(Y, _Y) << std::endl;
  std::cout << distance(r, __r) << ", " << distance(Y, __Y) << std::endl;

  if (distance(r, _r) > 1.0e-6 || distance(Y, _Y) > 1.0e-6 ||
      distance(r, __r) > 1.0e-3 || distance(Y, __Y) > 1.0e-6)
    return false;
  else
    return true;
}

double distance(const Model& m1, const Model& m2) {
  double e = 0.0;
  mix* p1 = *(mix**)&m1;
  mix* p2 = *(mix**)&m2;
  for (int i = 0; i < m1.Rank(); i++) {
    e = std::max(e, fabs(p1->GetWeights()[i] - p2->GetWeights()[i]));
    e =
      std::max(e, distance(p1->GetCores()[i].getu(), p2->GetCores()[i].getu()));
    e =
      std::max(e, distance(p1->GetCores()[i].getl(), p2->GetCores()[i].getl()));
  }
  return e;
}

bool TestBatchTrain() {
  const int RANK = 8;
  const int DIM = 256;
  const int N = 10000;

  std::vector<double> weights(RANK);
  std::vector<VectorXd> means(RANK);
  std::vector<MatrixXd> covs(RANK);
  std::vector<MVNGenerator> generators(RANK);

  for (int i = 0; i < RANK; i++) {
    weights[i] = 1.0 / RANK;
    means[i] = VectorXd::Ones(DIM) * i * 10;
    covs[i] = MatrixXd::Identity(DIM, DIM);
    generators[i].Init(means[i], covs[i]);
  }

  Model m1(RANK, DIM);
  mix* p1 = *(mix**)&m1;
  p1->Initialize(weights, means, covs);

  Model m2(RANK, DIM);
  mix* p2 = *(mix**)&m2;
  p2->Initialize(weights, means, covs);

  Model m3(RANK, DIM);
  mix* p3 = *(mix**)&m3;
  p3->Initialize(weights, means, covs);

  MatrixXd data(DIM, N);
  for (int i = 0; i < N; i++) {
    data.col(i) = generators[i % RANK].Gen();
  }

  // Single train
  Trainer t1(m1);
  Timer t_single("SingleTrain");
  t_single.start();
  t1.Reset();
  for (int i = 0; i < N; i++) {
    t1.Train(data.col(i));
  }
  t1.Update();
  t_single.stop();

  double e1 = distance(m1, m2);

  // Batch train
  Trainer t2(m2);
  Timer t_batch("BatchTrain");
  t_batch.start();
  t2.Reset();
  t2.BatchTrain(data);
  t2.Update();
  t_batch.stop();

  double e2 = distance(m1, m2);

  // Fast train
  Trainer t3(m3);
  Timer t_fast("FastTrain");
  t_fast.start();
  t3.Reset();
  // 分块计算可提高精度
  for (int i = 0; i < data.cols(); i += 250) {
    t3.FastTrain(data.middleCols(i, 250));
  }
  t3.Update();
  t_fast.stop();

  double e3 = distance(m2, m3);

  std::cout << e1 << ", " << e2 << ", " << e3 << std::endl;
  if (e2 > 1.0e-6 || e3 > 1.0e-3)
    return false;
  else
    return true;
}

void DebugPredict() {
  std::vector<double> weights = {1.0 / 3, 1.0 / 3, 1.0 / 3};
  std::vector<VectorXd> means = {
    -10 * VectorXd::Ones(3), VectorXd::Zero(3), 10 * VectorXd::Ones(3)};
  std::vector<MatrixXd> covs = {MatrixXd::Identity(3, 3),
                                MatrixXd::Identity(3, 3),
                                MatrixXd::Identity(3, 3)};

  MatrixXd samples = MatrixXd::Ones(3, 3);
  samples.col(0) *= -10;
  samples.col(1) *= 0;
  samples.col(2) *= 10;

  Model m(3, 3);
  mix* p = *(mix**)&m;
  p->Initialize(weights, means, covs);

  {
    std::cout << "Single Predict" << std::endl;
    VectorXd y;
    for (int i = 0; i < 3; i++) {
      auto r = m.Predict(samples.col(i).head(2), y);
      std::cout << r << ", " << y[0] << std::endl;
    }
  }

  {
    std::cout << "Batch Predict" << std::endl;
    MatrixXd Y;
    auto R = m.BatchPredict(samples.topRows(2), Y);
    for (int i = 0; i < 3; i++) {
      std::cout << R[i] << ", " << Y.col(i)[0] << std::endl;
    }
  }

  {
    std::cout << "Fast Predict" << std::endl;
    MatrixXd Y;
    auto R = m.FastPredict(samples.topRows(2), Y);
    for (int i = 0; i < 3; i++) {
      std::cout << R[i] << ", " << Y.col(i)[0] << std::endl;
    }
  }
}

void DebugTrain() {
  std::vector<double> weights = {1.0 / 3, 1.0 / 3, 1.0 / 3};
  std::vector<VectorXd> means = {
    -10 * VectorXd::Ones(3), VectorXd::Zero(3), 10 * VectorXd::Ones(3)};
  std::vector<MatrixXd> covs = {MatrixXd::Identity(3, 3),
                                MatrixXd::Identity(3, 3),
                                MatrixXd::Identity(3, 3)};

  std::vector<MVNGenerator> generators(3);
  for (int i = 0; i < 3; i++) {
    generators[i].Init(means[i], covs[i], i + 1);
  }

  const int N = 100;

  MatrixXd samples = MatrixXd::Zero(3, N);
  for (int i = 0; i < N; i++) {
    samples.col(i) = generators[i % 3].Gen();
  }

  {
    std::cout << "Single Train" << std::endl;
    Model m(3, 3);
    mix* p = *(mix**)&m;
    p->Initialize(weights, means, covs);
    Trainer t(m);

    t.Reset();
    for (int i = 0; i < N; i++) {
      t.Train(samples.col(i));
    }
    auto e1 = t.Update();

    t.Reset();
    for (int i = 0; i < N; i++) {
      t.Train(samples.col(i));
    }
    auto e2 = t.Update();

    std::cout << e1 << ", " << e2 << std::endl;
  }

  {
    std::cout << "Batch Train" << std::endl;
    Model m(3, 3);
    mix* p = *(mix**)&m;
    p->Initialize(weights, means, covs);
    Trainer t(m);

    t.Reset();
    t.BatchTrain(samples);
    auto e1 = t.Update();

    t.Reset();
    t.BatchTrain(samples);
    auto e2 = t.Update();

    std::cout << e1 << ", " << e2 << std::endl;
  }

  {
    std::cout << "Fast Train" << std::endl;
    Model m(3, 3);
    mix* p = *(mix**)&m;
    p->Initialize(weights, means, covs);
    Trainer t(m);

    t.Reset();
    t.FastTrain(samples);
    auto e1 = t.Update();

    t.Reset();
    t.FastTrain(samples);
    auto e2 = t.Update();

    std::cout << e1 << ", " << e2 << std::endl;
  }
}

int TestPredicts() {
  // load model
  std::ifstream ifs("elite.model", std::ios::binary);
  if (!ifs) {
    std::cout << "cannot open model file" << std::endl;
    return 1;
  }
  // get file size
  ifs.seekg(0, std::ios::end);
  auto file_size = ifs.tellg();
  ifs.seekg(0, std::ios::beg);
  // read to buffer
  std::vector<char> buffer(file_size);
  ifs.read(buffer.data(), buffer.size());
  // load model data
  aha::Model model;
  if (!model.Load(buffer)) {
    std::cout << "loading model error" << std::endl;
    return 1;
  }

  Random random;

  aha::MatrixXd in = aha::MatrixXd::Zero(100, 3);
  for (int i = 0; i < 100; i++) {
    for (int j = 0; j < 3; j++) {
      in(i, j) = random.randNorm(0.0, 1.0);
    }
  }

  aha::MatrixXd out(20, 3);
  aha::VectorXd y;
  model.Predict(in.col(0), y);
  out.col(0) = y;
  model.Predict(in.col(1), y);
  out.col(1) = y;
  model.Predict(in.col(2), y);
  out.col(2) = y;
  std::cout << out << std::endl << std::endl;

  model.BatchPredict(in, out);
  std::cout << out << std::endl << std::endl;

  model.FastPredict(in, out);
  std::cout << out << std::endl;

  return 0;
}

// 测试GMM条件协方差估计
int TestPredictEx() {
  // 生成训练数据
  int N = 1000000;
  Random rand((uint32_t)std::time(0));
  MatrixXd data(3, N);

  for (int i = 0; i < N; i++) {
    double w = rand.rand();
    double a = rand.randNorm(0, 1);
    double b = rand.randNorm(0, 1);
    double c = rand.randNorm(0, 1);
    if (w < 0.2) {
      data(0, i) = a;
      data(1, i) = a + 2.0 * b + 10.0;
      data(2, i) = a + 3.0 * c + 20.0;
    } else if (w < 0.5) {
      data(0, i) = a;
      data(1, i) = a + 2.0 * b + 10.0;
      data(2, i) = a - 3.0 * c - 20.0;
    } else {
      data(0, i) = a;
      data(1, i) = a - 2.0 * b - 10.0;
      data(2, i) = a - 3.0 * c - 20.0;
    }
  }

  // 训练模型
  Model m(3, 3);
  Trainer t(m);
  for (int k = 0; k < 30; k++) {
    t.Reset();
    for (int i = 0; i < N; i += 250) {
      t.FastTrain(data.middleCols(i, 250));
    }
    double e = t.Update();
    std::cout << e << std::endl;
  }
  m.Sort();
  std::cout << m.Export() << std::endl;

  // 计算条件均值和条件方差
  VectorXd x(1);
  x(0) = -1.0;
  VectorXd y(2);
  MatrixXd cov;
  m.PredictEx(x, y, cov);
  std::cout << "\nCalulate:\n" << y << std::endl;
  std::cout << "\n" << cov << std::endl;

  // 检验条件均值和条件方差
  VectorXd mean = VectorXd::Zero(2);
  MatrixXd covar = MatrixXd::Zero(2, 2);
  for (int i = 0; i < N; i++) {
    double w = rand.rand();
    double a = x(0);
    double b = rand.randNorm(0, 1);
    double c = rand.randNorm(0, 1);
    if (w < 0.2) {
      data(0, i) = a;
      data(1, i) = a + 2.0 * b + 10.0;
      data(2, i) = a + 3.0 * c + 20.0;
    } else if (w < 0.5) {
      data(0, i) = a;
      data(1, i) = a + 2.0 * b + 10.0;
      data(2, i) = a - 3.0 * c - 20.0;
    } else {
      data(0, i) = a;
      data(1, i) = a - 2.0 * b - 10.0;
      data(2, i) = a - 3.0 * c - 20.0;
    }
    auto v = data.col(i).tail(2);
    mean += v;
    covar += v * v.transpose();
  }
  mean /= N;
  covar = covar / N - mean * mean.transpose();
  std::cout << "Verify:\n" << mean << std::endl;
  std::cout << "\n" << covar << std::endl;

  return 0;
}

// 测试GMM条件协方差批量估计
int TestBatchPredictEx() {
  // 生成训练数据
  int N = 1000000;
  Random rand((uint32_t)std::time(0));
  MatrixXd data(3, N);

  for (int i = 0; i < N; i++) {
    double w = rand.rand();
    double a = rand.randNorm(0, 1);
    double b = rand.randNorm(0, 1);
    double c = rand.randNorm(0, 1);
    if (w < 0.2) {
      data(0, i) = a;
      data(1, i) = a + 2.0 * b + 10.0;
      data(2, i) = a + 3.0 * c + 20.0;
    } else if (w < 0.5) {
      data(0, i) = a;
      data(1, i) = a + 2.0 * b + 10.0;
      data(2, i) = a - 3.0 * c - 20.0;
    } else {
      data(0, i) = a;
      data(1, i) = a - 2.0 * b - 10.0;
      data(2, i) = a - 3.0 * c - 20.0;
    }
  }

  // 训练模型
  Model m(3, 3);
  Trainer t(m);
  for (int k = 0; k < 30; k++) {
    t.Reset();
    for (int i = 0; i < N; i += 250) {
      t.FastTrain(data.middleCols(i, 250));
    }
    double e = t.Update();
    std::cout << e << std::endl;
  }
  m.Sort();
  std::cout << m.Export() << std::endl;

  // 单步计算条件均值和条件方差
  int M = 10000;
  MatrixXd Y(2, M);
  MatrixXd C(2, 2 * M);
  VectorXd y(2);
  MatrixXd c(2, 2);
  Timer t_single("Single");
  t_single.start();
  for (int i = 0; i < 10000; i++) {
    m.PredictEx(data.col(i).head(1), y, c);
    Y.col(i) = y;
    C.middleCols(i * 2, 2) = c;
  }
  t_single.stop();

  // 批量计算条件均值和条件方差
  MatrixXd _Y(2, 250);
  MatrixXd _C(2, 2 * 250);
  Timer t_batch("Batch");
  t_batch.start();
  for (int i = 0; i < 10000; i += 250) {
    // m.BatchPredictEx(data.middleCols(i, 250).topRows(1), _Y, _C);
    m.FastPredictEx(data.middleCols(i, 250).topRows(1), _Y, _C);
    Y.middleCols(i, 250) -= _Y;
    C.middleCols(i * 2, 250 * 2) -= _C;
  }
  t_batch.stop();

  std::cout << "Error = " << Y.lpNorm<Infinity>() << " " << C.lpNorm<Infinity>()
            << std::endl;
  return 0;
}

int main() {
  // TestGaussian();
  // TestRand();
  // TestTrain();
  // TestAha();
  // TestNonLinear();
  // TestMVNGenerator();
  // TestImportExport();
  // TestDumpLoad();
  // TestSpitSwallow();
  // DebugPredict();
  // DebugTrain();
  // TestPredicts();
  // TestPredictEx();
  // TestBatchPredictEx();

  std::cout << "Aha version " << aha::Version() << std::endl;

  if (FVTest()) {
    std::cout << "### FVTest OK ###" << std::endl;
  } else {
    std::cout << "*** FVTest Failed ***" << std::endl;
  }

  if (TestBatchPredict()) {
    std::cout << "### TestBatchPredict OK ###" << std::endl;
  } else {
    std::cout << "*** TestBatchPredict Failed ***" << std::endl;
  }

  if (TestBatchTrain()) {
    std::cout << "### TestBatchTrain OK ###" << std::endl;
  } else {
    std::cout << "*** TestBatchTrain Failed ***" << std::endl;
  }

  return 0;
}
