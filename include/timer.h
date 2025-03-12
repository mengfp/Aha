/*
** Copyright 2025 Meng, Fanping. All rights reserved.
*/
#ifndef AHA_TIMER_H
#define AHA_TIMER_H

#include <iostream>
#include <chrono>
#include <string>

namespace aha {

#ifdef USING_TIMER

class Timer {
 public:
  Timer(const std::string& name)
    : name(name), elapsed_time(0), count(0), running(false) {
  }

  ~Timer() {
    std::cout << "Timer " << name << ": " << count << " times, " << elapsed_time
              << " seconds." << std::endl;
  }

  // 开始计时
  void start() {
    if (!running) {
      start_time = std::chrono::high_resolution_clock::now();
      running = true;
    }
  }

  // 停止计时
  void stop() {
    if (running) {
      auto end_time = std::chrono::high_resolution_clock::now();
      elapsed_time +=
        std::chrono::duration<double>(end_time - start_time).count();
      count++;
      running = false;
    }
  }

 private:
  std::string name;
  std::chrono::high_resolution_clock::time_point start_time;
  double elapsed_time;  // 记录累积时间（单位：秒）
  uint64_t count;       // 记录累计工作次数
  bool running;
};

#else

class Timer {
 public:
  Timer(const std::string& name) {
  }

  ~Timer() {
  }

  void start() {};

  void stop() {
  }
};

#endif

class TimerGuard {
 public:
  TimerGuard(Timer& timer) : timer(timer) {
    timer.start();
  }

  ~TimerGuard() {
    timer.stop();
  }

 private:
  Timer& timer;
};

}  // namespace aha

#endif
