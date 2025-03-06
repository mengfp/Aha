/*
** Copyright 2025 Meng, Fanping. All rights reserved.
*/
#ifndef AHA_EIGEN_H
#define AHA_EIGEN_H

#include <Eigen/Dense>

namespace aha {

template <typename T>
using Vector = Eigen::Matrix<T, -1, 1>;

template <typename T>
using Matrix = Eigen::Matrix<T, -1, -1>;

}

#endif
