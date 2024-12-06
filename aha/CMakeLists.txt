cmake_minimum_required(VERSION 3.10.0)

project(aha)

set(SOURCE_FILES aha.cpp)
add_library(aha SHARED STATIC ${SOURCE_FILES})
