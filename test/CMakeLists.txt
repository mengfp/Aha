cmake_minimum_required(VERSION 3.10.0)

project(test)

set(SOURCE_FILES test.cpp)
add_executable(test ${SOURCE_FILES})
target_link_libraries(test aha)
