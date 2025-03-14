cmake_minimum_required(VERSION 3.10.0)

project(aha)

set(SOURCE_FILES aha.cpp)
add_library(aha STATIC ${SOURCE_FILES})

install(TARGETS aha DESTINATION lib)
install(FILES ../include/aha.h DESTINATION include)
