
cmake_minimum_required(VERSION 3.16)
project(ex4-amitat.turkel)

set(CMAKE_CXX_STANDARD 14)

add_executable(ex4-amitat.turkel Matrix.cpp Activation.cpp Dense.h MlpNetwork.cpp Dense.cpp oh_tests_exceptions.cpp)