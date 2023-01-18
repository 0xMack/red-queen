cmake_minimum_required(VERSION 3.24)
project(RedQueen)

set(CMAKE_CXX_STANDARD 14)


include_directories(.)

add_executable(RedQueen
        Main.cpp
        Main.h DatasetLoader.cpp Population.h Individual.cpp Population.cpp Utils.h Utils.cpp)
