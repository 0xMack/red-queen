cmake_minimum_required(VERSION 3.15...3.27)

project(redqueen LANGUAGES CXX)

set(CMAKE_VERBOSE_MAKEFILE ON)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

execute_process(
        COMMAND ${Python3_EXECUTABLE} -m pybind11 --cmakedir
        OUTPUT_VARIABLE PYBIND11_MODULE_PATH
        OUTPUT_STRIP_TRAILING_WHITESPACE
)

message(STATUS "pybind11 CMake directory: ${PYBIND11_MODULE_PATH}")
# Verify that the specified module path exists
if(NOT EXISTS ${PYBIND11_MODULE_PATH}/pybind11Config.cmake)
    message(FATAL_ERROR "Invalid pybind11 module path: ${PYBIND11_MODULE_PATH}")
endif()

# Include the pybind11Config.cmake file from the specified module path
list(APPEND CMAKE_PREFIX_PATH ${PYBIND11_MODULE_PATH})

find_package(pybind11 REQUIRED CONFIG)

file (GLOB SOURCE_FILES "src/linear/*.cpp")
file (GLOB HEADER_FILES "src/linear/*.h")

pybind11_add_module(redqueen ${SOURCE_FILES} ${HEADER_FILES})
target_compile_definitions(redqueen PRIVATE VERSION_INFO=${PROJECT_VERSION})
install(TARGETS redqueen
        LIBRARY DESTINATION .  # installs to site-packages root
        ARCHIVE DESTINATION .  # needed for Windows static libs
        RUNTIME DESTINATION .) # for Windows .dlls / .pyds
