#set(CMAKE_VERBOSE_MAKEFILE ON)
#
#cmake_minimum_required(VERSION 3.20)
#project(redqueen)
#
#set(CMAKE_CXX_STANDARD 14)
#
#find_package(Python COMPONENTS Interpreter Development)
#message("Python Location: ${Python_EXECUTABLE}")
#set(pybind11_DIR "venv/Lib/site-packages/pybind11/share/cmake/pybind11")
#
#find_package(pybind11 REQUIRED)
#

#pybind11_add_module(redqueen ${SOURCE_FILES} ${HEADER_FILES})
#
##add_executable(RedQueenTest ${SOURCE_FILES} ${HEADER_FILES})
#
#target_compile_definitions(redqueen
#        PRIVATE VERSION_INFO=${REDQUEEN_VERSION_INFO})
#
### pybind11 method:
##pybind11_add_module(redqueen ${SOURCE_FILES} ${HEADER_FILES})
#
##install(TARGETS redqueen
##        COMPONENT Python
##        LIBRARY DESTINATION "C:/Users/mackp/dev/RedQueen/venv/Lib/site-packages")


project(binding_example)
cmake_minimum_required(VERSION 3.15...3.25)

#project(${SKBUILD_PROJECT_NAME} VERSION ${SKBUILD_PROJECT_VERSION})


# Explicitly finding pip installed pybind11 config to resolve bug on windows
execute_process(COMMAND pybind11-config --cmakedir OUTPUT_VARIABLE PYBIND11_MODULE_PATH OUTPUT_STRIP_TRAILING_WHITESPACE)

# Display the value of the variable for verification (optional)
message(STATUS "pybind11 CMake directory: ${PYBIND11_MODULE_PATH}")

# Verify that the specified module path exists
if(NOT EXISTS ${PYBIND11_MODULE_PATH}/pybind11Config.cmake)
    message(FATAL_ERROR "Invalid pybind11 module path: ${PYBIND11_MODULE_PATH}")
endif()

# Include the pybind11Config.cmake file from the specified module path
list(APPEND CMAKE_PREFIX_PATH ${PYBIND11_MODULE_PATH})

find_package(Python REQUIRED COMPONENTS Interpreter Development)

find_package(pybind11 REQUIRED)

#find_package(pybind11 CONFIG REQUIRED)
file (GLOB SOURCE_FILES "src/linear_gp/bindtest.cpp")
file (GLOB HEADER_FILES "src/linear_gp/bindtest.h")
pybind11_add_module(redqueen ${SOURCE_FILES} ${HEADER_FILES})


#target_compile_definitions(_core PRIVATE VERSION_INFO=${PROJECT_VERSION})
#install(TARGETS redqueen
#        COMPONENT Python
#        LIBRARY DESTINATION "C:/Users/mackp/dev/RedQueen/venv/Lib/site-packages")
#install(TARGETS _core DESTINATION RedQueen)
