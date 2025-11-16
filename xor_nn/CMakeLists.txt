cmake_minimum_required(VERSION 3.14)
project(xor_neural_net CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Release build gives better default optimisations
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_DEBUG   "-g -O0 -Wall -Wextra")
set(CMAKE_CXX_FLAGS_RELEASE "-O2 -Wall -Wextra")

# ---- Sources ----------------------------------------------------------------
set(SOURCES
    src/main.cpp
    src/neural_network.cpp
    src/matrix.cpp
    data/xor_data.cpp
)

# ---- Include paths ----------------------------------------------------------
include_directories(
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/data
)

# ---- Executable -------------------------------------------------------------
add_executable(xor_nn ${SOURCES})
