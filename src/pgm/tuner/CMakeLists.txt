cmake_minimum_required(VERSION 3.16)

project(pgm_tuner LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(tuner tuner.cpp)

target_include_directories(tuner
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/..
        ${CMAKE_CURRENT_SOURCE_DIR}/../..
)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(tuner PRIVATE OpenMP::OpenMP_CXX)
endif()
