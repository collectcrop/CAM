cmake_minimum_required(VERSION 3.16)

project(CAM LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(CAM_ENABLE_WARNINGS "Enable common compiler warnings" ON)

add_library(cam_cache STATIC
    src/cache/CacheFactory.cpp
)

target_include_directories(cam_cache
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

if(CAM_ENABLE_WARNINGS)
    target_compile_options(cam_cache PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic>
    )
endif()

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(cam_cache PUBLIC OpenMP::OpenMP_CXX)
endif()

add_executable(pgm_bench
    exp/pgm_bench.cpp
)

add_executable(pgm_range_bench
    exp/pgm_range_bench.cpp
)

add_executable(pgm_cam_covariance
    exp/pgm_cam_covariance.cpp
)

add_executable(sort_cache_tradeoff
    exp/sort_cache_tradeoff.cpp
)

add_executable(pgm_cache_simulate
    exp/pgm_cache_simulate.cpp
)

add_executable(pgm_range_cache_simulate
    exp/pgm_range_cache_simulate.cpp
)

add_executable(rmi_bench
    exp/rmi_bench.cpp
    src/rmi/rmi_eval/generated/books_rmi_linear_spline_linear_64.cpp
    src/rmi/rmi_eval/generated/books_rmi_linear_spline_linear_128.cpp
    src/rmi/rmi_eval/generated/books_rmi_linear_spline_linear_256.cpp
    src/rmi/rmi_eval/generated/books_rmi_linear_spline_linear_512.cpp
    src/rmi/rmi_eval/generated/books_rmi_linear_spline_linear_1024.cpp
    src/rmi/rmi_eval/generated/books_rmi_linear_spline_linear_2048.cpp
    src/rmi/rmi_eval/generated/books_rmi_linear_spline_linear_4096.cpp
    src/rmi/rmi_eval/generated/books_rmi_linear_spline_linear_8192.cpp
    src/rmi/rmi_eval/generated/books_rmi_linear_spline_linear_16384.cpp
    src/rmi/rmi_eval/generated/books_rmi_linear_spline_linear_32768.cpp
    src/rmi/rmi_eval/generated/books_rmi_linear_spline_linear_65536.cpp
    src/rmi/rmi_eval/generated/books_rmi_linear_spline_linear_131072.cpp
    src/rmi/rmi_eval/generated/books_rmi_linear_spline_linear_262144.cpp
    src/rmi/rmi_eval/generated/books_rmi_linear_spline_linear_524288.cpp
    src/rmi/rmi_eval/generated/books_rmi_linear_spline_linear_1048576.cpp
    src/rmi/rmi_eval/generated/books_rmi_linear_spline_linear_2097152.cpp
)

add_executable(motivation_query_breakdown
    exp/motivation_query_breakdown.cpp
)

add_executable(pgm_join_fit
    exp/pgm_join_fit.cpp
)

add_executable(pgm_hybrid_join
    exp/pgm_hybrid_join.cpp
)

add_executable(tuner
    src/pgm/tuner/tuner.cpp
)

add_executable(pgm_index_sizes
    exp/pgm_index_sizes.cpp
)

target_include_directories(pgm_bench
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_include_directories(pgm_range_bench
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_include_directories(pgm_cam_covariance
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_include_directories(sort_cache_tradeoff
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_include_directories(pgm_cache_simulate
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_include_directories(pgm_range_cache_simulate
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_include_directories(rmi_bench
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/src/rmi/rmi_eval/generated
)

target_include_directories(motivation_query_breakdown
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/src/rmi/rmi_eval/generated
)

target_include_directories(pgm_join_fit
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_include_directories(pgm_hybrid_join
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_include_directories(tuner
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/src/pgm/tuner
)

target_include_directories(pgm_index_sizes
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(pgm_bench
    PRIVATE
        cam_cache
)

target_link_libraries(pgm_range_bench
    PRIVATE
        cam_cache
)

target_link_libraries(pgm_cam_covariance
    PRIVATE
        cam_cache
)

target_link_libraries(sort_cache_tradeoff
    PRIVATE
        cam_cache
)

target_link_libraries(pgm_cache_simulate
    PRIVATE
        cam_cache
)

target_link_libraries(pgm_range_cache_simulate
    PRIVATE
        cam_cache
)

target_link_libraries(rmi_bench
    PRIVATE
        cam_cache
)

target_link_libraries(motivation_query_breakdown
    PRIVATE
        cam_cache
)

target_link_libraries(pgm_join_fit
    PRIVATE
        cam_cache
)

target_link_libraries(pgm_hybrid_join
    PRIVATE
        cam_cache
)

target_link_libraries(tuner
    PRIVATE
        cam_cache
)

target_link_libraries(pgm_index_sizes
    PRIVATE
        cam_cache
)

if(CAM_ENABLE_WARNINGS)
    target_compile_options(pgm_bench PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic>
    )
    target_compile_options(pgm_range_bench PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic>
    )
    target_compile_options(pgm_cam_covariance PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic>
    )
    target_compile_options(sort_cache_tradeoff PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic>
    )
    target_compile_options(pgm_cache_simulate PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic>
    )
    target_compile_options(pgm_range_cache_simulate PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic>
    )
    target_compile_options(rmi_bench PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic>
    )
    target_compile_options(motivation_query_breakdown PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic>
    )
    target_compile_options(pgm_join_fit PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic>
    )
    target_compile_options(pgm_hybrid_join PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic>
    )
    target_compile_options(tuner PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic>
    )
    target_compile_options(pgm_index_sizes PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic>
    )
endif()
