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

set(CAM_RMI_GENERATED_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src/rmi/rmi_eval/generated" CACHE PATH "Generated RMI C++ artifact directory")
set(CAM_RMI_NAMESPACE_PREFIX "books_rmi" CACHE STRING "Generated RMI namespace prefix")
set(CAM_RMI_MODELS "linear_spline,linear" CACHE STRING "Comma-separated RMI layer models")
set(CAM_RMI_BRANCH_FACTORS
    "64;128;256;512;1024;2048;4096;8192;16384;32768;65536;131072;262144;524288;1048576;2097152"
    CACHE STRING "Generated RMI branch factors compiled into rmi_bench"
)
set(CAM_RMI_CONFIGS_FILE "" CACHE FILEPATH
    "Optional manifest of namespace|layers|branch_factor entries compiled into rmi_bench"
)

set(CAM_RMI_GENERATED_SOURCES)
set(CAM_RMI_EXPECTED_SOURCES)
set(CAM_RMI_REGISTRY_INCLUDES "#pragma once\n")
set(CAM_RMI_REGISTRY_ENTRIES "")

if(CAM_RMI_CONFIGS_FILE)
    if(NOT EXISTS "${CAM_RMI_CONFIGS_FILE}")
        message(FATAL_ERROR "CAM_RMI_CONFIGS_FILE does not exist: ${CAM_RMI_CONFIGS_FILE}")
    endif()
    file(STRINGS "${CAM_RMI_CONFIGS_FILE}" CAM_RMI_CONFIG_LINES)
    foreach(CAM_RMI_CONFIG_LINE IN LISTS CAM_RMI_CONFIG_LINES)
        string(STRIP "${CAM_RMI_CONFIG_LINE}" CAM_RMI_CONFIG_LINE)
        if(CAM_RMI_CONFIG_LINE STREQUAL "")
            continue()
        endif()
        string(REPLACE "|" ";" CAM_RMI_CONFIG_FIELDS "${CAM_RMI_CONFIG_LINE}")
        list(LENGTH CAM_RMI_CONFIG_FIELDS CAM_RMI_CONFIG_FIELD_COUNT)
        if(NOT CAM_RMI_CONFIG_FIELD_COUNT EQUAL 3)
            message(FATAL_ERROR
                "Invalid RMI config manifest entry (expected namespace|layers|branch_factor): "
                "${CAM_RMI_CONFIG_LINE}"
            )
        endif()
        list(GET CAM_RMI_CONFIG_FIELDS 0 CAM_RMI_NAMESPACE)
        list(GET CAM_RMI_CONFIG_FIELDS 1 CAM_RMI_LAYERS)
        list(GET CAM_RMI_CONFIG_FIELDS 2 CAM_RMI_BF)
        if(NOT CAM_RMI_BF MATCHES "^[0-9]+$")
            message(FATAL_ERROR "Invalid RMI branch factor in manifest: ${CAM_RMI_CONFIG_LINE}")
        endif()
        set(CAM_RMI_SOURCE "${CAM_RMI_GENERATED_DIR}/${CAM_RMI_NAMESPACE}.cpp")
        list(APPEND CAM_RMI_EXPECTED_SOURCES "${CAM_RMI_SOURCE}")
        if(NOT EXISTS "${CAM_RMI_SOURCE}")
            message(FATAL_ERROR "Generated RMI source does not exist: ${CAM_RMI_SOURCE}")
        endif()
        list(APPEND CAM_RMI_GENERATED_SOURCES "${CAM_RMI_SOURCE}")
        string(APPEND CAM_RMI_REGISTRY_INCLUDES
            "#include \"${CAM_RMI_NAMESPACE}.h\"\n"
        )
        string(APPEND CAM_RMI_REGISTRY_ENTRIES
            "    X(${CAM_RMI_NAMESPACE}, \"${CAM_RMI_LAYERS}\", ${CAM_RMI_BF}) \\\n"
        )
    endforeach()
else()
    string(REPLACE "," "_" CAM_RMI_MODEL_TAG "${CAM_RMI_MODELS}")
    string(REPLACE "-" "_" CAM_RMI_MODEL_TAG "${CAM_RMI_MODEL_TAG}")
    foreach(CAM_RMI_BF IN LISTS CAM_RMI_BRANCH_FACTORS)
        set(CAM_RMI_NAMESPACE
            "${CAM_RMI_NAMESPACE_PREFIX}_${CAM_RMI_MODEL_TAG}_${CAM_RMI_BF}"
        )
        set(CAM_RMI_SOURCE "${CAM_RMI_GENERATED_DIR}/${CAM_RMI_NAMESPACE}.cpp")
        list(APPEND CAM_RMI_EXPECTED_SOURCES "${CAM_RMI_SOURCE}")
        if(EXISTS "${CAM_RMI_SOURCE}")
            list(APPEND CAM_RMI_GENERATED_SOURCES "${CAM_RMI_SOURCE}")
            string(APPEND CAM_RMI_REGISTRY_INCLUDES
                "#include \"${CAM_RMI_NAMESPACE}.h\"\n"
            )
            string(APPEND CAM_RMI_REGISTRY_ENTRIES
                "    X(${CAM_RMI_NAMESPACE}, \"${CAM_RMI_MODELS}\", ${CAM_RMI_BF}) \\\n"
            )
        endif()
    endforeach()
endif()

set(CAM_RMI_REGISTRY_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
file(MAKE_DIRECTORY "${CAM_RMI_REGISTRY_DIR}")
file(WRITE "${CAM_RMI_REGISTRY_DIR}/rmi_bench_registry.h"
    "${CAM_RMI_REGISTRY_INCLUDES}\n#define CAM_RMI_MODEL_LIST(X) \\\n${CAM_RMI_REGISTRY_ENTRIES}\n"
)

set_source_files_properties(${CAM_RMI_EXPECTED_SOURCES} PROPERTIES GENERATED TRUE)

add_executable(rmi_bench EXCLUDE_FROM_ALL
    exp/rmi_bench.cpp
    ${CAM_RMI_GENERATED_SOURCES}
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
        ${CAM_RMI_GENERATED_DIR}
        ${CAM_RMI_REGISTRY_DIR}
)

target_include_directories(motivation_query_breakdown
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CAM_RMI_GENERATED_DIR}
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

target_compile_options(rmi_bench PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-O3>
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
        $<$<CXX_COMPILER_ID:GNU,Clang>:-O3 -Wall -Wextra -Wpedantic>
    )
    target_compile_options(tuner PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic>
    )
    target_compile_options(pgm_index_sizes PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic>
    )
endif()
