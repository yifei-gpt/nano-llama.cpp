cmake_minimum_required(VERSION 3.18)
project(nano-llama C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

option(NANO_CUDA "build the CUDA backend" ON)

# ggml (vendored, trimmed to CPU + CUDA)
if (NANO_CUDA)
    set(GGML_CUDA        ON CACHE BOOL "" FORCE)
    set(GGML_CUDA_GRAPHS ON CACHE BOOL "" FORCE)  # CUDA graph capture/replay for fast token generation
endif()
add_subdirectory(ggml)

# nanollama library
file(GLOB_RECURSE NANO_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/nanollama/*.cpp)
add_library(nanollama STATIC ${NANO_SOURCES})
target_include_directories(nanollama PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/ggml/include
    ${CMAKE_CURRENT_SOURCE_DIR}/vendor
    ${CMAKE_CURRENT_SOURCE_DIR}/vendor/cpp-httplib)
target_link_libraries(nanollama PUBLIC ggml ggml-base ggml-cpu)
if (NANO_CUDA)
    target_link_libraries(nanollama PUBLIC ggml-cuda)
endif()
find_package(Threads REQUIRED)
target_link_libraries(nanollama PUBLIC Threads::Threads)

# cpp-httplib (split single-header: needs its implementation compiled) — for the server
add_library(cpp-httplib STATIC vendor/cpp-httplib/httplib.cpp)
target_include_directories(cpp-httplib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/vendor/cpp-httplib)
target_compile_features(cpp-httplib PRIVATE cxx_std_17)
find_package(Threads REQUIRED)
target_link_libraries(cpp-httplib PUBLIC Threads::Threads)

# tools
foreach(tool example bench server)
    if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tools/${tool}/main.cpp)
        add_executable(nano-${tool} tools/${tool}/main.cpp)
        target_link_libraries(nano-${tool} PRIVATE nanollama)
    endif()
endforeach()
if (TARGET nano-server)
    target_link_libraries(nano-server PRIVATE cpp-httplib)
endif()
