cmake_minimum_required(VERSION 3.10)
project(filon_cpp CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(OpenMP REQUIRED)

include_directories(inc)
include_directories(src)

file(GLOB SOURCES "src/*.cc")

add_library(filon SHARED ${SOURCES})

if(OpenMP_CXX_FOUND)
    target_link_libraries(filon PUBLIC OpenMP::OpenMP_CXX)
endif()

target_compile_definitions(filon PRIVATE FILON_EXPORTS)

# Ensure the output name is consistent (libfilon.* across platforms)
set_target_properties(filon PROPERTIES OUTPUT_NAME "filon" PREFIX "lib")
