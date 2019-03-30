# cmake version
cmake_minimum_required(VERSION 3.0)


# project name and version
project(klt_tracker VERSION 1.0.0)


# flags
add_compile_options(-Wall -Wextra -Wpedantic)


# files
file(GLOB hello_SRC "${PROJECT_SOURCE_DIR}/src/hello/*.cpp")
file(GLOB hello_HDR "${PROJECT_SOURCE_DIR}/src/hello/*.hpp")

add_library(hello ${hello_SRC})


# include
include_directories(include src libs)


# target
add_executable(main "${PROJECT_SOURCE_DIR}/src/main.cpp")


# linking
#add_subdirectory(./libs/external_lib2)
#target_link_libraries(main external_lib2)

target_link_libraries(main hello)


# unit testing
