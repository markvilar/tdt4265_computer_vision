# Tells the minimum required version of CMake
cmake_minimum_required(VERSION 3.5)


# Adding compilation options (warnings in this case)
if (MSVC)
	add_compile_options(/W4)
else()
	add_compile_options(-Wall -Wextra -Wpedantic)
endif()


# Add executable
add_executable(intro main.cpp)


# Tells CMake that the compiler needs C++ lambdas
target_compile_features(intro PRIVATE cxx_lambda_init_captures)
