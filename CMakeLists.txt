cmake_minimum_required(VERSION 3.20)

project(SobelEdgeDetector
    VERSION 1.0.0
    LANGUAGES CXX
)

find_package(OpenMP REQUIRED COMPONENTS CXX)

add_executable(SobelEdgeDetector
    src/main.cpp
    src/BMP.cpp
    src/GrayscaleConverter.cpp
    src/GaussianBlur.cpp
    src/Sobel.cpp
)

target_compile_features(SobelEdgeDetector PRIVATE cxx_std_23)

target_include_directories(SobelEdgeDetector PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

target_link_libraries(SobelEdgeDetector PRIVATE OpenMP::OpenMP_CXX)

target_compile_options(SobelEdgeDetector PRIVATE
    -O3
    -flto
    -march=native
    -funroll-loops
    -fomit-frame-pointer
)