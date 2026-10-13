cmake_minimum_required(VERSION 3.20)

project(SobelEdgeDetector
    VERSION 1.1.0
    LANGUAGES CXX
)

find_package(OpenMP REQUIRED COMPONENTS CXX)

# Cross-platform OpenCL discovery (Windows MSYS2/SDK, Linux, macOS)
find_package(OpenCL QUIET)
if(NOT OpenCL_FOUND)
    find_library(OpenCL_LIBRARY NAMES OpenCL libOpenCL PATHS
        "C:/Windows/System32"
        "C:/msys64/ucrt64/lib"
        "C:/msys64/clang64/lib"
        "/usr/lib"
        "/usr/local/lib"
        "/usr/lib/x86_64-linux-gnu"
    )
    find_path(OpenCL_INCLUDE_DIR NAMES CL/cl.h OpenCL/cl.h PATHS
        "C:/msys64/ucrt64/include"
        "C:/msys64/clang64/include"
        "/usr/include"
        "/usr/local/include"
    )
    if(OpenCL_LIBRARY AND OpenCL_INCLUDE_DIR)
        set(OpenCL_FOUND TRUE)
        set(OpenCL_LIBRARIES "${OpenCL_LIBRARY}")
        set(OpenCL_INCLUDE_DIRS "${OpenCL_INCLUDE_DIR}")
        add_library(OpenCL::OpenCL UNKNOWN IMPORTED)
        set_target_properties(OpenCL::OpenCL PROPERTIES
            IMPORTED_LOCATION "${OpenCL_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${OpenCL_INCLUDE_DIR}"
        )
    endif()
endif()

add_executable(SobelEdgeDetector
    src/main.cpp
    src/BMP.cpp
    src/GrayscaleConverter.cpp
    src/GaussianBlur.cpp
    src/Sobel.cpp
    src/GPUAccelerator.cpp
)

target_compile_features(SobelEdgeDetector PRIVATE cxx_std_23)

target_include_directories(SobelEdgeDetector PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

target_link_libraries(SobelEdgeDetector PRIVATE OpenMP::OpenMP_CXX)

if(OpenCL_FOUND)
    message(STATUS "GPU Acceleration: OpenCL enabled (${OpenCL_LIBRARIES})")
    target_compile_definitions(SobelEdgeDetector PRIVATE HAVE_OPENCL=1)
    target_link_libraries(SobelEdgeDetector PRIVATE OpenCL::OpenCL)
else()
    message(WARNING "GPU Acceleration: OpenCL not found. Compiling in CPU-only mode.")
endif()

target_compile_options(SobelEdgeDetector PRIVATE
    -O3
    -march=native
    -funroll-loops
    -fomit-frame-pointer
    -Wall
    -Wextra
)