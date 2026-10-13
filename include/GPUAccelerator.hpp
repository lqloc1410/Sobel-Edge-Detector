#pragma once
#include "BMP.hpp"
#include <string>

namespace GPUAccelerator {
    // Checks if a compatible OpenCL GPU device is detected and operational.
    bool isAvailable();

    // Returns the descriptive name of the active GPU device (or "None").
    std::string getDeviceName();

    // Executes the complete accelerated pipeline (Gaussian Blur -> Sobel Filter)
    // on the GPU. Input and intermediate data remain in GPU VRAM to avoid PCIe round-trips.
    // Returns true on success, or false if an error occurred (reason printed to std::cerr).
    bool runPipeline(const BMP::GrayImage &input, BMP::GrayImage &output);

    // Individual filter operations on GPU (allocates temporary device buffers)
    bool applyGaussianBlur(const BMP::GrayImage &input, BMP::GrayImage &output);
    bool applySobel(const BMP::GrayImage &input, BMP::GrayImage &output);
}
