#pragma once
#include "BMP.hpp"
#include <cstdint>

namespace Sobel{
    inline constexpr int SobelKernelX[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    inline constexpr int SobelKernelY[3][3] = {{1, 2, 1}, {0, 0, 0}, {-1, -2, -1}};

    BMP::GrayImage applyFilter(const BMP::GrayImage &original_pixels);
}