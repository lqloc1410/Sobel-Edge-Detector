#pragma once
#include "BMP.hpp"
#include <cstdint>

namespace Sobel{
    inline int16_t SobelKernelX[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    inline int16_t SobelKernelY[3][3] = {{1, 2, 1}, {0, 0, 0}, {-1, -2, -1}};

    BMP::Image applyFilter(const BMP::Image &original_pixels);
}