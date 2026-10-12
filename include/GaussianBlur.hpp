#pragma once
#include "BMP.hpp"

namespace GaussianBlur{
    inline int GaussianKernel[3][3] = {{1, 2, 1}, {2, 4, 2}, {1, 2, 1}};

    BMP::Image applyFilterAll(const BMP::Image &original_pixels);
}