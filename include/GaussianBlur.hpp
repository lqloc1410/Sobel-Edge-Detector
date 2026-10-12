#pragma once
#include "BMP.hpp"

namespace GaussianBlur{
    inline constexpr int GaussianKernel[3][3] = {{1, 2, 1}, {2, 4, 2}, {1, 2, 1}};

    BMP::GrayImage applyFilterAll(const BMP::GrayImage &original_pixels);
}