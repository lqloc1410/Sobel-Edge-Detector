#pragma once
#include "BMP.h"

namespace GaussianBlur{
    inline int GaussianKernel[3][3] = {{1, 2, 1}, {2, 4, 2}, {1, 2, 1}};

    void applyFilter(int r, int c, const BMP::Image &original_pixels, BMP::Image &destination_pixels);

    BMP::Image applyFilterAll(const BMP::Image &original_pixels);
}