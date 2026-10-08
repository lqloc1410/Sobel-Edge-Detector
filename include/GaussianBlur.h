#pragma once

namespace GaussianBlur{
    inline std::vector<std::vector<int>> GaussianKernel = {{1, 2, 1}, {2, 4, 2}, {1, 2, 1}};

    void applyFilter(int r, int c, const BMP::Image &original_pixels, BMP::Image &destination_pixels);

    BMP::Image applyFilterAll(const BMP::Image &original_pixels);
}