#pragma once

namespace GaussianBlur{
    inline std::vector<std::vector<int>> GaussianKernel = {{1, 2, 1}, {2, 4, 2}, {1, 2, 1}};

    void applyFilter(int r, int c, const std::vector<std::vector<BMP::RGBPixel>> &original_pixels, std::vector<std::vector<BMP::RGBPixel>> &destination_pixels);

    std::vector<std::vector<BMP::RGBPixel>> applyFilterAll(const std::vector<std::vector<BMP::RGBPixel>> &original_pixels);
}