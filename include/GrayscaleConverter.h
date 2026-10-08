#pragma once
#include "../include/BMP.h"

namespace GrayscaleConverter{
    void grayscale_converter(BMP::RGBPixel &original);

    void grayscale_converter_avg(BMP::RGBPixel &original);
    
    void grayscale_converter_all(std::vector<std::vector<BMP::RGBPixel>> &destination);
}