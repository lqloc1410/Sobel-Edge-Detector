#pragma once
#include "../include/BMP.h"

namespace GrayscaleConverter{
    void grayscale_converter(BMP::RGBPixel &original);

    void grayscale_converter_avg(BMP::RGBPixel &original);
    
    void applyGrayscale(BMP::Image &destination);
}