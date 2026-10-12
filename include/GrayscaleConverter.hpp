#pragma once
#include "BMP.hpp"

namespace GrayscaleConverter{
    // Integer approximation of Rec. 601 luma: (77 R + 150 G + 29 B) / 256. Weights sum to 256, so the result never exceeds 255.
    constexpr uint8_t luminance(uint8_t red, uint8_t green, uint8_t blue){
        return static_cast<uint8_t>((red * 77 + green * 150 + blue * 29) >> 8);
    }

    constexpr uint8_t luminance(const BMP::RGBPixel &pixel){
        return luminance(pixel.red, pixel.green, pixel.blue);
    }

    // Converts an RGB image into a 1-byte-per-pixel grayscale image.
    BMP::GrayImage applyGrayscale(const BMP::Image &source);
}