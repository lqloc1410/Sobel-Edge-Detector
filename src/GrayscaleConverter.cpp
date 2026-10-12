#include "GrayscaleConverter.hpp"

namespace GrayscaleConverter{
    inline void grayscale_converter(BMP::RGBPixel &original){
        uint8_t converted_value = (static_cast<int>(original.blue)*29 + static_cast<int>(original.green)*150 + static_cast<int>(original.red)*77) >> 8;

        original = {converted_value, converted_value, converted_value};
    }

    void grayscale_converter_avg(BMP::RGBPixel &original){
        uint8_t converted_value = (static_cast<int>(original.blue)+ static_cast<int>(original.green) + static_cast<int>(original.red))/3;

        original = {converted_value, converted_value, converted_value};
    }

    void applyGrayscale(BMP::Image &destination){
        int height = destination.height;
        int width = destination.width;

        #pragma omp parallel for schedule(static)
        for (int r = 0; r < height; ++r){
            for (int c = 0; c < width; ++c){
                grayscale_converter(destination(r, c));
            }
        }
    }
}