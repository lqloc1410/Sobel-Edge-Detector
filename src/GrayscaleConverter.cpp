#include "../include/GrayscaleConverter.h"

namespace GrayscaleConverter{
    void grayscale_converter(BMP::RGBPixel &original){
        uint8_t converted_value = (static_cast<float>(original.blue)*0.114 + static_cast<float>(original.green)*0.587 + static_cast<float>(original.red)*0.299);

        original = {converted_value, converted_value, converted_value};
    }

    void grayscale_converter_avg(BMP::RGBPixel &original){
        uint8_t converted_value = (static_cast<int>(original.blue)+ static_cast<int>(original.green) + static_cast<int>(original.red))/3;

        original = {converted_value, converted_value, converted_value};
    }

    void applyGrayscale(BMP::Image &destination){
        int height = destination.height;
        int width = destination.width;

        for (int r = 0; r < height; ++r){
            for (int c = 0; c < width; ++c){
                grayscale_converter(destination(r, c));
            }
        }
    }
}