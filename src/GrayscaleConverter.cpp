#include "GrayscaleConverter.hpp"

namespace GrayscaleConverter{
    BMP::GrayImage applyGrayscale(const BMP::Image &source){
        BMP::GrayImage destination(source.height, source.width);
        const std::ptrdiff_t count = static_cast<std::ptrdiff_t>(source.data.size());

        #pragma omp parallel for schedule(static)
        for (std::ptrdiff_t i = 0; i < count; ++i){
            destination.data[static_cast<size_t>(i)] = luminance(source.data[static_cast<size_t>(i)]);
        }

        return destination;
    }
}