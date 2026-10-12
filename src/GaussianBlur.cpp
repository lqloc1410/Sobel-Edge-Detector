#include "GaussianBlur.hpp"

namespace GaussianBlur{
    namespace {
        // Out-of-bounds taps are skipped and the result is renormalized by the weights actually used.
        inline uint8_t applyFilter(std::ptrdiff_t r, std::ptrdiff_t c, const BMP::GrayImage &original_pixels){
            constexpr std::ptrdiff_t halfKernel = 1;
            const std::ptrdiff_t height = static_cast<std::ptrdiff_t>(original_pixels.height);
            const std::ptrdiff_t width = static_cast<std::ptrdiff_t>(original_pixels.width);

            int sumKernel = 0;
            int sum = 0;
            for (std::ptrdiff_t dr = -halfKernel; dr <= halfKernel; ++dr){
                for (std::ptrdiff_t dc = -halfKernel; dc <= halfKernel; ++dc){
                    const std::ptrdiff_t nr = r + dr, nc = c + dc;
                    if (nr < 0 || nr >= height || nc < 0 || nc >= width){
                        continue;
                    }

                    const int weight = GaussianKernel[dr + halfKernel][dc + halfKernel];
                    sumKernel += weight;
                    sum += original_pixels(static_cast<size_t>(nr), static_cast<size_t>(nc)) * weight;
                }
            }

            return static_cast<uint8_t>(sum / sumKernel);
        }
    }

    BMP::GrayImage applyFilterAll(const BMP::GrayImage &original_pixels){
        const std::ptrdiff_t height = static_cast<std::ptrdiff_t>(original_pixels.height);
        const std::ptrdiff_t width = static_cast<std::ptrdiff_t>(original_pixels.width);
        // Every pixel is overwritten below, so allocate instead of copying the source.
        BMP::GrayImage destination(original_pixels.height, original_pixels.width);

        #pragma omp parallel for schedule(static)
        for (std::ptrdiff_t r = 0; r < height; ++r){
            for (std::ptrdiff_t c = 0; c < width; ++c){
                destination(static_cast<size_t>(r), static_cast<size_t>(c)) = applyFilter(r, c, original_pixels);
            }
        }

        return destination;
    }
}
