#include "Sobel.hpp"
#include <algorithm>
#include <cmath>
#include <limits>

namespace Sobel{
    namespace {
        // Gradient magnitude at (r, c), clamping out-of-bounds taps to the nearest edge pixel.
        // |Gx|, |Gy| <= 4 * 255, so the magnitude is at most ~1443 and fits in int16_t.
        inline int16_t gradientMagnitude(std::ptrdiff_t r, std::ptrdiff_t c, const BMP::GrayImage &original_pixels){
            constexpr std::ptrdiff_t halfKernel = 1;
            const std::ptrdiff_t height = static_cast<std::ptrdiff_t>(original_pixels.height);
            const std::ptrdiff_t width = static_cast<std::ptrdiff_t>(original_pixels.width);

            int gradientX = 0;
            int gradientY = 0;
            for (std::ptrdiff_t dr = -halfKernel; dr <= halfKernel; ++dr){
                for (std::ptrdiff_t dc = -halfKernel; dc <= halfKernel; ++dc){
                    const std::ptrdiff_t nr = std::clamp<std::ptrdiff_t>(r + dr, 0, height - 1);
                    const std::ptrdiff_t nc = std::clamp<std::ptrdiff_t>(c + dc, 0, width - 1);
                    const int value = original_pixels(static_cast<size_t>(nr), static_cast<size_t>(nc));

                    gradientX += value * SobelKernelX[dr + halfKernel][dc + halfKernel];
                    gradientY += value * SobelKernelY[dr + halfKernel][dc + halfKernel];
                }
            }

            return static_cast<int16_t>(std::sqrt(static_cast<double>(gradientX * gradientX + gradientY * gradientY)));
        }
    }

    BMP::GrayImage applyFilter(const BMP::GrayImage &original_pixels){
        const std::ptrdiff_t height = static_cast<std::ptrdiff_t>(original_pixels.height);
        const std::ptrdiff_t width = static_cast<std::ptrdiff_t>(original_pixels.width);
        BMP::GrayImage destination(original_pixels.height, original_pixels.width);
        if (destination.empty()){
            return destination;
        }

        // Pass 1: find the magnitude range for normalization. Magnitudes are recomputed in
        // pass 2 instead of being stored, which avoids a 2-bytes-per-pixel intermediate buffer.
        // The recomputation is cheap next to the memory traffic it saves on large images.
        int16_t min_val = std::numeric_limits<int16_t>::max(), max_val = 0;
        #pragma omp parallel for reduction(min:min_val) reduction(max:max_val) schedule(static)
        for (std::ptrdiff_t r = 0; r < height; ++r){
            for (std::ptrdiff_t c = 0; c < width; ++c){
                const int16_t magnitude = gradientMagnitude(r, c, original_pixels);
                min_val = std::min(min_val, magnitude);
                max_val = std::max(max_val, magnitude);
            }
        }

        // Pass 2: recompute and scale into [0, 255].
        const int range = max_val - min_val > 0 ? max_val - min_val : 1;
        #pragma omp parallel for schedule(static)
        for (std::ptrdiff_t r = 0; r < height; ++r){
            for (std::ptrdiff_t c = 0; c < width; ++c){
                const int magnitude = gradientMagnitude(r, c, original_pixels);
                destination(static_cast<size_t>(r), static_cast<size_t>(c)) =
                    static_cast<uint8_t>((magnitude - min_val) * 255 / range);
            }
        }

        return destination;
    }
}
