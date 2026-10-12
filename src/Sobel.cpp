#include "../include/Sobel.hpp"
#include <iomanip>
#include <cmath>

namespace Sobel{
    struct ConvolutionResult{
        int16_t resX = 0;
        int16_t resY = 0;
    };

    inline ConvolutionResult applyConvolution(int r, int c, const BMP::Image &original_pixels){
        int kernelSize = 3;
        int halfKernel = kernelSize/2;
        int sumKernel = 0;

        int height = original_pixels.height, width = original_pixels.width;
        ConvolutionResult res;
        for (int dr = -halfKernel; dr <= halfKernel; ++dr){
            for (int dc = -halfKernel; dc <= halfKernel; ++dc){
                int nr = std::min(std::max(r + dr, 0), height - 1);
                int nc = std::min(std::max(c + dc, 0), width - 1);

                res.resX += static_cast<int16_t>(original_pixels(nr, nc).red) * SobelKernelX[dr + halfKernel][dc + halfKernel];
                res.resY += static_cast<int16_t>(original_pixels(nr, nc).red) * SobelKernelY[dr + halfKernel][dc + halfKernel];
            }
        }

        return res;
    }

    BMP::Image applyFilter(const BMP::Image &original_pixels){
        int height = original_pixels.height, width = original_pixels.width;

        std::vector<int16_t> intermediate(height*width);
        BMP::Image destination (height, width);
        
        int16_t min_val = std::numeric_limits<int16_t>::max(), max_val = 0;
        #pragma omp parallel for reduction(min:min_val) reduction(max:max_val) schedule(static)
        for (int r = 0; r < height; ++r){
            for (int c = 0; c < width; ++c){
                ConvolutionResult tmp = applyConvolution(r, c, original_pixels);
                intermediate[r*width + c] = std::sqrt((static_cast<int>(tmp.resX)*tmp.resX) + (static_cast<int>(tmp.resY)*tmp.resY));

                min_val = std::min(min_val, intermediate[r*width + c]);
                max_val = std::max(max_val, intermediate[r*width + c]);
            }
        }

        
        int range = max_val - min_val > 0 ? max_val - min_val : 1;
        #pragma omp parallel for schedule(static)
        for (int r = 0; r < height; ++r){
            for (int c = 0; c < width; ++c){
                uint8_t pixel_out = static_cast<uint8_t>((intermediate[r*width + c] - min_val)*255/(range));
                destination(r, c).blue = pixel_out;
                destination(r, c).green = pixel_out;
                destination(r, c).red = pixel_out;
            }
        }

        return destination;
    }
}

