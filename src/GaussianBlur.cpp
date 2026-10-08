#include "../include/BMP.h"
#include "../include/GaussianBlur.h"
//using namespace std;
namespace GaussianBlur{
    inline void applyFilter(int r, int c, const BMP::Image &original_pixels, BMP::Image &destination_pixels){
        int kernelSize = 3;
        int halfKernel = kernelSize/2;
        int sumKernel = 0;

        int height = original_pixels.height, width = original_pixels.width;
        int sum_red = 0, sum_green = 0, sum_blue = 0;
        for (int dr = -halfKernel; dr <= halfKernel; ++dr){
            for (int dc = -halfKernel; dc <= halfKernel; ++dc){
                int nr = r + dr, nc = c + dc;
                if (nr < 0 or nr >= height or nc < 0 or nc >= width){
                    continue;
                }

                sumKernel += GaussianKernel[dr + halfKernel][dc + halfKernel];
                sum_red += static_cast<int>(original_pixels(nr, nc).red) * GaussianKernel[dr + halfKernel][dc + halfKernel];
                sum_green += static_cast<int>(original_pixels(nr, nc).green) * GaussianKernel[dr + halfKernel][dc + halfKernel];
                sum_blue += static_cast<int>(original_pixels(nr, nc).blue) * GaussianKernel[dr + halfKernel][dc + halfKernel];
            }
        }

        destination_pixels(r, c) = {static_cast<uint8_t>(sum_blue/sumKernel), static_cast<uint8_t>(sum_green/sumKernel), static_cast<uint8_t>(sum_red/sumKernel)};
    }

    BMP::Image applyFilterAll(const BMP::Image &original_pixels){
        int height = original_pixels.height, width = original_pixels.width;
        BMP::Image destination = original_pixels;
        
        #pragma omp parallel for schedule(static)
        for (int r = 0; r < height; ++r){
            for (int c = 0; c < width; ++c){
                applyFilter(r, c, original_pixels, destination);
            }
        }

        return destination;
    }
}
