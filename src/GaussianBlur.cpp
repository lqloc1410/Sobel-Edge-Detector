#include "../include/BMP.h"
#include "../include/GaussianBlur.h"
//using namespace std;
namespace GaussianBlur{
    void applyFilter(int r, int c, const BMP::Image &original_pixels, BMP::Image &destination_pixels){
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

    struct HorizontalPixel {
        int blue;
        int green;
        int red;
        int weight;
    };

    BMP::Image applyFilterAll(const BMP::Image &original_pixels){
        int height = original_pixels.height, width = original_pixels.width;
        std::vector<HorizontalPixel> horizontal(static_cast<std::size_t>(height) * width);
        BMP::Image destination(height, width);

        #pragma omp parallel for schedule(static)
        for (int r = 0; r < height; ++r){
            for (int c = 0; c < width; ++c){
                HorizontalPixel value{0, 0, 0, 0};
                for (int dc = -1; dc <= 1; ++dc){
                    const int nc = c + dc;
                    if (nc < 0 || nc >= width){
                        continue;
                    }

                    const int weight = GaussianKernel[1][dc + 1];
                    const auto& pixel = original_pixels(r, nc);
                    value.blue += static_cast<int>(pixel.blue) * weight;
                    value.green += static_cast<int>(pixel.green) * weight;
                    value.red += static_cast<int>(pixel.red) * weight;
                    value.weight += weight;
                }

                horizontal[static_cast<std::size_t>(r) * width + c] = value;
            }
        }

        #pragma omp parallel for schedule(static)
        for (int r = 0; r < height; ++r){
            for (int c = 0; c < width; ++c){
                int sum_blue = 0, sum_green = 0, sum_red = 0;
                int vertical_weight = 0;

                for (int dr = -1; dr <= 1; ++dr){
                    const int nr = r + dr;
                    if (nr < 0 || nr >= height){
                        continue;
                    }

                    const int weight = GaussianKernel[dr + 1][1];
                    const auto& pixel = horizontal[static_cast<std::size_t>(nr) * width + c];
                    sum_blue += pixel.blue * weight;
                    sum_green += pixel.green * weight;
                    sum_red += pixel.red * weight;
                    vertical_weight += weight;
                }

                const int horizontal_weight =
                    horizontal[static_cast<std::size_t>(r) * width + c].weight;
                const int divisor = horizontal_weight * vertical_weight;
                destination(r, c) = {
                    static_cast<uint8_t>(sum_blue / divisor),
                    static_cast<uint8_t>(sum_green / divisor),
                    static_cast<uint8_t>(sum_red / divisor)
                };
            }
        }

        return destination;
    }
}
