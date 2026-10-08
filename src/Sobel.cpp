#include "../include/BMP.h"
#include "../include/GaussianBlur.h"
#include "../include/GrayscaleConverter.h"
#include <iomanip>
#include <cmath>
#include <chrono>
#include <string_view>

std::vector<std::vector<int>> SobelKernelX = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
std::vector<std::vector<int>> SobelKernelY = {{1, 2, 1}, {0, 0, 0}, {-1, -2, -1}};

class Timer {
public:
    // Store the start time and an optional name for identification
    explicit Timer(std::string_view name = "Timer")
        : m_name(name), m_start(std::chrono::steady_clock::now()) {}

    // Destructor calculates and prints the elapsed time
    void StopTimer() {
        auto end = std::chrono::steady_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end - m_start;
        std::cout << "[" << m_name << "] Elapsed time: " << elapsed.count() << " ms\n";
    }

    // Disable copy/move to prevent accidental duplicate timing prints
    Timer(const Timer&) = delete;
    Timer& operator=(const Timer&) = delete;
    Timer(Timer&&) = delete;
    Timer& operator=(Timer&&) = delete;

private:
    std::string_view m_name;
    std::chrono::steady_clock::time_point m_start;
};

struct ConvolutionResult{
    int resX = 0;
    int resY = 0;
};

ConvolutionResult applyConvolution(int r, int c, const BMP::Image &original_pixels){
    int kernelSize = SobelKernelX.size();
    int halfKernel = kernelSize/2;
    int sumKernel = 0;

    int height = original_pixels.height, width = original_pixels.width;
    ConvolutionResult res;
    for (int dr = -halfKernel; dr <= halfKernel; ++dr){
        for (int dc = -halfKernel; dc <= halfKernel; ++dc){
            int nr = std::min(std::max(r + dr, 0), height - 1);
            int nc = std::min(std::max(c + dc, 0), width - 1);

            res.resX += static_cast<int>(original_pixels(nr, nc).red) * SobelKernelX[dr + halfKernel][dc + halfKernel];
            res.resY += static_cast<int>(original_pixels(nr, nc).red) * SobelKernelY[dr + halfKernel][dc + halfKernel];
        }
    }

    return res;
}

BMP::Image applyFilter(const BMP::Image &original_pixels){
    int height = original_pixels.height, width = original_pixels.width;

    std::vector<std::vector<int>> intermediate(height, std::vector<int>(width, 0));
    BMP::Image destination = original_pixels;
    
    int min_val = INT_MAX/2, max_val = 0;
    for (int r = 0; r < height; ++r){
        for (int c = 0; c < width; ++c){
            ConvolutionResult tmp = applyConvolution(r, c, original_pixels);
            intermediate[r][c] = sqrt(tmp.resX*tmp.resX + tmp.resY*tmp.resY);

            min_val = std::min(min_val, intermediate[r][c]);
            max_val = std::max(max_val, intermediate[r][c]);
        }
    }

    int range = max_val - min_val > 0 ? max_val - min_val : 1;
    for (int r = 0; r < height; ++r){
        for (int c = 0; c < width; ++c){
            uint8_t pixel_out = static_cast<uint8_t>((intermediate[r][c] - min_val)*255/(range));
            destination(r, c).blue = pixel_out;
            destination(r, c).green = pixel_out;
            destination(r, c).red = pixel_out;
        }
    }

    return destination;
}

int main(){
    std::cout << "Enter file name: ";
    
    std::string filename;
    std::cin >> filename;

    Timer total("Full program");

    Timer timer("Image parsing");
    BMP::Image image = BMP::BMPParser(filename);
    timer.StopTimer();
    
    Timer timer2("Apply grayscale");
    GrayscaleConverter::applyGrayscale(image);
    timer2.StopTimer();

    Timer timer3("Apply blur");
    BMP::Image image_blurred = GaussianBlur::applyFilterAll(image);
    timer3.StopTimer();

    Timer timer4("Apply Sobel");
    BMP::Image image_edge = applyFilter(image_blurred);
    timer4.StopTimer();

    Timer timer5("Write file");
    BMP::BMPWriter("result.bmp", image_edge);
    timer5.StopTimer();
    total.StopTimer();
}