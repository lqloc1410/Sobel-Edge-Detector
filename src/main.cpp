#include "BMP.hpp"
#include "GaussianBlur.hpp"
#include "Sobel.hpp"
#include "GPUAccelerator.hpp"
#include <iostream>
#include <new>
#include <string>
#include <chrono>
#include <utility>
#include <vector>

class ScopedTimer {
private:
    std::string name;
    std::chrono::high_resolution_clock::time_point start_time;

public:
    explicit ScopedTimer(std::string timerName) 
        : name(std::move(timerName)), 
          start_time(std::chrono::high_resolution_clock::now()) {}

    void stop() {
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end_time - start_time;
        std::cout << "[" << name << "] took " << elapsed.count() << " ms\n";
    }
};

// Reads a whole line so paths containing spaces work. Returns false on EOF or empty input.
static bool promptPath(const char *prompt, std::string &path){
    std::cout << prompt;
    return std::getline(std::cin, path) && !path.empty();
}

enum class DeviceMode { Auto, GPU, CPU };

int main(int argc, char *argv[]){
    std::string filename, output;
    DeviceMode mode = DeviceMode::Auto;

    // Parse command line arguments: [input.bmp output.bmp] [--gpu | --cpu | --auto]
    std::vector<std::string> positionalArgs;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--gpu") {
            mode = DeviceMode::GPU;
        } else if (arg == "--cpu") {
            mode = DeviceMode::CPU;
        } else if (arg == "--auto") {
            mode = DeviceMode::Auto;
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: " << argv[0] << " [input.bmp output.bmp] [--gpu | --cpu | --auto]\n\n"
                      << "Options:\n"
                      << "  --gpu   Force GPU acceleration (OpenCL)\n"
                      << "  --cpu   Force CPU multi-threading (OpenMP)\n"
                      << "  --auto  Default: Use GPU if available, fallback to CPU\n";
            return 0;
        } else {
            positionalArgs.push_back(std::move(arg));
        }
    }

    if (positionalArgs.size() == 2) {
        filename = positionalArgs[0];
        output = positionalArgs[1];
    } else if (positionalArgs.empty()) {
        if (!promptPath("Enter input file name: ", filename) ||
            !promptPath("Enter output file name: ", output)) {
            std::cerr << "Error: a file name is required.\n";
            return 1;
        }
    } else {
        std::cerr << "Usage: " << argv[0] << " [input.bmp output.bmp] [--gpu | --cpu | --auto]\n";
        return 1;
    }

    bool useGPU = false;
    if (mode == DeviceMode::GPU) {
        if (!GPUAccelerator::isAvailable()) {
            std::cerr << "Error: --gpu was requested, but no compatible OpenCL GPU was detected.\n";
            return 1;
        }
        useGPU = true;
    } else if (mode == DeviceMode::CPU) {
        useGPU = false;
    } else { // Auto
        useGPU = GPUAccelerator::isAvailable();
    }

    if (useGPU) {
        std::cout << "Using device: [GPU] " << GPUAccelerator::getDeviceName() << "\n";
    } else {
        std::cout << "Using device: [CPU] OpenMP\n";
    }

    try {
        ScopedTimer full_timer("All steps");

        // Loads directly into 1 byte/pixel grayscale; full RGB image is never held in memory.
        ScopedTimer timer("File parsing + Grayscale");
        BMP::GrayImage Data = BMP::BMPParserGray(filename);
        if (Data.empty()){
            return 1;
        }
        timer.stop();

        if (useGPU) {
            ScopedTimer gpuTimer("GPU Pipeline (Gaussian Blur + Sobel)");
            BMP::GrayImage filtered;
            if (!GPUAccelerator::runPipeline(Data, filtered)) {
                if (mode == DeviceMode::Auto) {
                    std::cerr << "Warning: GPU execution failed; falling back to CPU OpenMP...\n";
                    ScopedTimer timer3("Gaussian Blur (CPU)");
                    Data = GaussianBlur::applyFilterAll(Data);
                    timer3.stop();

                    ScopedTimer timer4("Sobel (CPU)");
                    Data = Sobel::applyFilter(Data);
                    timer4.stop();
                } else {
                    return 1;
                }
            } else {
                Data = std::move(filtered);
                gpuTimer.stop();
            }
        } else {
            ScopedTimer timer3("Gaussian Blur (CPU)");
            Data = GaussianBlur::applyFilterAll(Data);
            timer3.stop();

            ScopedTimer timer4("Sobel (CPU)");
            Data = Sobel::applyFilter(Data);
            timer4.stop();
        }

        ScopedTimer timer5("File writing");
        if (!BMP::BMPWriter(output, Data)){
            return 1;
        }
        timer5.stop();
        full_timer.stop();
    } catch (const std::bad_alloc &){
        std::cerr << "Error: not enough memory to process this image.\n";
        return 1;
    }

    return 0;
}