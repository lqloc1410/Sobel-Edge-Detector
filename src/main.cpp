#include "BMP.hpp"
#include "GaussianBlur.hpp"
#include "Sobel.hpp"
#include <iostream>
#include <new>
#include <string>
#include <chrono>
#include <utility>

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

int main(int argc, char *argv[]){
    // Usage: SobelEdgeDetector [input.bmp output.bmp]  (prompts if no arguments are given)
    std::string filename, output;
    if (argc == 3){
        filename = argv[1];
        output = argv[2];
    } else if (argc == 1){
        if (!promptPath("Enter input file name: ", filename) ||
            !promptPath("Enter output file name: ", output)){
            std::cerr << "Error: a file name is required.\n";
            return 1;
        }
    } else {
        std::cerr << "Usage: " << argv[0] << " [input.bmp output.bmp]\n";
        return 1;
    }

    try {
        ScopedTimer full_timer("All steps");

        // Loads straight into 1 byte/pixel grayscale; the full RGB image is never held in memory.
        ScopedTimer timer("File parsing + Grayscale");
        BMP::GrayImage Data = BMP::BMPParserGray(filename);
        if (Data.empty()){
            return 1; // The parser has already printed the reason.
        }
        timer.stop();

        ScopedTimer timer3("Gaussian Blur");
        Data = GaussianBlur::applyFilterAll(Data);
        timer3.stop();

        ScopedTimer timer4("Sobel");
        Data = Sobel::applyFilter(Data);
        timer4.stop();

        ScopedTimer timer5("File writing");
        if (!BMP::BMPWriter(output, Data)){
            return 1; // The writer has already printed the reason.
        }
        timer5.stop();
        full_timer.stop();
    } catch (const std::bad_alloc &){
        std::cerr << "Error: not enough memory to process this image.\n";
        return 1;
    }

    return 0;
}