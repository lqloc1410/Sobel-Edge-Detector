#include "BMP.hpp"
#include "GaussianBlur.hpp"
#include "GrayscaleConverter.hpp"
#include "Sobel.hpp"
#include <iostream>
#include <string>
#include <chrono>

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

int main(){
    std::cout << "Enter input file name: ";
    std::string filename;
    std::cin >> filename;

    std::cout << "Enter output file name: ";
    std::string output;
    std::cin >> output;

    ScopedTimer full_timer("All steps");
    ScopedTimer timer("File parsing");
    BMP::Image Data = BMP::BMPParser(filename);
    timer.stop();

    ScopedTimer timer2("Grayscale");
    GrayscaleConverter::applyGrayscale(Data);
    timer2.stop();

    ScopedTimer timer3("Gaussian Blur");
    Data = GaussianBlur::applyFilterAll(Data);
    timer3.stop();

    ScopedTimer timer4("Sobel");
    Data = Sobel::applyFilter(Data);
    timer4.stop();

    ScopedTimer timer5("File writing");
    BMP::BMPWriter(output, Data);
    timer5.stop();
    full_timer.stop();
}