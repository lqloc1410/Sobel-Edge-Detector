#pragma once
#include <iostream>
#include <fstream>
#include <string>
#include <cstdint>
#include <vector>
#include <cmath>
#include <cstring>

namespace BMP{
    #pragma pack(push, 1)
    struct BMPFileHeader{
        uint16_t file_signature;
        uint32_t file_size;
        uint16_t reserved1, reserved2;
        uint32_t offset;
    };

    struct BMPInfoHeader {
        uint32_t size{0};            // Size of this header (40 bytes)
        int32_t  width{0};           // Image width in pixels
        int32_t  height{0};          // Image height in pixels (negative = top-down)
        uint16_t planes{1};          // Always 1
        uint16_t bitCount{0};        // Bits per pixel (e.g., 24, 32)
        uint32_t compression{0};     // Compression type (0 = BI_RGB uncompressed)
        uint32_t sizeImage{0};       // Image size in bytes (can be 0 for BI_RGB)
        int32_t  xPixelsPerMeter{0}; // Horizontal resolution
        int32_t  yPixelsPerMeter{0}; // Vertical resolution
        uint32_t colorsUsed{0};      // Number of color palette indices
        uint32_t colorsImportant{0}; // Important colors count
    };

    struct RGBPixel{
        uint8_t blue;
        uint8_t green;
        uint8_t red;
    };

    #pragma pack(pop)

    struct Image{
        size_t height = 0;
        size_t width = 0;
        std::vector<RGBPixel> data;

        Image() = default;
        Image(int h, int w){
            height = h;
            width = w;
            data.resize(h * w);
        }

        RGBPixel& operator()(size_t r, size_t c) {
            return data[r * width + c];
        }

        const RGBPixel& operator()(size_t r, size_t c) const {
            return data[r * width + c];
        }
        bool empty() const{
            return width == 0 || height == 0;
        }
    };

    Image BMPParser(std::string filename);
    bool BMPWriter(const std::string &filename, const Image &pixels);
}
