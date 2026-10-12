#pragma once
#include <string>
#include <cstdint>
#include <cstddef>
#include <vector>

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

    // The raw header reads and the memcpy of pixel rows rely on these exact layouts.
    static_assert(sizeof(BMPFileHeader) == 14, "BMPFileHeader must be 14 bytes");
    static_assert(sizeof(BMPInfoHeader) == 40, "BMPInfoHeader must be 40 bytes");
    static_assert(sizeof(RGBPixel) == 3, "RGBPixel must be 3 bytes");

    // Row-major 2D pixel buffer. Image = 3 bytes/pixel, GrayImage = 1 byte/pixel.
    template <typename Pixel>
    struct BasicImage{
        size_t height = 0;
        size_t width = 0;
        std::vector<Pixel> data;

        BasicImage() = default;
        BasicImage(size_t h, size_t w) : height(h), width(w), data(h * w) {}

        Pixel& operator()(size_t r, size_t c) {
            return data[r * width + c];
        }

        const Pixel& operator()(size_t r, size_t c) const {
            return data[r * width + c];
        }
        bool empty() const{
            return width == 0 || height == 0;
        }
    };

    using Image = BasicImage<RGBPixel>;
    using GrayImage = BasicImage<uint8_t>;

    // Loaders return an empty image on failure (the reason is printed to std::cerr).
    // Supported input: uncompressed 24-bit BI_RGB, or 32-bit BI_RGB / BI_BITFIELDS with BGRA masks.
    Image BMPParser(const std::string &filename);

    // Loads directly into grayscale, converting row by row, so the full RGB image never exists in memory.
    GrayImage BMPParserGray(const std::string &filename);

    // Both writers produce a 24-bit uncompressed BMP. They return false on any failure, including I/O errors.
    bool BMPWriter(const std::string &filename, const Image &pixels);
    bool BMPWriter(const std::string &filename, const GrayImage &pixels);
}
