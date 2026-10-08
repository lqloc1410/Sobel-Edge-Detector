#include "../include/BMP.h"

namespace BMP {

    Image BMPParser(std::string filename) {
        std::ifstream file(filename, std::ios::binary);

        if (!file.is_open()) {
            std::cerr << "Failed to open file.\n";
            return Image{};
        }

        BMPFileHeader FileHeader{};
        file.read(reinterpret_cast<char*>(&FileHeader), sizeof(BMPFileHeader));

        BMPInfoHeader InfoHeader{};
        file.read(reinterpret_cast<char*>(&InfoHeader), sizeof(BMPInfoHeader));

        if (!file) {
            std::cerr << "Failed to read file headers.\n";
            return {};
        }

        // Verify magic number ('B' 'M')
        if (FileHeader.file_signature != 0x4D42) {
            std::cerr << "Invalid BMP signature.\n";
            return {};
        }

        file.seekg(FileHeader.offset, std::ios::beg);

        int width = std::abs(InfoHeader.width);
        int height = std::abs(InfoHeader.height);
        int bytesPerPixel = InfoHeader.bitCount / 8;

        int rowSizeWithoutPadding = width * bytesPerPixel;
        int rowStride = (rowSizeWithoutPadding + 3) & ~3;
        int padding = rowStride - rowSizeWithoutPadding;

        Image pixels(height, width);
        bool isTopDown = InfoHeader.height < 0;

        for (int r = 0; r < height; ++r) {
            int target_r = isTopDown ? r : (height - 1 - r);

            if (InfoHeader.bitCount != 24 && InfoHeader.bitCount != 32){
                std::cerr << "Unsupported bit depth: " << InfoHeader.bitCount << "\n";
                return Image{};
            }
            for (int c = 0; c < width; ++c) {
                file.read(reinterpret_cast<char*>(&pixels(target_r, c)), sizeof(RGBPixel));
                if (InfoHeader.bitCount == 32){
                    file.seekg(1, std::ios::cur);
                }
            }
            if (padding > 0) {
                file.seekg(padding, std::ios::cur);
            }
        }

        return pixels;
    }

    bool BMPWriter(const std::string &filename, const Image &pixels) {
        if (pixels.empty()) return false;

        std::ofstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "Failed to open output file.\n";
            return false;
        }

        int height = pixels.height;
        int width = pixels.width;

        int rowSizeWithoutPadding = width * static_cast<int>(sizeof(RGBPixel));
        int rowStride = (rowSizeWithoutPadding + 3) & ~3;
        int padding = rowStride - rowSizeWithoutPadding;

        BMPInfoHeader infoHeader{};
        infoHeader.size = sizeof(BMPInfoHeader);
        infoHeader.width = width;
        infoHeader.height = height; // Positive height = bottom-up storage
        infoHeader.planes = 1;
        infoHeader.bitCount = 24;
        infoHeader.compression = 0; // BI_RGB (uncompressed)
        infoHeader.sizeImage = rowStride * height;

        BMPFileHeader fileHeader{};
        fileHeader.file_signature = 0x4D42; // 'BM'
        fileHeader.reserved1 = 0;
        fileHeader.reserved2 = 0;
        fileHeader.offset = sizeof(BMPFileHeader) + sizeof(BMPInfoHeader);
        fileHeader.file_size = fileHeader.offset + infoHeader.sizeImage;

        file.write(reinterpret_cast<const char*>(&fileHeader), sizeof(BMPFileHeader));
        file.write(reinterpret_cast<const char*>(&infoHeader), sizeof(BMPInfoHeader));

        uint8_t paddingBytes[3] = {0, 0, 0};
        for (int r = height - 1; r >= 0; --r) {
            for (int c = 0; c < width; ++c){
                file.write(reinterpret_cast<const char*>(&pixels(r, c)), sizeof(RGBPixel));
            }
            //file.write(reinterpret_cast<const char*>(pixels[r].data()), rowSizeWithoutPadding);
            if (padding > 0) {
                file.write(reinterpret_cast<const char*>(paddingBytes), padding);
            }
        }

        return true;
    }
}