#include "../include/BMP.h"

namespace BMP {

    std::vector<std::vector<RGBPixel>> BMPParser(std::string filename) {
        std::ifstream file(filename, std::ios::binary);

        if (!file.is_open()) {
            std::cerr << "Failed to open file.\n";
            return {};
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

        std::vector<std::vector<RGBPixel>> pixels(height, std::vector<RGBPixel>(width));
        bool isTopDown = InfoHeader.height < 0;

        for (int r = 0; r < height; ++r) {
            int target_r = isTopDown ? r : (height - 1 - r);

            if (InfoHeader.bitCount == 24) {
                file.read(reinterpret_cast<char*>(pixels[target_r].data()), width * sizeof(RGBPixel));
            } else if (InfoHeader.bitCount == 32) {
                for (int c = 0; c < width; ++c) {
                    file.read(reinterpret_cast<char*>(&pixels[target_r][c]), sizeof(RGBPixel));
                    file.seekg(1, std::ios::cur); // Skip alpha byte
                }
            } else {
                std::cerr << "Unsupported bit depth: " << InfoHeader.bitCount << "\n";
                return {};
            }

            if (padding > 0) {
                file.seekg(padding, std::ios::cur);
            }
        }

        return pixels;
    }

    bool BMPWriter(const std::string &filename, const std::vector<std::vector<RGBPixel>> &pixels) {
        if (pixels.empty() || pixels[0].empty()) return false;

        std::ofstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "Failed to open output file.\n";
            return false;
        }

        int height = pixels.size();
        int width = pixels[0].size();

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
            file.write(reinterpret_cast<const char*>(pixels[r].data()), rowSizeWithoutPadding);
            if (padding > 0) {
                file.write(reinterpret_cast<const char*>(paddingBytes), padding);
            }
        }

        return true;
    }
}