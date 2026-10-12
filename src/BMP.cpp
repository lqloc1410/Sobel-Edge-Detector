#include "BMP.hpp"
#include "GrayscaleConverter.hpp"
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>

namespace BMP {
    namespace {
        constexpr uint16_t kSignature = 0x4D42; // 'B' 'M' (little-endian)
        constexpr uint32_t kCompressionRGB = 0;
        constexpr uint32_t kCompressionBitfields = 3;

        struct PixelLayout {
            size_t width = 0;
            size_t height = 0;
            size_t bytesPerPixel = 0;
            size_t rowStride = 0; // Bytes per row in the file, including padding to 4 bytes
            bool isTopDown = false;
        };

        bool readError(const std::string &filename, const std::string &reason) {
            std::cerr << "Error reading '" << filename << "': " << reason << "\n";
            return false;
        }

        bool writeError(const std::string &filename, const std::string &reason) {
            std::cerr << "Error writing '" << filename << "': " << reason << "\n";
            return false;
        }

        // Reads and validates both headers. Every value that later drives an allocation
        // or a read is checked against the real file size, so a corrupt or hostile header
        // cannot cause a huge allocation or a read past the end of the file.
        // On success the stream is positioned at the first byte of pixel data.
        bool readHeaders(std::ifstream &file, const std::string &filename, PixelLayout &layout) {
            file.seekg(0, std::ios::end);
            const std::streamoff endPosition = file.tellg();
            file.seekg(0, std::ios::beg);
            if (endPosition < 0 || !file) {
                return readError(filename, "cannot determine file size");
            }
            const uint64_t fileSize = static_cast<uint64_t>(endPosition);

            BMPFileHeader fileHeader{};
            BMPInfoHeader infoHeader{};
            file.read(reinterpret_cast<char*>(&fileHeader), sizeof(BMPFileHeader));
            file.read(reinterpret_cast<char*>(&infoHeader), sizeof(BMPInfoHeader));
            if (!file) {
                return readError(filename, "file is too small to contain BMP headers");
            }

            if (fileHeader.file_signature != kSignature) {
                return readError(filename, "invalid BMP signature (expected 'BM')");
            }
            // 40 = BITMAPINFOHEADER; larger values (V4 = 108, V5 = 124) extend it compatibly.
            // Smaller values (e.g. 12-byte BITMAPCOREHEADER) use a different layout.
            if (infoHeader.size < sizeof(BMPInfoHeader)) {
                return readError(filename, "unsupported info header size " +
                    std::to_string(infoHeader.size) + " (need at least 40)");
            }
            if (infoHeader.planes != 1) {
                return readError(filename, "invalid number of color planes (" +
                    std::to_string(infoHeader.planes) + ")");
            }
            if (infoHeader.bitCount != 24 && infoHeader.bitCount != 32) {
                return readError(filename, "unsupported bit depth " +
                    std::to_string(infoHeader.bitCount) + " (only 24 and 32 are supported)");
            }

            if (infoHeader.compression == kCompressionBitfields && infoHeader.bitCount == 32) {
                // Red, green and blue masks follow the 40-byte header directly, both for a
                // BITMAPINFOHEADER + BI_BITFIELDS file and inside V4/V5 headers.
                uint32_t masks[3] = {0, 0, 0};
                file.read(reinterpret_cast<char*>(masks), sizeof(masks));
                if (!file) {
                    return readError(filename, "truncated BI_BITFIELDS color masks");
                }
                if (masks[0] != 0x00FF0000u || masks[1] != 0x0000FF00u || masks[2] != 0x000000FFu) {
                    return readError(filename,
                        "unsupported BI_BITFIELDS channel masks (only BGRA byte order is supported)");
                }
            } else if (infoHeader.compression != kCompressionRGB) {
                return readError(filename, "unsupported compression type " +
                    std::to_string(infoHeader.compression) + " (only uncompressed BMPs are supported)");
            }

            if (infoHeader.width <= 0) {
                return readError(filename, "invalid width " + std::to_string(infoHeader.width));
            }
            if (infoHeader.height == 0 || infoHeader.height == std::numeric_limits<int32_t>::min()) {
                return readError(filename, "invalid height " + std::to_string(infoHeader.height));
            }

            layout.width = static_cast<size_t>(infoHeader.width);
            layout.isTopDown = infoHeader.height < 0;
            layout.height = static_cast<size_t>(layout.isTopDown
                ? -static_cast<int64_t>(infoHeader.height)
                : static_cast<int64_t>(infoHeader.height));
            layout.bytesPerPixel = infoHeader.bitCount / 8u;
            // width < 2^31, so this cannot overflow a 64-bit size_t.
            layout.rowStride = (layout.width * layout.bytesPerPixel + 3u) & ~static_cast<size_t>(3u);

            const uint64_t minimumOffset = sizeof(BMPFileHeader) + static_cast<uint64_t>(infoHeader.size);
            if (fileHeader.offset < minimumOffset || fileHeader.offset > fileSize) {
                return readError(filename, "pixel data offset " +
                    std::to_string(fileHeader.offset) + " is out of range");
            }

            // Written as a division so that rowStride * height cannot overflow.
            const uint64_t availableBytes = fileSize - fileHeader.offset;
            if (layout.rowStride > availableBytes / layout.height) {
                return readError(filename, "file is truncated: header declares " +
                    std::to_string(layout.width) + "x" + std::to_string(layout.height) +
                    " pixels but only " + std::to_string(availableBytes) +
                    " bytes of pixel data are present");
            }

            file.seekg(static_cast<std::streamoff>(fileHeader.offset), std::ios::beg);
            if (!file) {
                return readError(filename, "cannot seek to pixel data");
            }
            return true;
        }

        // Shared loader: validates the headers, then streams one file row at a time
        // through convertRow(sourceRowBytes, destinationRow, layout).
        template <typename Pixel, typename ConvertRow>
        BasicImage<Pixel> loadImage(const std::string &filename, ConvertRow convertRow) {
            std::ifstream file(filename, std::ios::binary);
            if (!file.is_open()) {
                readError(filename, "cannot open file");
                return {};
            }

            PixelLayout layout;
            if (!readHeaders(file, filename, layout)) {
                return {};
            }

            BasicImage<Pixel> image(layout.height, layout.width);
            std::vector<uint8_t> row(layout.rowStride);
            for (size_t r = 0; r < layout.height; ++r) {
                file.read(reinterpret_cast<char*>(row.data()), static_cast<std::streamsize>(row.size()));
                if (!file) {
                    readError(filename, "unexpected end of pixel data");
                    return {};
                }

                const size_t targetRow = layout.isTopDown ? r : layout.height - 1 - r;
                convertRow(row.data(), &image(targetRow, 0), layout);
            }

            return image;
        }

        // Shared writer: emits a 24-bit bottom-up BMP. fillRow(sourceRow, destinationBytes, width)
        // writes width * 3 BGR bytes; the padding bytes of the row buffer stay zero.
        template <typename Pixel, typename FillRow>
        bool saveImage(const std::string &filename, const BasicImage<Pixel> &image, FillRow fillRow) {
            if (image.empty()) {
                return writeError(filename, "image is empty");
            }

            constexpr size_t maxDimension = static_cast<size_t>(std::numeric_limits<int32_t>::max());
            if (image.width > maxDimension || image.height > maxDimension) {
                return writeError(filename, "image dimensions exceed the BMP limit");
            }

            const uint64_t rowSizeWithoutPadding = static_cast<uint64_t>(image.width) * sizeof(RGBPixel);
            const uint64_t rowStride = (rowSizeWithoutPadding + 3u) & ~static_cast<uint64_t>(3u);
            const uint64_t imageSize = rowStride * image.height;
            const uint64_t pixelOffset = sizeof(BMPFileHeader) + sizeof(BMPInfoHeader);
            if (pixelOffset + imageSize > std::numeric_limits<uint32_t>::max()) {
                return writeError(filename, "image is too large for the BMP format (over 4 GB)");
            }

            BMPInfoHeader infoHeader{};
            infoHeader.size = sizeof(BMPInfoHeader);
            infoHeader.width = static_cast<int32_t>(image.width);
            infoHeader.height = static_cast<int32_t>(image.height); // Positive height = bottom-up storage
            infoHeader.planes = 1;
            infoHeader.bitCount = 24;
            infoHeader.compression = kCompressionRGB;
            infoHeader.sizeImage = static_cast<uint32_t>(imageSize);

            BMPFileHeader fileHeader{};
            fileHeader.file_signature = kSignature;
            fileHeader.reserved1 = 0;
            fileHeader.reserved2 = 0;
            fileHeader.offset = static_cast<uint32_t>(pixelOffset);
            fileHeader.file_size = static_cast<uint32_t>(pixelOffset + imageSize);

            // Opened only after validation, so a rejected image never truncates an existing file.
            std::ofstream file(filename, std::ios::binary);
            if (!file.is_open()) {
                return writeError(filename, "cannot open file for writing");
            }

            file.write(reinterpret_cast<const char*>(&fileHeader), sizeof(BMPFileHeader));
            file.write(reinterpret_cast<const char*>(&infoHeader), sizeof(BMPInfoHeader));

            std::vector<uint8_t> row(static_cast<size_t>(rowStride), 0);
            for (size_t i = 0; i < image.height && file; ++i) {
                const size_t r = image.height - 1 - i;
                fillRow(&image(r, 0), row.data(), image.width);
                file.write(reinterpret_cast<const char*>(row.data()), static_cast<std::streamsize>(row.size()));
            }

            file.close();
            if (!file) {
                return writeError(filename, "write failed (disk full or I/O error?)");
            }
            return true;
        }
    }

    Image BMPParser(const std::string &filename) {
        return loadImage<RGBPixel>(filename,
            [](const uint8_t *source, RGBPixel *destination, const PixelLayout &layout) {
                if (layout.bytesPerPixel == sizeof(RGBPixel)) {
                    std::memcpy(destination, source, layout.width * sizeof(RGBPixel));
                } else {
                    for (size_t c = 0; c < layout.width; ++c) {
                        const uint8_t *pixel = source + c * layout.bytesPerPixel;
                        destination[c] = {pixel[0], pixel[1], pixel[2]};
                    }
                }
            });
    }

    GrayImage BMPParserGray(const std::string &filename) {
        return loadImage<uint8_t>(filename,
            [](const uint8_t *source, uint8_t *destination, const PixelLayout &layout) {
                for (size_t c = 0; c < layout.width; ++c) {
                    const uint8_t *pixel = source + c * layout.bytesPerPixel; // B, G, R[, A]
                    destination[c] = GrayscaleConverter::luminance(pixel[2], pixel[1], pixel[0]);
                }
            });
    }

    bool BMPWriter(const std::string &filename, const Image &pixels) {
        return saveImage(filename, pixels,
            [](const RGBPixel *source, uint8_t *destination, size_t width) {
                std::memcpy(destination, source, width * sizeof(RGBPixel));
            });
    }

    bool BMPWriter(const std::string &filename, const GrayImage &pixels) {
        return saveImage(filename, pixels,
            [](const uint8_t *source, uint8_t *destination, size_t width) {
                for (size_t c = 0; c < width; ++c) {
                    destination[3 * c] = source[c];
                    destination[3 * c + 1] = source[c];
                    destination[3 * c + 2] = source[c];
                }
            });
    }
}