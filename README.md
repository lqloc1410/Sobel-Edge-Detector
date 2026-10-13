# Sobel Edge Detector

A high-performance image processing program that implements an edge detection pipeline written from scratch in modern C++23.

## Features

- **24/32-bit BMP Support**: Fast parser and writer with header validation, bottom-up and top-down orientation handling, and row stride padding support.
- **GPU Acceleration**: Cross-platform hardware acceleration using OpenCL (Intel Arc / Iris, AMD Radeon, NVIDIA GeForce, etc.) keeping data resident in GPU VRAM between Gaussian Blur and Sobel passes.
- **CPU Multi-threading**: Highly parallel CPU fallback using OpenMP.
- **Low Memory Footprint**: Single-channel grayscale pipeline with stream conversion and zero unnecessary intermediate buffers.

## Tech Stack & Prerequisites

- **Language**: C++23
- **Build System**: CMake 3.20+ / Ninja
- **Compiler**: GCC / Clang / MSVC
- **Dependencies**:
  - OpenMP (CPU multi-threading)
  - OpenCL (Cross-platform GPU acceleration; optional, auto-detected)

## Usage

```bash
# Run with automatic hardware detection (uses GPU if available, else CPU)
SobelEdgeDetector input.bmp output.bmp

# Force GPU mode
SobelEdgeDetector input.bmp output.bmp --gpu

# Force CPU mode
SobelEdgeDetector input.bmp output.bmp --cpu

# Interactive prompt mode
SobelEdgeDetector
```
