#include "GPUAccelerator.hpp"
#include <iostream>
#include <vector>
#include <memory>
#include <mutex>
#include <algorithm>
#include <limits>

#if defined(HAVE_OPENCL)
#ifndef CL_TARGET_OPENCL_VERSION
#define CL_TARGET_OPENCL_VERSION 120
#endif
#include <CL/cl.h>

namespace GPUAccelerator {
    namespace {
        const char* const kKernelSource = R"CLC(
__constant int gaussian_weights[9] = {1, 2, 1, 2, 4, 2, 1, 2, 1};
__constant int sobel_kx[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
__constant int sobel_ky[9] = { 1, 2, 1,  0, 0, 0, -1, -2, -1};

__kernel void gaussian_blur(
    __global const uchar* src,
    __global uchar* dst,
    int width,
    int height
) {
    int c = get_global_id(0);
    int r = get_global_id(1);
    if (c >= width || r >= height) return;

    if (r >= 1 && r < height - 1 && c >= 1 && c < width - 1) {
        int sum = (int)src[(r - 1) * width + (c - 1)] * 1 +
                  (int)src[(r - 1) * width + c]       * 2 +
                  (int)src[(r - 1) * width + (c + 1)] * 1 +
                  (int)src[r * width + (c - 1)]       * 2 +
                  (int)src[r * width + c]             * 4 +
                  (int)src[r * width + (c + 1)]       * 2 +
                  (int)src[(r + 1) * width + (c - 1)] * 1 +
                  (int)src[(r + 1) * width + c]       * 2 +
                  (int)src[(r + 1) * width + (c + 1)] * 1;
        dst[r * width + c] = (uchar)(sum >> 4);
    } else {
        int sum = 0;
        int sumKernel = 0;
        int k = 0;
        for (int dr = -1; dr <= 1; ++dr) {
            int nr = r + dr;
            bool row_valid = (nr >= 0 && nr < height);
            for (int dc = -1; dc <= 1; ++dc, ++k) {
                int nc = c + dc;
                if (row_valid && nc >= 0 && nc < width) {
                    int w = gaussian_weights[k];
                    sumKernel += w;
                    sum += (int)src[nr * width + nc] * w;
                }
            }
        }
        dst[r * width + c] = (uchar)(sum / sumKernel);
    }
}

short calc_magnitude(__global const uchar* src, int c, int r, int width, int height) {
    int gx, gy;
    if (r >= 1 && r < height - 1 && c >= 1 && c < width - 1) {
        int p00 = src[(r - 1) * width + (c - 1)];
        int p02 = src[(r - 1) * width + (c + 1)];
        int p10 = src[r * width + (c - 1)];
        int p12 = src[r * width + (c + 1)];
        int p20 = src[(r + 1) * width + (c - 1)];
        int p22 = src[(r + 1) * width + (c + 1)];

        gx = -p00 + p02 - 2 * p10 + 2 * p12 - p20 + p22;

        int p01 = src[(r - 1) * width + c];
        int p21 = src[(r + 1) * width + c];
        gy =  p00 + 2 * p01 + p02 - p20 - 2 * p21 - p22;
    } else {
        gx = 0;
        gy = 0;
        int k = 0;
        for (int dr = -1; dr <= 1; ++dr) {
            int nr = clamp(r + dr, 0, height - 1);
            for (int dc = -1; dc <= 1; ++dc, ++k) {
                int nc = clamp(c + dc, 0, width - 1);
                int val = src[nr * width + nc];
                gx += val * sobel_kx[k];
                gy += val * sobel_ky[k];
            }
        }
    }
    return (short)sqrt((float)(gx * gx + gy * gy));
}

__kernel void sobel_find_minmax(
    __global const uchar* src,
    int width,
    int height,
    __local short* s_min,
    __local short* s_max,
    __global short* out_min,
    __global short* out_max
) {
    int c = get_global_id(0);
    int r = get_global_id(1);
    int lid = get_local_id(1) * get_local_size(0) + get_local_id(0);
    int lsize = get_local_size(0) * get_local_size(1);

    short val = 0;
    if (c < width && r < height) {
        val = calc_magnitude(src, c, r, width, height);
        s_min[lid] = val;
        s_max[lid] = val;
    } else {
        s_min[lid] = 32767;
        s_max[lid] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int stride = lsize / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            if (s_min[lid + stride] < s_min[lid]) s_min[lid] = s_min[lid + stride];
            if (s_max[lid + stride] > s_max[lid]) s_max[lid] = s_max[lid + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        int gid = get_group_id(1) * get_num_groups(0) + get_group_id(0);
        out_min[gid] = s_min[0];
        out_max[gid] = s_max[0];
    }
}

__kernel void sobel_normalize(
    __global const uchar* src,
    __global uchar* dst,
    int width,
    int height,
    short min_val,
    int range
) {
    int c = get_global_id(0);
    int r = get_global_id(1);
    if (c >= width || r >= height) return;

    short mag = calc_magnitude(src, c, r, width, height);
    dst[r * width + c] = (uchar)((mag - min_val) * 255 / range);
}
)CLC";

        struct OpenCLState {
            cl_platform_id platform = nullptr;
            cl_device_id device = nullptr;
            cl_context context = nullptr;
            cl_command_queue queue = nullptr;
            cl_program program = nullptr;
            cl_kernel k_blur = nullptr;
            cl_kernel k_minmax = nullptr;
            cl_kernel k_norm = nullptr;
            cl_mem d_reduction_min = nullptr;
            cl_mem d_reduction_max = nullptr;
            size_t reductionCapacity = 0;
            std::string deviceName = "None";
            bool valid = false;

            ~OpenCLState() {
                cleanup();
            }

            void cleanup() {
                if (d_reduction_min) { clReleaseMemObject(d_reduction_min); d_reduction_min = nullptr; }
                if (d_reduction_max) { clReleaseMemObject(d_reduction_max); d_reduction_max = nullptr; }
                reductionCapacity = 0;
                if (k_norm) { clReleaseKernel(k_norm); k_norm = nullptr; }
                if (k_minmax) { clReleaseKernel(k_minmax); k_minmax = nullptr; }
                if (k_blur) { clReleaseKernel(k_blur); k_blur = nullptr; }
                if (program) { clReleaseProgram(program); program = nullptr; }
                if (queue) { clReleaseCommandQueue(queue); queue = nullptr; }
                if (context) { clReleaseContext(context); context = nullptr; }
                valid = false;
            }

            bool ensureReductionBuffers(size_t totalGroups) {
                if (totalGroups <= reductionCapacity && d_reduction_min && d_reduction_max) {
                    return true;
                }
                if (d_reduction_min) clReleaseMemObject(d_reduction_min);
                if (d_reduction_max) clReleaseMemObject(d_reduction_max);

                cl_int err1, err2;
                d_reduction_min = clCreateBuffer(context, CL_MEM_WRITE_ONLY, totalGroups * sizeof(short), nullptr, &err1);
                d_reduction_max = clCreateBuffer(context, CL_MEM_WRITE_ONLY, totalGroups * sizeof(short), nullptr, &err2);
                if (err1 != CL_SUCCESS || err2 != CL_SUCCESS || !d_reduction_min || !d_reduction_max) {
                    reductionCapacity = 0;
                    return false;
                }
                reductionCapacity = totalGroups;
                return true;
            }

            bool init() {
                if (valid) return true;

                cl_uint numPlatforms = 0;
                cl_int err = clGetPlatformIDs(0, nullptr, &numPlatforms);
                if (err != CL_SUCCESS || numPlatforms == 0) return false;

                std::vector<cl_platform_id> platforms(numPlatforms);
                clGetPlatformIDs(numPlatforms, platforms.data(), nullptr);

                cl_device_id selectedDevice = nullptr;
                cl_platform_id selectedPlatform = nullptr;

                for (cl_uint i = 0; i < numPlatforms; ++i) {
                    cl_uint numDevices = 0;
                    if (clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, 0, nullptr, &numDevices) == CL_SUCCESS && numDevices > 0) {
                        std::vector<cl_device_id> devices(numDevices);
                        clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, numDevices, devices.data(), nullptr);
                        selectedPlatform = platforms[i];
                        selectedDevice = devices[0];
                        break;
                    }
                }

                if (!selectedDevice) return false;

                platform = selectedPlatform;
                device = selectedDevice;

                char nameBuf[256] = {};
                clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(nameBuf), nameBuf, nullptr);
                deviceName = nameBuf;

                context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
                if (err != CL_SUCCESS || !context) {
                    cleanup();
                    return false;
                }

#if defined(CL_VERSION_2_0)
                queue = clCreateCommandQueueWithProperties(context, device, nullptr, &err);
#else
                queue = clCreateCommandQueue(context, device, 0, &err);
#endif
                if (err != CL_SUCCESS || !queue) {
                    cleanup();
                    return false;
                }

                const char* srcPtr = kKernelSource;
                program = clCreateProgramWithSource(context, 1, &srcPtr, nullptr, &err);
                if (err != CL_SUCCESS || !program) {
                    cleanup();
                    return false;
                }

                err = clBuildProgram(program, 1, &device, "", nullptr, nullptr);
                if (err != CL_SUCCESS) {
                    size_t logSize = 0;
                    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);
                    std::vector<char> buildLog(logSize + 1);
                    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logSize, buildLog.data(), nullptr);
                    std::cerr << "OpenCL Build Log:\n" << buildLog.data() << "\n";
                    cleanup();
                    return false;
                }

                k_blur = clCreateKernel(program, "gaussian_blur", &err);
                k_minmax = clCreateKernel(program, "sobel_find_minmax", &err);
                k_norm = clCreateKernel(program, "sobel_normalize", &err);

                if (err != CL_SUCCESS || !k_blur || !k_minmax || !k_norm) {
                    cleanup();
                    return false;
                }

                valid = true;
                return true;
            }
        };

        OpenCLState& getState() {
            static OpenCLState state;
            static std::once_flag initFlag;
            std::call_once(initFlag, []() {
                state.init();
            });
            return state;
        }

        void get2DGrid(size_t width, size_t height, size_t globalSize[2], size_t localSize[2]) {
            localSize[0] = 16;
            localSize[1] = 16;
            globalSize[0] = ((width + localSize[0] - 1) / localSize[0]) * localSize[0];
            globalSize[1] = ((height + localSize[1] - 1) / localSize[1]) * localSize[1];
        }

        bool computeSobelMinMax(OpenCLState &state, cl_mem d_source, int width, int height,
                                size_t globalSize[2], size_t localSize[2],
                                short &out_min, int &out_range) {
            size_t numGroupsX = globalSize[0] / localSize[0];
            size_t numGroupsY = globalSize[1] / localSize[1];
            size_t totalGroups = numGroupsX * numGroupsY;

            if (!state.ensureReductionBuffers(totalGroups)) {
                return false;
            }

            clSetKernelArg(state.k_minmax, 0, sizeof(cl_mem), &d_source);
            clSetKernelArg(state.k_minmax, 1, sizeof(int), &width);
            clSetKernelArg(state.k_minmax, 2, sizeof(int), &height);
            clSetKernelArg(state.k_minmax, 3, 256 * sizeof(short), nullptr);
            clSetKernelArg(state.k_minmax, 4, 256 * sizeof(short), nullptr);
            clSetKernelArg(state.k_minmax, 5, sizeof(cl_mem), &state.d_reduction_min);
            clSetKernelArg(state.k_minmax, 6, sizeof(cl_mem), &state.d_reduction_max);

            cl_int err = clEnqueueNDRangeKernel(state.queue, state.k_minmax, 2, nullptr, globalSize, localSize, 0, nullptr, nullptr);
            if (err != CL_SUCCESS) return false;

            std::vector<short> group_mins(totalGroups);
            std::vector<short> group_maxs(totalGroups);
            clEnqueueReadBuffer(state.queue, state.d_reduction_min, CL_FALSE, 0, totalGroups * sizeof(short), group_mins.data(), 0, nullptr, nullptr);
            clEnqueueReadBuffer(state.queue, state.d_reduction_max, CL_TRUE, 0, totalGroups * sizeof(short), group_maxs.data(), 0, nullptr, nullptr);

            short min_val = std::numeric_limits<short>::max();
            short max_val = 0;
            #pragma omp parallel for reduction(min:min_val) reduction(max:max_val) if(totalGroups > 10000)
            for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(totalGroups); ++i) {
                if (group_mins[static_cast<size_t>(i)] < min_val) min_val = group_mins[static_cast<size_t>(i)];
                if (group_maxs[static_cast<size_t>(i)] > max_val) max_val = group_maxs[static_cast<size_t>(i)];
            }

            out_min = min_val;
            out_range = (max_val - min_val > 0) ? (max_val - min_val) : 1;
            return true;
        }
    }

    bool isAvailable() {
        return getState().valid;
    }

    std::string getDeviceName() {
        return getState().deviceName;
    }

    bool runPipeline(const BMP::GrayImage &input, BMP::GrayImage &output) {
        if (input.empty()) return false;
        auto &state = getState();
        if (!state.valid) return false;

        const int width = static_cast<int>(input.width);
        const int height = static_cast<int>(input.height);
        const size_t imgBytes = static_cast<size_t>(width) * static_cast<size_t>(height);

        // Pre-allocate output buffer on host
        output = BMP::GrayImage(input.height, input.width);

        // Utilize CL_MEM_USE_HOST_PTR for zero-copy UMA architecture on integrated/unified GPUs (Intel Arc 140T),
        // falling back to standard buffers if unmapped/unsupported.
        cl_int err = CL_SUCCESS;
        cl_mem d_input = clCreateBuffer(state.context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                                        imgBytes, const_cast<uint8_t*>(input.data.data()), &err);
        bool useHostPtr = (err == CL_SUCCESS && d_input != nullptr);
        if (!useHostPtr) {
            d_input = clCreateBuffer(state.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                     imgBytes, const_cast<uint8_t*>(input.data.data()), &err);
        }

        cl_mem d_blur = clCreateBuffer(state.context, CL_MEM_READ_WRITE, imgBytes, nullptr, &err);

        cl_mem d_output = nullptr;
        if (useHostPtr) {
            d_output = clCreateBuffer(state.context, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR,
                                      imgBytes, output.data.data(), &err);
            if (err != CL_SUCCESS || !d_output) {
                d_output = clCreateBuffer(state.context, CL_MEM_WRITE_ONLY, imgBytes, nullptr, &err);
                useHostPtr = false;
            }
        } else {
            d_output = clCreateBuffer(state.context, CL_MEM_WRITE_ONLY, imgBytes, nullptr, &err);
        }

        if (!d_input || !d_blur || !d_output) {
            std::cerr << "GPU: Failed to allocate GPU memory buffers (" << (imgBytes / (1024 * 1024)) << " MB)\n";
            if (d_input) clReleaseMemObject(d_input);
            if (d_blur) clReleaseMemObject(d_blur);
            if (d_output) clReleaseMemObject(d_output);
            return false;
        }

        size_t globalSize[2], localSize[2];
        get2DGrid(static_cast<size_t>(width), static_cast<size_t>(height), globalSize, localSize);

        // 1. Gaussian Blur (d_input -> d_blur)
        clSetKernelArg(state.k_blur, 0, sizeof(cl_mem), &d_input);
        clSetKernelArg(state.k_blur, 1, sizeof(cl_mem), &d_blur);
        clSetKernelArg(state.k_blur, 2, sizeof(int), &width);
        clSetKernelArg(state.k_blur, 3, sizeof(int), &height);

        err = clEnqueueNDRangeKernel(state.queue, state.k_blur, 2, nullptr, globalSize, localSize, 0, nullptr, nullptr);
        if (err != CL_SUCCESS) {
            std::cerr << "GPU: Gaussian Blur kernel launch failed (" << err << ")\n";
            clReleaseMemObject(d_input);
            clReleaseMemObject(d_blur);
            clReleaseMemObject(d_output);
            return false;
        }

        // Release d_input as soon as blur has finished queueing
        clReleaseMemObject(d_input);

        // 2. Sobel Min/Max Reduction (reads d_blur)
        short min_val = 0;
        int range = 1;
        if (!computeSobelMinMax(state, d_blur, width, height, globalSize, localSize, min_val, range)) {
            std::cerr << "GPU: Sobel min-max reduction failed\n";
            clReleaseMemObject(d_blur);
            clReleaseMemObject(d_output);
            return false;
        }

        // 3. Sobel Normalization (reads d_blur, writes d_output)
        clSetKernelArg(state.k_norm, 0, sizeof(cl_mem), &d_blur);
        clSetKernelArg(state.k_norm, 1, sizeof(cl_mem), &d_output);
        clSetKernelArg(state.k_norm, 2, sizeof(int), &width);
        clSetKernelArg(state.k_norm, 3, sizeof(int), &height);
        clSetKernelArg(state.k_norm, 4, sizeof(short), &min_val);
        clSetKernelArg(state.k_norm, 5, sizeof(int), &range);

        err = clEnqueueNDRangeKernel(state.queue, state.k_norm, 2, nullptr, globalSize, localSize, 0, nullptr, nullptr);
        if (err != CL_SUCCESS) {
            std::cerr << "GPU: Sobel Normalize kernel launch failed (" << err << ")\n";
            clReleaseMemObject(d_blur);
            clReleaseMemObject(d_output);
            return false;
        }

        // 4. Synchronization / Readback
        if (useHostPtr) {
            // Direct zero-copy: synchronize GPU writes into host cache
            void* mapped = clEnqueueMapBuffer(state.queue, d_output, CL_TRUE, CL_MAP_READ, 0, imgBytes, 0, nullptr, nullptr, &err);
            if (mapped) {
                clEnqueueUnmapMemObject(state.queue, d_output, mapped, 0, nullptr, nullptr);
            }
            clFinish(state.queue);
        } else {
            err = clEnqueueReadBuffer(state.queue, d_output, CL_TRUE, 0, imgBytes, output.data.data(), 0, nullptr, nullptr);
        }

        clReleaseMemObject(d_blur);
        clReleaseMemObject(d_output);

        if (err != CL_SUCCESS) {
            std::cerr << "GPU: Output synchronization failed (" << err << ")\n";
            return false;
        }

        return true;
    }

    bool applyGaussianBlur(const BMP::GrayImage &input, BMP::GrayImage &output) {
        if (input.empty()) return false;
        auto &state = getState();
        if (!state.valid) return false;

        const int width = static_cast<int>(input.width);
        const int height = static_cast<int>(input.height);
        const size_t imgBytes = static_cast<size_t>(width) * static_cast<size_t>(height);

        output = BMP::GrayImage(input.height, input.width);

        cl_int err = CL_SUCCESS;
        cl_mem d_input = clCreateBuffer(state.context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                                        imgBytes, const_cast<uint8_t*>(input.data.data()), &err);
        bool useHostPtr = (err == CL_SUCCESS && d_input != nullptr);
        if (!useHostPtr) {
            d_input = clCreateBuffer(state.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                     imgBytes, const_cast<uint8_t*>(input.data.data()), &err);
        }

        cl_mem d_dst = nullptr;
        if (useHostPtr) {
            d_dst = clCreateBuffer(state.context, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR,
                                   imgBytes, output.data.data(), &err);
            if (err != CL_SUCCESS || !d_dst) {
                d_dst = clCreateBuffer(state.context, CL_MEM_WRITE_ONLY, imgBytes, nullptr, &err);
                useHostPtr = false;
            }
        } else {
            d_dst = clCreateBuffer(state.context, CL_MEM_WRITE_ONLY, imgBytes, nullptr, &err);
        }

        if (!d_input || !d_dst) {
            if (d_input) clReleaseMemObject(d_input);
            if (d_dst) clReleaseMemObject(d_dst);
            return false;
        }

        size_t globalSize[2], localSize[2];
        get2DGrid(static_cast<size_t>(width), static_cast<size_t>(height), globalSize, localSize);

        clSetKernelArg(state.k_blur, 0, sizeof(cl_mem), &d_input);
        clSetKernelArg(state.k_blur, 1, sizeof(cl_mem), &d_dst);
        clSetKernelArg(state.k_blur, 2, sizeof(int), &width);
        clSetKernelArg(state.k_blur, 3, sizeof(int), &height);

        err = clEnqueueNDRangeKernel(state.queue, state.k_blur, 2, nullptr, globalSize, localSize, 0, nullptr, nullptr);
        if (err == CL_SUCCESS) {
            if (useHostPtr) {
                void* mapped = clEnqueueMapBuffer(state.queue, d_dst, CL_TRUE, CL_MAP_READ, 0, imgBytes, 0, nullptr, nullptr, &err);
                if (mapped) clEnqueueUnmapMemObject(state.queue, d_dst, mapped, 0, nullptr, nullptr);
                clFinish(state.queue);
            } else {
                err = clEnqueueReadBuffer(state.queue, d_dst, CL_TRUE, 0, imgBytes, output.data.data(), 0, nullptr, nullptr);
            }
        }

        clReleaseMemObject(d_input);
        clReleaseMemObject(d_dst);
        return (err == CL_SUCCESS);
    }

    bool applySobel(const BMP::GrayImage &input, BMP::GrayImage &output) {
        if (input.empty()) return false;
        auto &state = getState();
        if (!state.valid) return false;

        const int width = static_cast<int>(input.width);
        const int height = static_cast<int>(input.height);
        const size_t imgBytes = static_cast<size_t>(width) * static_cast<size_t>(height);

        output = BMP::GrayImage(input.height, input.width);

        cl_int err = CL_SUCCESS;
        cl_mem d_src = clCreateBuffer(state.context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                                      imgBytes, const_cast<uint8_t*>(input.data.data()), &err);
        bool useHostPtr = (err == CL_SUCCESS && d_src != nullptr);
        if (!useHostPtr) {
            d_src = clCreateBuffer(state.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   imgBytes, const_cast<uint8_t*>(input.data.data()), &err);
        }

        cl_mem d_dst = nullptr;
        if (useHostPtr) {
            d_dst = clCreateBuffer(state.context, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR,
                                   imgBytes, output.data.data(), &err);
            if (err != CL_SUCCESS || !d_dst) {
                d_dst = clCreateBuffer(state.context, CL_MEM_WRITE_ONLY, imgBytes, nullptr, &err);
                useHostPtr = false;
            }
        } else {
            d_dst = clCreateBuffer(state.context, CL_MEM_WRITE_ONLY, imgBytes, nullptr, &err);
        }

        if (!d_src || !d_dst) {
            if (d_src) clReleaseMemObject(d_src);
            if (d_dst) clReleaseMemObject(d_dst);
            return false;
        }

        size_t globalSize[2], localSize[2];
        get2DGrid(static_cast<size_t>(width), static_cast<size_t>(height), globalSize, localSize);

        short min_val = 0;
        int range = 1;
        if (!computeSobelMinMax(state, d_src, width, height, globalSize, localSize, min_val, range)) {
            clReleaseMemObject(d_src);
            clReleaseMemObject(d_dst);
            return false;
        }

        clSetKernelArg(state.k_norm, 0, sizeof(cl_mem), &d_src);
        clSetKernelArg(state.k_norm, 1, sizeof(cl_mem), &d_dst);
        clSetKernelArg(state.k_norm, 2, sizeof(int), &width);
        clSetKernelArg(state.k_norm, 3, sizeof(int), &height);
        clSetKernelArg(state.k_norm, 4, sizeof(short), &min_val);
        clSetKernelArg(state.k_norm, 5, sizeof(int), &range);

        err = clEnqueueNDRangeKernel(state.queue, state.k_norm, 2, nullptr, globalSize, localSize, 0, nullptr, nullptr);
        if (err == CL_SUCCESS) {
            if (useHostPtr) {
                void* mapped = clEnqueueMapBuffer(state.queue, d_dst, CL_TRUE, CL_MAP_READ, 0, imgBytes, 0, nullptr, nullptr, &err);
                if (mapped) clEnqueueUnmapMemObject(state.queue, d_dst, mapped, 0, nullptr, nullptr);
                clFinish(state.queue);
            } else {
                err = clEnqueueReadBuffer(state.queue, d_dst, CL_TRUE, 0, imgBytes, output.data.data(), 0, nullptr, nullptr);
            }
        }

        clReleaseMemObject(d_src);
        clReleaseMemObject(d_dst);
        return (err == CL_SUCCESS);
    }
}

#else

namespace GPUAccelerator {
    bool isAvailable() {
        return false;
    }
    std::string getDeviceName() {
        return "None (OpenCL not available at build time)";
    }
    bool runPipeline(const BMP::GrayImage&, BMP::GrayImage&) {
        return false;
    }
    bool applyGaussianBlur(const BMP::GrayImage&, BMP::GrayImage&) {
        return false;
    }
    bool applySobel(const BMP::GrayImage&, BMP::GrayImage&) {
        return false;
    }
}

#endif
