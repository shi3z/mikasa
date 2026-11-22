// Phase 0: check every tensor in the GGUF and load it to the GPU.
// Build: see engine/Makefile. Run: ./phase0_load <model.gguf> [--no-gpu]
#include "gguf.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <map>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static uint64_t fnv1a(const uint8_t * p, uint64_t n) {
    uint64_t h = 1469598103934665603ull;
    for (uint64_t i = 0; i < n; ++i) { h ^= p[i]; h *= 1099511628211ull; }
    return h;
}

#define CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    std::fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); std::exit(1); } } while (0)

int main(int argc, char ** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: %s model.gguf [--no-gpu]\n", argv[0]); return 2; }
    bool use_gpu = true;
    for (int i = 2; i < argc; ++i) if (std::strcmp(argv[i], "--no-gpu") == 0) use_gpu = false;

    gguf::File f = gguf::open(argv[1]);
    std::printf("file: %s (%.2f GB), tensors: %zu, alignment: %llu, data_start: %llu\n",
        argv[1], f.file_size / 1e9, f.tensors.size(), (unsigned long long) f.alignment,
        (unsigned long long) f.data_start);
    std::printf("arch: %s\n", f.kv_str.count("general.architecture") ? f.kv_str["general.architecture"].c_str() : "?");

    // 1. static checks: type known, size matches, range inside the file, no overlap
    uint64_t total = 0, n_bad = 0;
    std::vector<std::pair<uint64_t, uint64_t>> ranges;
    std::map<uint32_t, uint64_t> bytes_by_type, count_by_type;
    for (const auto & t : f.tensors) {
        auto ti = gguf::type_info(t.type);
        if (ti.bytes == 0) { std::printf("BAD type %u for %s\n", t.type, t.name.c_str()); n_bad++; continue; }
        if (t.n_elements() % ti.blck != 0) { std::printf("BAD block size for %s\n", t.name.c_str()); n_bad++; continue; }
        uint64_t abs = f.data_start + t.offset;
        uint64_t n = t.n_bytes();
        if (abs % 32 != 0) { std::printf("WARN offset not 32-aligned: %s\n", t.name.c_str()); }
        if (abs + n > f.file_size) { std::printf("BAD range for %s\n", t.name.c_str()); n_bad++; continue; }
        ranges.push_back({abs, abs + n});
        total += n;
        bytes_by_type[t.type] += n;
        count_by_type[t.type]++;
    }
    // overlap check after sort
    std::sort(ranges.begin(), ranges.end());
    for (size_t i = 1; i < ranges.size(); ++i) {
        if (ranges[i].first < ranges[i - 1].second) { std::printf("BAD overlap at %llu\n", (unsigned long long) ranges[i].first); n_bad++; }
    }
    std::printf("\nper type:\n");
    for (auto & kv : count_by_type) {
        auto ti = gguf::type_info(kv.first);
        std::printf("  %-6s (id %2u): %4llu tensors, %9.3f MB\n", ti.name, kv.first,
            (unsigned long long) kv.second, bytes_by_type[kv.first] / 1e6);
    }
    std::printf("total tensor bytes: %.3f GB, checks failed: %llu\n", total / 1e9, (unsigned long long) n_bad);

    // 2. upload each tensor and verify the device copy by hashing it back
    if (use_gpu) {
        int dev = 0;
        cudaDeviceProp prop;
        CHECK(cudaGetDevice(&dev));
        CHECK(cudaGetDeviceProperties(&prop, dev));
        std::printf("\ndevice: %s, sm_%d%d, %.1f GB\n", prop.name, prop.major, prop.minor, prop.totalGlobalMem / 1e9);

        uint64_t n_ok = 0, n_mismatch = 0, dev_bytes = 0;
        std::vector<uint8_t> back;
        for (const auto & t : f.tensors) {
            uint64_t n = t.n_bytes();
            const uint8_t * host = f.base + f.data_start + t.offset;
            void * d = nullptr;
            CHECK(cudaMalloc(&d, n));
            CHECK(cudaMemcpy(d, host, n, cudaMemcpyHostToDevice));
            back.resize(n);
            CHECK(cudaMemcpy(back.data(), d, n, cudaMemcpyDeviceToHost));
            if (std::memcmp(back.data(), host, n) == 0) n_ok++; else n_mismatch++;
            dev_bytes += n;
            CHECK(cudaFree(d));
        }
        std::printf("uploaded %llu tensors (%.3f GB), round-trip ok: %llu, mismatched: %llu\n",
            (unsigned long long) f.tensors.size(), dev_bytes / 1e9,
            (unsigned long long) n_ok, (unsigned long long) n_mismatch);
        n_bad += n_mismatch;
    }

    // 3. hash of the full tensor region, to compare with a later re-run or a second tool
    std::printf("fnv1a of tensor region: %016llx\n",
        (unsigned long long) fnv1a(f.base + f.data_start, total));

    gguf::close(f);
    return n_bad == 0 ? 0 : 1;
}
