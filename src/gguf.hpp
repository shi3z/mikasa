// GGUF v3 reader: header, metadata and tensor table. Data is read with mmap.
#pragma once

#include <cstdint>
#include <cstring>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace gguf {

enum ggml_type : uint32_t {
    F32 = 0, F16 = 1, Q8_0 = 8, BF16 = 30, NVFP4 = 40,
};

// Block layout: elements per block, bytes per block.
struct TypeInfo { const char * name; uint64_t blck; uint64_t bytes; };

inline TypeInfo type_info(uint32_t t) {
    switch (t) {
        case F32:   return {"F32", 1, 4};
        case F16:   return {"F16", 1, 2};
        case BF16:  return {"BF16", 1, 2};
        case Q8_0:  return {"Q8_0", 32, 34};    // half d + 32 x int8
        case NVFP4: return {"NVFP4", 64, 36};   // 4 x UE4M3 scale + 32 bytes E2M1
        default:    return {"?", 0, 0};
    }
}

struct Tensor {
    std::string name;
    std::vector<uint64_t> ne;   // ggml order: ne[0] is the fastest dimension
    uint32_t type = 0;
    uint64_t offset = 0;        // relative to data_start
    uint64_t n_elements() const { uint64_t n = 1; for (auto d : ne) n *= d; return n; }
    uint64_t n_bytes() const { auto ti = type_info(type); return n_elements() / ti.blck * ti.bytes; }
};

struct File {
    uint32_t version = 0;
    uint64_t alignment = 32;
    uint64_t data_start = 0;    // absolute file offset of tensor data
    uint64_t file_size = 0;
    std::map<std::string, std::string> kv_str;
    std::map<std::string, uint64_t> kv_u64;
    std::map<std::string, double> kv_f64;
    std::map<std::string, std::vector<int64_t>> kv_arr_i;
    std::vector<Tensor> tensors;
    const uint8_t * base = nullptr;  // mmap base
    std::string path;
};

File open(const std::string & path);  // parses header, maps the file read-only
void close(File & f);

}  // namespace gguf
