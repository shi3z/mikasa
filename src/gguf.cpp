// GGUF v3 parser. Only the parts the engine needs.
#include "gguf.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace gguf {
namespace {

struct Cursor {
    const uint8_t * p;
    const uint8_t * end;

    template <typename T> T get() {
        if (p + sizeof(T) > end) throw std::runtime_error("gguf: truncated header");
        T v;
        std::memcpy(&v, p, sizeof(T));
        p += sizeof(T);
        return v;
    }
    std::string str() {
        uint64_t n = get<uint64_t>();
        if (p + n > end) throw std::runtime_error("gguf: truncated string");
        std::string s(reinterpret_cast<const char *>(p), n);
        p += n;
        return s;
    }
};

enum vtype : uint32_t {
    U8, I8, U16, I16, U32, I32, F32v, BOOL, STR, ARR, U64, I64, F64,
};

void read_value(Cursor & c, const std::string & key, uint32_t t, File & f) {
    switch (t) {
        case U8:  f.kv_u64[key] = c.get<uint8_t>(); break;
        case I8:  c.get<int8_t>(); break;
        case U16: f.kv_u64[key] = c.get<uint16_t>(); break;
        case I16: c.get<int16_t>(); break;
        case U32: f.kv_u64[key] = c.get<uint32_t>(); break;
        case I32: c.get<int32_t>(); break;
        case F32v: f.kv_f64[key] = c.get<float>(); break;
        case BOOL: c.get<uint8_t>(); break;
        case STR: f.kv_str[key] = c.str(); break;
        case U64: f.kv_u64[key] = c.get<uint64_t>(); break;
        case I64: c.get<int64_t>(); break;
        case F64: f.kv_f64[key] = c.get<double>(); break;
        case ARR: {
            uint32_t et = c.get<uint32_t>();
            uint64_t n = c.get<uint64_t>();
            if (et == STR) {
                for (uint64_t i = 0; i < n; ++i) c.str();
            } else {
                // only integer and bool arrays are kept (layer patterns and similar)
                size_t sz = 0;
                switch (et) {
                    case U8: case I8: case BOOL: sz = 1; break;
                    case U16: case I16: sz = 2; break;
                    case U32: case I32: case F32v: sz = 4; break;
                    case U64: case I64: case F64: sz = 8; break;
                    default: throw std::runtime_error("gguf: bad array type");
                }
                std::vector<int64_t> v;
                v.reserve(n);
                for (uint64_t i = 0; i < n; ++i) {
                    if (c.p + sz > c.end) throw std::runtime_error("gguf: truncated array");
                    if (et == BOOL || et == U8) v.push_back(c.get<uint8_t>());
                    else if (et == I8) v.push_back(c.get<int8_t>());
                    else if (et == U16) v.push_back(c.get<uint16_t>());
                    else if (et == I16) v.push_back(c.get<int16_t>());
                    else if (et == U32) v.push_back(c.get<uint32_t>());
                    else if (et == I32) v.push_back(c.get<int32_t>());
                    else if (et == U64) v.push_back((int64_t) c.get<uint64_t>());
                    else if (et == I64) v.push_back(c.get<int64_t>());
                    else { c.p += sz; }  // floats: skipped
                }
                if (!v.empty()) f.kv_arr_i[key] = std::move(v);
            }
            break;
        }
        default: throw std::runtime_error("gguf: unknown metadata type");
    }
}

}  // namespace

File open(const std::string & path) {
    File f;
    f.path = path;
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) throw std::runtime_error("gguf: cannot open " + path);
    struct stat st;
    if (fstat(fd, &st) != 0) { ::close(fd); throw std::runtime_error("gguf: stat failed"); }
    f.file_size = (uint64_t) st.st_size;
    void * m = mmap(nullptr, f.file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    ::close(fd);
    if (m == MAP_FAILED) throw std::runtime_error("gguf: mmap failed");
    f.base = static_cast<const uint8_t *>(m);

    Cursor c{f.base, f.base + f.file_size};
    if (std::memcmp(c.p, "GGUF", 4) != 0) throw std::runtime_error("gguf: bad magic");
    c.p += 4;
    f.version = c.get<uint32_t>();
    if (f.version != 3) throw std::runtime_error("gguf: expected version 3");
    uint64_t n_tensors = c.get<uint64_t>();
    uint64_t n_kv = c.get<uint64_t>();

    for (uint64_t i = 0; i < n_kv; ++i) {
        std::string key = c.str();
        uint32_t t = c.get<uint32_t>();
        read_value(c, key, t, f);
    }
    if (f.kv_u64.count("general.alignment")) f.alignment = f.kv_u64["general.alignment"];

    f.tensors.reserve(n_tensors);
    for (uint64_t i = 0; i < n_tensors; ++i) {
        Tensor t;
        t.name = c.str();
        uint32_t nd = c.get<uint32_t>();
        if (nd > 4) throw std::runtime_error("gguf: more than 4 dims in " + t.name);
        for (uint32_t d = 0; d < nd; ++d) t.ne.push_back(c.get<uint64_t>());
        t.type = c.get<uint32_t>();
        t.offset = c.get<uint64_t>();
        f.tensors.push_back(std::move(t));
    }

    uint64_t header_end = (uint64_t) (c.p - f.base);
    f.data_start = (header_end + f.alignment - 1) / f.alignment * f.alignment;
    return f;
}

void close(File & f) {
    if (f.base) munmap(const_cast<uint8_t *>(f.base), f.file_size);
    f.base = nullptr;
}

}  // namespace gguf
