#include "ru_dict.h"

#include <algorithm>
#include <cstring>
#include <fstream>

namespace ru {

namespace {

bool read_file(const std::string & path, std::vector<char> & out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    f.seekg(0, std::ios::end);
    std::streamoff n = f.tellg();
    if (n < 0) return false;
    f.seekg(0, std::ios::beg);
    out.resize((size_t) n);
    if (n > 0 && !f.read(out.data(), n)) return false;
    return true;
}

uint32_t rd_u32(const char * p) {
    return (uint32_t) (uint8_t) p[0] | ((uint32_t) (uint8_t) p[1] << 8) |
           ((uint32_t) (uint8_t) p[2] << 16) | ((uint32_t) (uint8_t) p[3] << 24);
}

uint64_t rd_u64(const char * p) {
    return (uint64_t) rd_u32(p) | ((uint64_t) rd_u32(p + 4) << 32);
}

}  // namespace

bool Dict::load(const std::string & path) {
    entries_.clear();
    blob_.clear();
    count_ = 0;

    std::vector<char> data;
    if (!read_file(path, data)) return false;
    const char * base = data.data();
    const size_t n = data.size();
    const char * p = base;

    if (n < 4 + 4 + 8 || std::memcmp(p, "RUAC", 4) != 0) return false;
    p += 4;
    if (rd_u32(p) != 1) return false;
    p += 4;
    count_ = rd_u64(p);
    p += 8;

    entries_.resize((size_t) count_);
    for (uint64_t i = 0; i < count_; i++) {
        if ((size_t) (p - base) + 4 > n) return false;
        uint32_t klen = rd_u32(p);
        p += 4;
        if ((size_t) (p - base) + klen > n) return false;
        uint32_t koff = (uint32_t) blob_.size();
        blob_.append(p, klen);
        p += klen;
        if ((size_t) (p - base) + 4 > n) return false;
        uint32_t vlen = rd_u32(p);
        p += 4;
        if ((size_t) (p - base) + vlen > n) return false;
        uint32_t voff = (uint32_t) blob_.size();
        blob_.append(p, vlen);
        p += vlen;
        entries_[i] = {koff, klen, voff, vlen};
    }
    if ((size_t) (p - base) != n) return false;
    return true;
}

const Dict::Entry * Dict::find(const std::string & key) const {
    size_t lo = 0, hi = entries_.size();
    while (lo < hi) {
        size_t mid = lo + (hi - lo) / 2;
        const Entry & e = entries_[mid];
        size_t n = std::min((size_t) e.key_len, key.size());
        int c = std::memcmp(blob_.data() + e.key_off, key.data(), n);
        if (c == 0) {
            if (e.key_len == key.size()) return &e;
            if (e.key_len < key.size()) lo = mid + 1;
            else hi = mid;
        } else if (c < 0) lo = mid + 1;
        else hi = mid;
    }
    return nullptr;
}

bool Dict::get(const std::string & key, std::string & out) const {
    const Entry * e = find(key);
    if (!e) return false;
    out.assign(blob_.data() + e->val_off, e->val_len);
    return true;
}

bool Dict::get_list(const std::string & key, std::vector<std::string> & out) const {
    std::string v;
    if (!get(key, v)) return false;
    out.clear();
    size_t start = 0;
    for (size_t i = 0; i <= v.size(); i++) {
        if (i == v.size() || v[i] == '\n') {
            out.emplace_back(v.data() + start, i - start);
            start = i + 1;
        }
    }
    return true;
}

}  // namespace ru
