// RUAccent dictionary loader (binary format produced by
// py-gguf/convert_ruaccent_to_gguf/ruaccent_dict_to_binary.py).
//
// Four dictionaries in the reference pipeline (dictionary/<name>.json.gz):
//   accents          word -> accented word            (str -> str)
//   omographs        word -> syllable-split variants  (str -> [str])
//   yo_words         word -> yo-fixed word            (str -> str)
//   yo_homographs    word -> yo variant               (str -> str)
//
// Binary layout (little-endian):
//   magic    "RUAC"
//   version  u32 = 1
//   count    u64
//   entries, sorted by UTF-8 byte order:
//     key_len u32, key bytes, val_len u32, value bytes
//
// List values (omographs) are stored as '\n'-joined variants. Keys are
// looked up case-sensitively; the pipeline lower-cases before lookup.
#ifndef ru_dict_h
#define ru_dict_h

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace ru {

class Dict {
public:
    bool load(const std::string & path);

    // String value. Returns false if the key is absent.
    bool get(const std::string & key, std::string & out) const;
    // List value ('\n'-joined in the file). Returns false if absent.
    bool get_list(const std::string & key, std::vector<std::string> & out) const;

    size_t size() const { return (size_t) count_; }

private:
    struct Entry { uint32_t key_off, key_len, val_off, val_len; };
    const Entry * find(const std::string & key) const;

    std::vector<Entry> entries_;
    std::string blob_;   // all keys and values, concatenated
    uint64_t count_ = 0;
};

}  // namespace ru

#endif
