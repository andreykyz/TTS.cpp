// RUAccent tokenizers ported from the reference Python (ruaccent 1.5.8.3).
//
// Three tokenizer families, one per model group:
//   CharTokenizer      -> nn_accent (RoFormer, char-level, [bos]/[eos])
//   WordPieceTokenizer -> nn_stress_usage_predictor + nn_yo_homograph_resolver
//                         (BERT/DistilBERT, [CLS]/[SEP], offsets + special mask)
//   BPETokenizer       -> nn_omograph/turbo3.1 (DeBERTa, GPT-2 byte-level BPE,
//                         RoBERTa <s>/</s> pair post-processing)
//
// Inputs are expected to be pre-normalized by the RUAccent pipeline (the
// normalize regex strips control chars / disallowed symbols), so the
// normalizer step is a no-op here. Verified against golden files generated
// by the reference tokenizers (tests/ru_tokenizer/gen_golden.py).
#ifndef ru_tokenizer_h
#define ru_tokenizer_h

#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace ru {

// Output of a tokenization. Fields not produced by a given tokenizer are left
// empty (e.g. offsets/special_mask only for WordPiece, token_type_ids only for
// the BERT/DistilBERT models).
struct TokOut {
    std::vector<int> input_ids;
    std::vector<int> attention_mask;
    std::vector<int> token_type_ids;
    std::vector<std::pair<int, int>> offsets;   // WordPiece only
    std::vector<int> special_mask;              // WordPiece only
};

// Char-level tokenizer for the accent model. Lower-cases, then
// [bos] + per-char ids + [eos]. Unknown char -> [unk].
class CharTokenizer {
public:
    // Loads vocab.txt (one token per line, index = line number).
    bool load(const std::string & vocab_path);
    TokOut tokenize(const std::string & text) const;
    int vocab_size() const { return (int) vocab_.size(); }
private:
    std::unordered_map<std::string, int> vocab_;   // token -> id
    int bos_ = 2, eos_ = 3, unk_ = 1;
};

// WordPiece tokenizer (BERT/DistilBERT). Pre-tokenizes (letters/numbers/
// punctuation/whitespace runs), greedy longest-match with "##" subword prefix,
// then [CLS] + tokens + [SEP]. Produces offsets + special_tokens_mask.
class WordPieceTokenizer {
public:
    // Loads vocab.txt (one token per line, index = line number).
    bool load(const std::string & vocab_path);
    TokOut tokenize(const std::string & text) const;
    int vocab_size() const { return (int) vocab_.size(); }
    // Token string for an id (keeps the "##" subword prefix, if any).
    const std::string & id_to_token(int id) const;
private:
    std::unordered_map<std::string, int> vocab_;   // token -> id
    std::vector<std::string> id2tok_;              // id -> token (reverse map)
    int cls_ = 2, sep_ = 3, unk_ = 1;
    int max_chars_ = 100;
};

// Byte-level BPE tokenizer (GPT-2 style) for the omograph model. Pair
// post-processing: <s> A </s> </s> B </s> (RoBERTa).
class BPETokenizer {
public:
    // Loads vocab.json + merges.txt (+ added_tokens.json) from `dir`.
    bool load(const std::string & dir);
    TokOut tokenize_pair(const std::string & a, const std::string & b) const;
    int vocab_size() const { return (int) vocab_.size(); }
private:
    std::unordered_map<std::string, int> vocab_;   // token -> id
    std::unordered_map<std::string, int> merge_ranks_;   // "a b" -> rank
    int cls_ = 0, sep_ = 2;   // <s>=0, </s>=2
};

}  // namespace ru

#endif
