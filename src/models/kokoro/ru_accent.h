// RUAccent stress pipeline (full port of the reference Python `RUAccent`).
//
// Composes the 4 RUAccent models + 3 dictionaries + text pre/post-processing
// into the stress step of the Russian G2P frontend:
//
//   text -> normalize -> split sentences
//        -> per sentence: split words
//             -> stress_usage (BERT)      per-word STRESS/NO_STRESS/PUNCT
//             -> yo_homograph (DistilBERT)+ yo dicts   е/ё resolution
//             -> omograph (DeBERTa)       + omographs dict  disambiguation
//             -> accent (RoFormer)        + accents dict    '+' stress marks
//        -> reassemble -> delete spaces before punctuation
//
// Output is text with '+' inserted before each stressed vowel; the frontend's
// plus_to_acute() then moves it to a U+0301 combining acute after the vowel.
//
// Verified against the Python reference (tests/ru_accent/gen_oracle.py).
#ifndef ru_accent_h
#define ru_accent_h

#include <string>
#include <vector>

#include "ru_encoder.h"
#include "ru_tokenizer.h"
#include "ru_dict.h"

namespace ru {

class RUAccent {
public:
    RUAccent() = default;
    ~RUAccent();
    RUAccent(const RUAccent &) = delete;
    RUAccent & operator=(const RUAccent &) = delete;

    // data_dir   : holds nn/ (tokenizer files) and dictionary/ (*.bin).
    // models_dir : holds accent.gguf, stress_usage.gguf, yo_homograph.gguf,
    //              omograph.gguf.
    bool load(const std::string & data_dir, const std::string & models_dir);

    // Full pipeline: text -> text with '+' stress marks.
    std::string process(const std::string & text) const;

    bool loaded() const { return loaded_; }

private:
    // text helpers
    std::string normalize(const std::string & text) const;
    std::vector<std::string> split_by_sentences(const std::string & text) const;
    void split_by_words(const std::string & s,
                        std::vector<std::string> & words,
                        std::vector<std::string> & remaining) const;
    std::string delete_spaces_before_punc(const std::string & text) const;
    int  count_vowels(const std::string & text) const;
    bool has_punctuation(const std::string & text) const;
    std::string fix_capital(const std::string & source, const std::string & target) const;

    // model wrappers
    // WordPiece sequence model (stress_usage / yo_homograph): returns one label
    // string per word (AVERAGE aggregation over subwords), in word order.
    std::vector<std::string> wordpiece_predict(
            const std::string & sentence,
            const Encoder & enc, const WordPieceTokenizer & tok) const;
    std::vector<std::string> process_yo(std::vector<std::string> words,
                                        const std::string & sentence) const;
    std::vector<std::string> process_omographs(std::vector<std::string> words) const;
    std::vector<std::string> process_accent(std::vector<std::string> words,
                                            const std::vector<std::string> & stress_usages) const;
    std::string put_accent(const std::string & word) const;

    // models
    Encoder accent_, stress_, omograph_, yo_;
    // tokenizers
    CharTokenizer      accent_tok_;
    WordPieceTokenizer stress_tok_, yo_tok_;
    BPETokenizer       omograph_tok_;
    // dictionaries
    Dict accents_, omographs_, yo_words_, yo_homographs_;

    bool loaded_ = false;
};

}  // namespace ru

#endif
