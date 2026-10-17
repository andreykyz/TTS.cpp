// Russian G2P frontend ported from kokoro-ru/ru_g2p.py.
//
// This is the pure-text portion of the pipeline (no ONNX, no espeak). It runs
// around the espeak-ng phonemization:
//
//   text -> brackets -> [RUAccent stress] -> plus_to_acute -> lower -> respell
//        -> espeak(IPA) -> normalize -> lengthen_sch -> reduce_vowels
//
// The RUAccent stress step is a separate model-based component (ru_accent.cpp,
// class ru::RUAccent::process); it is composed around these pure-text steps.
// The functions below operate on UTF-8 std::string and are unit-testable in
// isolation.
#ifndef ru_frontend_h
#define ru_frontend_h

#include <cstdint>
#include <string>
#include <vector>

namespace ru {

// UTF-8 <-> codepoint helpers.
std::vector<uint32_t> utf8_decode(const std::string & s);
std::string           utf8_encode(const std::vector<uint32_t> & cps);

// text -> text
//   fold '[' '{' -> '(' and ']' '}' -> ')' so espeak treats asides as pauses
std::string brackets(const std::string & text);
//   RUAccent emits '+' before the stressed vowel; move it to U+0301 after it
std::string plus_to_acute(const std::string & text);
//   orthoepic respelling (дорого->дорога, солнце->сонце, -ого->-ово, ...)
//   input is expected to be lower-cased and acute-marked (U+0301)
std::string respell(const std::string & text);

// ipa -> ipa
//   fold espeak-ng leaks (u", ɭ, ɵ, ʑ, ʐ, ʧʲ) into the model vocab
std::string normalize(const std::string & ipa);
//   щ is long [ɕː]; espeak emits a short [ɕ] and never emits ː
std::string lengthen_sch(const std::string & ipa);
//   positional Russian vowel reduction (akanye + post-tonic schwa)
std::string reduce_vowels(const std::string & ipa);

}  // namespace ru

#endif
