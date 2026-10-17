// RUAccent encoder runner (GGUF architecture "ruaccent.encoder").
//
// Universal transformer-encoder forward pass for the 4 RUAccent models:
//   accent       (RoFormer, INT8-quantized, RoPE,          token-level logits)
//   stress_usage (BERT,     f32,                           token-level logits)
//   omograph     (DeBERTa,  f32, pooled,                   sequence-level logits)
//   yo_homograph (DistilBERT,f32,                          token-level logits)
//
// The forward pass is eager C++ (no ggml graph): the INT8
// DynamicQuantizeLinear + MatMulInteger path of the accent model must be
// reproduced bit-exactly (verified against onnxruntime in Phase 1), which a
// static ggml graph cannot express. Weights are loaded from a GGUF file via
// gguf/ggml and kept in the weight context; the forward pass reads their data
// pointers directly.
//
// All linear weights are [out, in] (row-major). Attention uses the strong mask
// scores += (1.0 - mask) * (-FLT_MAX) (matches the ONNX graphs; a no-op for
// the all-ones masks the pipeline actually feeds).
#ifndef ru_encoder_h
#define ru_encoder_h

#include <cstdint>
#include <string>
#include <vector>

struct ggml_context;
struct gguf_context;
struct ggml_tensor;

namespace ru {

struct EncoderInput {
    std::vector<int> input_ids;       // [batch * seq]
    std::vector<int> attention_mask;  // [batch * seq] (0/1)
    std::vector<int> token_type_ids;  // [batch * seq] (only if has_type_embd)
    int batch = 1;
    int seq = 0;
};

struct EncoderOutput {
    // token-level: [batch * seq * n_labels]; pooled (omograph): [batch * n_labels]
    std::vector<float> logits;
    bool pooled = false;
};

class Encoder {
public:
    Encoder() = default;
    ~Encoder();
    Encoder(const Encoder &) = delete;
    Encoder & operator=(const Encoder &) = delete;

    // Load a GGUF file with architecture "ruaccent.encoder".
    bool load(const std::string & path);

    EncoderOutput forward(const EncoderInput & in) const;

    int n_layers() const { return n_layers_; }
    int hidden() const { return hidden_; }
    int heads() const { return heads_; }
    int vocab() const { return vocab_; }
    int n_labels() const { return n_labels_; }
    int max_length() const { return max_length_; }
    bool has_type_embd() const { return has_type_embd_; }
    bool use_rope() const { return use_rope_; }
    bool quantized() const { return quantized_; }
    bool pooled() const { return pooled_; }
    float ln_eps() const { return ln_eps_; }
    const std::vector<std::string> & labels() const { return labels_; }

private:
    struct Block {
        const float * q = nullptr, * k = nullptr, * v = nullptr, * o = nullptr;
        const float * q_b = nullptr, * k_b = nullptr, * v_b = nullptr, * o_b = nullptr;
        const float * attn_norm_w = nullptr, * attn_norm_b = nullptr;
        const float * ffn0 = nullptr, * ffn0_b = nullptr;
        const float * ffn1 = nullptr, * ffn1_b = nullptr;
        const float * ffn_norm_w = nullptr, * ffn_norm_b = nullptr;
        // INT8 (accent only): q8 weights [out, in] + per-tensor f32 scale.
        const int8_t * q_q8 = nullptr, * k_q8 = nullptr, * v_q8 = nullptr, * o_q8 = nullptr;
        const int8_t * ffn0_q8 = nullptr, * ffn1_q8 = nullptr;
        float q_q8_s = 0, k_q8_s = 0, v_q8_s = 0, o_q8_s = 0, ffn0_q8_s = 0, ffn1_q8_s = 0;
    };

    EncoderOutput forward_f32(const EncoderInput & in) const;
    EncoderOutput forward_int8(const EncoderInput & in) const;

    // metadata
    int n_layers_ = 0, hidden_ = 0, heads_ = 0, vocab_ = 0, n_labels_ = 0, max_length_ = 0;
    int ffn_dim_ = 0;  // FFN intermediate size (2*hidden for RoFormer, 4*hidden for BERT-family)
    bool has_type_embd_ = false, use_rope_ = false, quantized_ = false, pooled_ = false;
    float ln_eps_ = 1e-12f;
    std::vector<std::string> labels_;

    // weights (data pointers into the weight context)
    const float * token_embd = nullptr;    // [vocab, hidden]
    const float * position_embd = nullptr; // [max_pos, hidden]
    const float * type_embd = nullptr;     // [2, hidden]
    const float * rope_embd = nullptr;     // [rope_max_pos, head_dim] (sin|cos)
    const float * norm_w = nullptr, * norm_b = nullptr;
    std::vector<Block> blocks_;
    const float * pooler_w = nullptr, * pooler_b = nullptr; // omograph only
    const float * cls_w = nullptr, * cls_b = nullptr;
    const int8_t * cls_q8 = nullptr;
    float cls_q8_s = 0;

    ggml_context * weight_ctx_ = nullptr;
    gguf_context * meta_ctx_ = nullptr;
};

}  // namespace ru

#endif
