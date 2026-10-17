#include "ru_encoder.h"

#include <cmath>
#include <cstring>
#include <algorithm>

#include "ggml.h"

#ifdef __AVX2__
#include <immintrin.h>
#endif

namespace ru {

namespace {

constexpr float kNegInf = -3.4028234663852886e38f;  // -FLT_MAX (ONNX mask constant)

// LayerNorm over the last dim (hidden). x/out are [rows, hidden] row-major.
void layer_norm(const float * x, const float * w, const float * b, float eps,
                int rows, int hidden, float * out) {
    for (int r = 0; r < rows; r++) {
        const float * xr = x + (size_t) r * hidden;
        float * or_ = out + (size_t) r * hidden;
        float mean = 0.0f;
        for (int i = 0; i < hidden; i++) mean += xr[i];
        mean /= (float) hidden;
        float var = 0.0f;
        for (int i = 0; i < hidden; i++) { float d = xr[i] - mean; var += d * d; }
        var /= (float) hidden;
        float rstd = 1.0f / std::sqrt(var + eps);
        for (int i = 0; i < hidden; i++) or_[i] = (xr[i] - mean) * rstd * w[i] + b[i];
    }
}

// GELU (erf form), matching numpy: 0.5*x*(1+erf(x/sqrt(2))) in double -> f32.
float gelu_f(float x) {
    double xd = x;
    return (float) (0.5 * xd * (1.0 + std::erf(xd * 0.70710678118654752440084436210485)));
}

// Softmax over the last dim (n). x/out are [rows, n] row-major.
void softmax(const float * x, int rows, int n, float * out) {
    for (int r = 0; r < rows; r++) {
        const float * xr = x + (size_t) r * n;
        float * or_ = out + (size_t) r * n;
        float mx = xr[0];
        for (int i = 1; i < n; i++) mx = std::max(mx, xr[i]);
        float sum = 0.0f;
        for (int i = 0; i < n; i++) sum += std::exp(xr[i] - mx);
        float inv = 1.0f / sum;
        for (int i = 0; i < n; i++) or_[i] = std::exp(xr[i] - mx) * inv;
    }
}

// C[i, o] = sum_j A[i, j] * W[o, j] + bias[o];  A is [m, k], W is [n, k] row-major.
void gemm_f32(const float * A, const float * W, const float * bias,
              int m, int k, int n, float * C) {
    for (int i = 0; i < m; i++) {
        const float * Ar = A + (size_t) i * k;
        float * Cr = C + (size_t) i * n;
        for (int o = 0; o < n; o++) {
            const float * Wr = W + (size_t) o * k;
            float acc = bias ? bias[o] : 0.0f;
            int j = 0;
#ifdef __AVX2__
            __m256 va = _mm256_setzero_ps();
            for (; j + 8 <= k; j += 8) {
                va = _mm256_fmadd_ps(_mm256_loadu_ps(Ar + j), _mm256_loadu_ps(Wr + j), va);
            }
            float tmp[8];
            _mm256_storeu_ps(tmp, va);
            acc += (tmp[0] + tmp[4]) + (tmp[1] + tmp[5]) + (tmp[2] + tmp[6]) + (tmp[3] + tmp[7]);
#endif
            for (; j < k; j++) acc += Ar[j] * Wr[j];
            Cr[o] = acc;
        }
    }
}

// src [B, s, hd] (B heads) -> dst [B, s, h] with h = nh*hd, i.e. de-interleave heads.
// dst[((bb*s + t)*h) + (hh*hd + d)] = src[((bb*nh + hh)*s + t)*hd + d]
void unreshape_heads(const float * src, float * dst, int b, int s, int nh, int hd) {
    const int h = nh * hd;
    for (int bb = 0; bb < b; bb++)
        for (int t = 0; t < s; t++)
            for (int hh = 0; hh < nh; hh++) {
                const float * sp = src + ((size_t) (bb * nh + hh) * s + t) * hd;
                float * dp = dst + ((size_t) (bb * s + t)) * h + hh * hd;
                std::memcpy(dp, sp, hd * sizeof(float));
            }
}

// src [B, s, hd] (B = b*nh) -> dst [B, hd, s] (transpose the last two dims).
void transpose_heads(const float * src, float * dst, int B, int s, int hd) {
    for (int bh = 0; bh < B; bh++) {
        const float * sp = src + (size_t) bh * s * hd;
        float * dp = dst + (size_t) bh * hd * s;
        for (int d = 0; d < hd; d++)
            for (int t = 0; t < s; t++)
                dp[d * s + t] = sp[t * hd + d];
    }
}

// Round to nearest, ties to even — matches numpy np.round and ORT's
// RoundToEven/RoundHalfToEven (used by DynamicQuantizeLinear for both the
// zero-point and the quantized values). std::round (half-away) would diverge
// from ORT on exact .5 ties.
float round_half_to_even(float x) {
    if (!std::isfinite(x)) return x;
    float r = std::round(x);
    if (std::fabs(x) - std::trunc(std::fabs(x)) == 0.5f) {  // exact tie
        if (std::fmod(std::fabs(r), 2.0f) != 0.0f) r -= std::copysign(1.0f, x);
    }
    return r;
}

// DynamicQuantizeLinear (uint8), matching ORT (all f32).
// mn0 = min(min(x),0), mx0 = max(max(x),0), scale = (mx0-mn0)/255,
// zp = clamp(rhe(-mn0/scale),0,255), q = clamp(rhe(x/scale)+zp,0,255),
// where rhe = round-half-to-even.
void dql(const float * x, int n, uint8_t * q, float & scale, int & zp) {
    float mn = 0.0f, mx = 0.0f;
    for (int i = 0; i < n; i++) {
        mn = std::min(mn, x[i]);
        mx = std::max(mx, x[i]);
    }
    scale = (mx - mn) / 255.0f;
    if (scale == 0.0f) scale = 1.0f;  // degenerate (all-zero) tensor
    zp = (int) std::clamp(round_half_to_even(-mn / scale), 0.0f, 255.0f);
    for (int i = 0; i < n; i++) {
        float qf = round_half_to_even(x[i] / scale) + (float) zp;
        q[i] = (uint8_t) std::clamp(qf, 0.0f, 255.0f);
    }
}

// Quantized linear (no weight zero-point): DQL(x) @ w_q8, dequantized, + bias.
// out[i, o] = (f32)(int32 sum_j (q[i,j]-zx) * w_q8[o,j]) * (sx * w_scale) + bias[o].
void qlinear_int8(const float * x, const int8_t * w_q8, float w_scale, const float * bias,
                  int m, int k, int n, float * out, std::vector<uint8_t> & qscratch) {
    qscratch.resize((size_t) m * k);
    float sx; int zx;
    dql(x, m * k, qscratch.data(), sx, zx);
    float scale = sx * w_scale;
    for (int i = 0; i < m; i++) {
        const uint8_t * qr = qscratch.data() + (size_t) i * k;
        float * or_ = out + (size_t) i * n;
        for (int o = 0; o < n; o++) {
            const int8_t * wr = w_q8 + (size_t) o * k;
            int32_t acc = 0;
            for (int j = 0; j < k; j++) acc += (int32_t) (qr[j] - zx) * (int32_t) wr[j];
            or_[o] = (float) acc * scale + (bias ? bias[o] : 0.0f);
        }
    }
}

}  // namespace

Encoder::~Encoder() {
    if (weight_ctx_) ggml_free(weight_ctx_);
    if (meta_ctx_) gguf_free(meta_ctx_);
}

bool Encoder::load(const std::string & path) {
    ggml_context * wctx = nullptr;
    meta_ctx_ = gguf_init_from_file(path.c_str(), {false, &wctx});
    if (!meta_ctx_) return false;
    weight_ctx_ = wctx;

    // Metadata keys are stored unprefixed (the converter writes them as-is).
    const std::string P = "";
    auto i32 = [&](const char * k) -> int {
        int id = gguf_find_key(meta_ctx_, (P + k).c_str());
        return id >= 0 ? gguf_get_val_i32(meta_ctx_, id) : 0;
    };
    auto f32 = [&](const char * k) -> float {
        int id = gguf_find_key(meta_ctx_, (P + k).c_str());
        return id >= 0 ? gguf_get_val_f32(meta_ctx_, id) : 0.0f;
    };
    auto bl = [&](const char * k) -> bool {
        int id = gguf_find_key(meta_ctx_, (P + k).c_str());
        return id >= 0 && gguf_get_val_bool(meta_ctx_, id);
    };

    n_layers_ = i32("n_layers");
    hidden_ = i32("hidden");
    heads_ = i32("heads");
    vocab_ = i32("vocab");
    max_length_ = i32("max_length");
    has_type_embd_ = bl("has_type_embd");
    use_rope_ = bl("use_rope");
    quantized_ = bl("quantized");
    ln_eps_ = f32("ln_eps");
    if (ln_eps_ <= 0.0f) ln_eps_ = 1e-12f;
    if (hidden_ <= 0 || heads_ <= 0 || n_layers_ <= 0 || hidden_ % heads_ != 0) return false;

    int lid = gguf_find_key(meta_ctx_, (P + "labels").c_str());
    if (lid >= 0) {
        int n = gguf_get_arr_n(meta_ctx_, lid);
        for (int i = 0; i < n; i++) labels_.push_back(gguf_get_arr_str(meta_ctx_, lid, i));
    }
    n_labels_ = (int) labels_.size();

    auto getf = [&](const char * name) -> const float * {
        ggml_tensor * t = ggml_get_tensor(weight_ctx_, name);
        return t ? (const float *) t->data : nullptr;
    };
    auto geti8 = [&](const char * name) -> const int8_t * {
        ggml_tensor * t = ggml_get_tensor(weight_ctx_, name);
        return t ? (const int8_t *) t->data : nullptr;
    };

    token_embd = getf("token_embd");
    if (!token_embd) return false;
    position_embd = getf("position_embd");
    type_embd = getf("type_embd");
    rope_embd = getf("rope_embd");
    norm_w = getf("norm");
    norm_b = getf("norm_bias");
    pooler_w = getf("pooler.weight");
    pooler_b = getf("pooler.bias");
    pooled_ = (pooler_w != nullptr);
    cls_w = getf("cls.weight");
    cls_b = getf("cls.bias");
    cls_q8 = geti8("cls.weight_q8");
    if (cls_q8) {
        const float * s = getf("cls.weight_q8_scale");
        cls_q8_s = s ? s[0] : 0.0f;
    }
    if (!quantized_ && !cls_w) return false;
    if (quantized_ && !cls_q8) return false;

    // FFN intermediate size comes from the weight shape (2*hidden for the
    // RoFormer accent model, 4*hidden for the BERT-family models).
    ggml_tensor * tffn0 = ggml_get_tensor(weight_ctx_, "blk.0.ffn_0");
    if (!tffn0) tffn0 = ggml_get_tensor(weight_ctx_, "blk.0.ffn_0_q8");
    if (!tffn0) return false;
    ffn_dim_ = (int) (ggml_nelements(tffn0) / hidden_);  // [ffn_dim, hidden]
    if (ffn_dim_ <= hidden_) return false;

    blocks_.resize(n_layers_);
    for (int i = 0; i < n_layers_; i++) {
        Block & bk = blocks_[i];
        std::string base = "blk." + std::to_string(i) + ".";
        bk.q = getf((base + "attn_q").c_str());
        bk.k = getf((base + "attn_k").c_str());
        bk.v = getf((base + "attn_v").c_str());
        bk.o = getf((base + "attn_o").c_str());
        bk.q_b = getf((base + "attn_q_bias").c_str());
        bk.k_b = getf((base + "attn_k_bias").c_str());
        bk.v_b = getf((base + "attn_v_bias").c_str());
        bk.o_b = getf((base + "attn_o_bias").c_str());
        bk.attn_norm_w = getf((base + "attn_norm").c_str());
        bk.attn_norm_b = getf((base + "attn_norm_bias").c_str());
        bk.ffn0 = getf((base + "ffn_0").c_str());
        bk.ffn0_b = getf((base + "ffn_0_bias").c_str());
        bk.ffn1 = getf((base + "ffn_1").c_str());
        bk.ffn1_b = getf((base + "ffn_1_bias").c_str());
        bk.ffn_norm_w = getf((base + "ffn_norm").c_str());
        bk.ffn_norm_b = getf((base + "ffn_norm_bias").c_str());
        if (!quantized_) {
            if (!bk.q || !bk.k || !bk.v || !bk.o || !bk.ffn0 || !bk.ffn1) return false;
            continue;
        }
        bk.q_q8 = geti8((base + "attn_q_q8").c_str());
        bk.k_q8 = geti8((base + "attn_k_q8").c_str());
        bk.v_q8 = geti8((base + "attn_v_q8").c_str());
        bk.o_q8 = geti8((base + "attn_o_q8").c_str());
        bk.ffn0_q8 = geti8((base + "ffn_0_q8").c_str());
        bk.ffn1_q8 = geti8((base + "ffn_1_q8").c_str());
        const float * s;
        s = getf((base + "attn_q_q8_scale").c_str()); bk.q_q8_s = s ? s[0] : 0.0f;
        s = getf((base + "attn_k_q8_scale").c_str()); bk.k_q8_s = s ? s[0] : 0.0f;
        s = getf((base + "attn_v_q8_scale").c_str()); bk.v_q8_s = s ? s[0] : 0.0f;
        s = getf((base + "attn_o_q8_scale").c_str()); bk.o_q8_s = s ? s[0] : 0.0f;
        s = getf((base + "ffn_0_q8_scale").c_str()); bk.ffn0_q8_s = s ? s[0] : 0.0f;
        s = getf((base + "ffn_1_q8_scale").c_str()); bk.ffn1_q8_s = s ? s[0] : 0.0f;
        if (!bk.q_q8 || !bk.k_q8 || !bk.v_q8 || !bk.o_q8 || !bk.ffn0_q8 || !bk.ffn1_q8) return false;
    }
    return true;
}

EncoderOutput Encoder::forward(const EncoderInput & in) const {
    if (quantized_) return forward_int8(in);
    return forward_f32(in);
}

EncoderOutput Encoder::forward_f32(const EncoderInput & in) const {
    const int b = in.batch, s = in.seq, h = hidden_, nh = heads_, hd = h / nh;
    const int bs = b * s;
    const int ffd = ffn_dim_;
    const float inv_sqrt_hd = 1.0f / std::sqrt((float) hd);

    std::vector<float> x((size_t) bs * h), hbuf((size_t) bs * h);
    std::vector<float> q((size_t) bs * h), k((size_t) bs * h), v((size_t) bs * h);
    std::vector<float> qh((size_t) b * nh * s * hd), kh((size_t) b * nh * s * hd), vh((size_t) b * nh * s * hd);
    std::vector<float> vT((size_t) b * nh * hd * s);
    std::vector<float> scores((size_t) b * nh * s * s), attn((size_t) b * nh * s * s);
    std::vector<float> ctx((size_t) b * nh * s * hd);
    std::vector<float> aout((size_t) bs * h), aout2((size_t) bs * h);
    std::vector<float> f((size_t) bs * ffd), f2((size_t) bs * h);
    EncoderOutput out;

    for (int i = 0; i < bs; i++) {
        const float * te = token_embd + (size_t) in.input_ids[i] * h;
        float * xr = x.data() + (size_t) i * h;
        std::memcpy(xr, te, h * sizeof(float));
        if (has_type_embd_) {
            const float * tt = type_embd + (size_t) in.token_type_ids[i] * h;
            for (int j = 0; j < h; j++) xr[j] += tt[j];
        }
        if (position_embd) {
            const float * pe = position_embd + (size_t) (i % s) * h;
            for (int j = 0; j < h; j++) xr[j] += pe[j];
        }
    }
    layer_norm(x.data(), norm_w, norm_b, ln_eps_, bs, h, x.data());

    for (int li = 0; li < n_layers_; li++) {
        const Block & bk = blocks_[li];
        std::memcpy(hbuf.data(), x.data(), (size_t) bs * h * sizeof(float));

        gemm_f32(x.data(), bk.q, bk.q_b, bs, h, h, q.data());
        gemm_f32(x.data(), bk.k, bk.k_b, bs, h, h, k.data());
        gemm_f32(x.data(), bk.v, bk.v_b, bs, h, h, v.data());

        // interleave into heads: qh[bh, s, hd]
        for (int bb = 0; bb < b; bb++)
            for (int t = 0; t < s; t++)
                for (int hh = 0; hh < nh; hh++) {
                    std::memcpy(qh.data() + ((size_t) (bb * nh + hh) * s + t) * hd,
                                q.data() + ((size_t) (bb * s + t)) * h + hh * hd, hd * sizeof(float));
                    std::memcpy(kh.data() + ((size_t) (bb * nh + hh) * s + t) * hd,
                                k.data() + ((size_t) (bb * s + t)) * h + hh * hd, hd * sizeof(float));
                    std::memcpy(vh.data() + ((size_t) (bb * nh + hh) * s + t) * hd,
                                v.data() + ((size_t) (bb * s + t)) * h + hh * hd, hd * sizeof(float));
                }

        for (int bh = 0; bh < b * nh; bh++) {
            float * qhp = qh.data() + (size_t) bh * s * hd;
            float * khp = kh.data() + (size_t) bh * s * hd;
            float * vhp = vh.data() + (size_t) bh * s * hd;
            float * vTp = vT.data() + (size_t) bh * hd * s;
            float * sp = scores.data() + (size_t) bh * s * s;
            float * ap = attn.data() + (size_t) bh * s * s;
            float * ctxp = ctx.data() + (size_t) bh * s * hd;

            if (pooled_) {
                for (int i = 0; i < s * hd; i++) qhp[i] *= inv_sqrt_hd;  // deberta: scale q
                gemm_f32(qhp, khp, nullptr, s, hd, s, sp);
            } else {
                gemm_f32(qhp, khp, nullptr, s, hd, s, sp);
                for (int i = 0; i < s * s; i++) sp[i] *= inv_sqrt_hd;
            }
            int bb = bh / nh;
            const int * maskb = in.attention_mask.data() + (size_t) bb * s;
            for (int t = 0; t < s; t++) {
                float mterm = (1.0f - (float) maskb[t]) * kNegInf;
                if (mterm == 0.0f) continue;
                for (int i = 0; i < s; i++) sp[i * s + t] += mterm;
            }
            softmax(sp, s, s, ap);

            for (int d = 0; d < hd; d++)
                for (int t = 0; t < s; t++)
                    vTp[d * s + t] = vhp[t * hd + d];
            gemm_f32(ap, vTp, nullptr, s, s, hd, ctxp);
        }

        unreshape_heads(ctx.data(), aout.data(), b, s, nh, hd);
        gemm_f32(aout.data(), bk.o, bk.o_b, bs, h, h, aout2.data());
        for (int i = 0; i < bs * h; i++) x.data()[i] = aout2.data()[i] + hbuf.data()[i];
        layer_norm(x.data(), bk.attn_norm_w, bk.attn_norm_b, ln_eps_, bs, h, x.data());

        gemm_f32(x.data(), bk.ffn0, bk.ffn0_b, bs, h, ffd, f.data());
        for (int i = 0; i < bs * ffd; i++) f.data()[i] = gelu_f(f.data()[i]);
        gemm_f32(f.data(), bk.ffn1, bk.ffn1_b, bs, ffd, h, f2.data());
        for (int i = 0; i < bs * h; i++) x.data()[i] = f2.data()[i] + x.data()[i];
        layer_norm(x.data(), bk.ffn_norm_w, bk.ffn_norm_b, ln_eps_, bs, h, x.data());
    }

    if (pooled_) {
        std::vector<float> p((size_t) b * h), p2((size_t) b * h);
        for (int bb = 0; bb < b; bb++)
            std::memcpy(p.data() + (size_t) bb * h, x.data() + (size_t) bb * s * h, h * sizeof(float));
        gemm_f32(p.data(), pooler_w, pooler_b, b, h, h, p2.data());
        for (int i = 0; i < b * h; i++) p2.data()[i] = gelu_f(p2.data()[i]);
        out.logits.resize((size_t) b * n_labels_);
        gemm_f32(p2.data(), cls_w, cls_b, b, h, n_labels_, out.logits.data());
        out.pooled = true;
    } else {
        out.logits.resize((size_t) bs * n_labels_);
        gemm_f32(x.data(), cls_w, cls_b, bs, h, n_labels_, out.logits.data());
        out.pooled = false;
    }
    return out;
}

EncoderOutput Encoder::forward_int8(const EncoderInput & in) const {
    const int b = in.batch, s = in.seq, h = hidden_, nh = heads_, hd = h / nh;
    const int bs = b * s;
    const int ffd = ffn_dim_;
    const int B = b * nh;
    const float inv_sqrt_hd = 1.0f / std::sqrt((float) hd);

    std::vector<float> x((size_t) bs * h), hbuf((size_t) bs * h);
    std::vector<float> q((size_t) bs * h), k((size_t) bs * h), v((size_t) bs * h);
    std::vector<float> qh((size_t) B * s * hd), kh((size_t) B * s * hd), vh((size_t) B * s * hd);
    std::vector<float> kT((size_t) B * hd * s);
    std::vector<float> scores((size_t) B * s * s), attn((size_t) B * s * s);
    std::vector<float> ctx((size_t) B * s * hd);
    std::vector<float> aout((size_t) bs * h), aout2((size_t) bs * h);
    std::vector<float> f((size_t) bs * ffd), f2((size_t) bs * h);
    std::vector<uint8_t> qv((size_t) B * s * hd), qq((size_t) B * s * hd), kq((size_t) B * hd * s), qa((size_t) B * s * s);
    std::vector<uint8_t> qlin;
    EncoderOutput out;

    for (int i = 0; i < bs; i++) {
        const float * te = token_embd + (size_t) in.input_ids[i] * h;
        float * xr = x.data() + (size_t) i * h;
        std::memcpy(xr, te, h * sizeof(float));
        if (has_type_embd_) {
            const float * tt = type_embd + (size_t) in.token_type_ids[i] * h;
            for (int j = 0; j < h; j++) xr[j] += tt[j];
        }
    }
    layer_norm(x.data(), norm_w, norm_b, ln_eps_, bs, h, x.data());

    // RoPE table for positions 0..s-1 (interleaved-pair: sin|cos, each pair duplicated).
    std::vector<float> sin_pos((size_t) s * hd), cos_pos((size_t) s * hd);
    for (int t = 0; t < s; t++) {
        const float * tab = rope_embd + (size_t) t * hd;
        for (int i = 0; i < hd / 2; i++) {
            sin_pos[t * hd + 2 * i] = tab[i];
            sin_pos[t * hd + 2 * i + 1] = tab[i];
            cos_pos[t * hd + 2 * i] = tab[hd / 2 + i];
            cos_pos[t * hd + 2 * i + 1] = tab[hd / 2 + i];
        }
    }

    for (int li = 0; li < n_layers_; li++) {
        const Block & bk = blocks_[li];
        std::memcpy(hbuf.data(), x.data(), (size_t) bs * h * sizeof(float));

        qlinear_int8(x.data(), bk.q_q8, bk.q_q8_s, bk.q_b, bs, h, h, q.data(), qlin);
        qlinear_int8(x.data(), bk.k_q8, bk.k_q8_s, bk.k_b, bs, h, h, k.data(), qlin);
        qlinear_int8(x.data(), bk.v_q8, bk.v_q8_s, bk.v_b, bs, h, h, v.data(), qlin);

        for (int bb = 0; bb < b; bb++)
            for (int t = 0; t < s; t++)
                for (int hh = 0; hh < nh; hh++) {
                    std::memcpy(qh.data() + ((size_t) (bb * nh + hh) * s + t) * hd,
                                q.data() + ((size_t) (bb * s + t)) * h + hh * hd, hd * sizeof(float));
                    std::memcpy(kh.data() + ((size_t) (bb * nh + hh) * s + t) * hd,
                                k.data() + ((size_t) (bb * s + t)) * h + hh * hd, hd * sizeof(float));
                    std::memcpy(vh.data() + ((size_t) (bb * nh + hh) * s + t) * hd,
                                v.data() + ((size_t) (bb * s + t)) * h + hh * hd, hd * sizeof(float));
                }
        // RoPE on q and k (interleaved pairs), q/k only.
        for (int bh = 0; bh < B; bh++) {
            float * qp = qh.data() + (size_t) bh * s * hd;
            float * kp = kh.data() + (size_t) bh * s * hd;
            for (int t = 0; t < s; t++) {
                for (int i = 0; i < hd / 2; i++) {
                    float sp_ = sin_pos[t * hd + 2 * i];
                    float cp_ = cos_pos[t * hd + 2 * i];
                    float q0 = qp[t * hd + 2 * i], q1 = qp[t * hd + 2 * i + 1];
                    qp[t * hd + 2 * i] = q0 * cp_ - q1 * sp_;
                    qp[t * hd + 2 * i + 1] = q1 * cp_ + q0 * sp_;
                    float k0 = kp[t * hd + 2 * i], k1 = kp[t * hd + 2 * i + 1];
                    kp[t * hd + 2 * i] = k0 * cp_ - k1 * sp_;
                    kp[t * hd + 2 * i + 1] = k1 * cp_ + k0 * sp_;
                }
            }
        }
        // DQL over the whole head tensors (per-tensor scale/zp).
        float sq, sk, sv; int zq, zk, zv;
        dql(qh.data(), B * s * hd, qq.data(), sq, zq);
        transpose_heads(kh.data(), kT.data(), B, s, hd);
        dql(kT.data(), B * hd * s, kq.data(), sk, zk);
        dql(vh.data(), B * s * hd, qv.data(), sv, zv);

        for (int bh = 0; bh < B; bh++) {
            const uint8_t * qqp = qq.data() + (size_t) bh * s * hd;
            const uint8_t * kqp = kq.data() + (size_t) bh * hd * s;
            float * sp = scores.data() + (size_t) bh * s * s;
            for (int i = 0; i < s; i++)
                for (int t = 0; t < s; t++) {
                    int32_t acc = 0;
                    for (int d = 0; d < hd; d++)
                        acc += (int32_t) (qqp[i * hd + d] - zq) * (int32_t) (kqp[d * s + t] - zk);
                    sp[i * s + t] = (float) acc * (sq * sk) * inv_sqrt_hd;
                }
            int bb = bh / nh;
            const int * maskb = in.attention_mask.data() + (size_t) bb * s;
            for (int t = 0; t < s; t++) {
                float mterm = (1.0f - (float) maskb[t]) * kNegInf;
                if (mterm == 0.0f) continue;
                for (int i = 0; i < s; i++) sp[i * s + t] += mterm;
            }
            softmax(sp, s, s, attn.data() + (size_t) bh * s * s);
        }

        float sa; int za;
        dql(attn.data(), B * s * s, qa.data(), sa, za);
        for (int bh = 0; bh < B; bh++) {
            const uint8_t * qap = qa.data() + (size_t) bh * s * s;
            const uint8_t * qvp = qv.data() + (size_t) bh * s * hd;
            float * ctxp = ctx.data() + (size_t) bh * s * hd;
            for (int i = 0; i < s; i++)
                for (int d = 0; d < hd; d++) {
                    int32_t acc = 0;
                    for (int t = 0; t < s; t++)
                        acc += (int32_t) (qap[i * s + t] - za) * (int32_t) (qvp[t * hd + d] - zv);
                    ctxp[i * hd + d] = (float) acc * (sa * sv);
                }
        }

        unreshape_heads(ctx.data(), aout.data(), b, s, nh, hd);
        qlinear_int8(aout.data(), bk.o_q8, bk.o_q8_s, bk.o_b, bs, h, h, aout2.data(), qlin);
        for (int i = 0; i < bs * h; i++) x.data()[i] = aout2.data()[i] + hbuf.data()[i];
        layer_norm(x.data(), bk.attn_norm_w, bk.attn_norm_b, ln_eps_, bs, h, x.data());

        qlinear_int8(x.data(), bk.ffn0_q8, bk.ffn0_q8_s, bk.ffn0_b, bs, h, ffd, f.data(), qlin);
        for (int i = 0; i < bs * ffd; i++) f.data()[i] = gelu_f(f.data()[i]);
        qlinear_int8(f.data(), bk.ffn1_q8, bk.ffn1_q8_s, bk.ffn1_b, bs, ffd, h, f2.data(), qlin);
        for (int i = 0; i < bs * h; i++) x.data()[i] = f2.data()[i] + x.data()[i];
        layer_norm(x.data(), bk.ffn_norm_w, bk.ffn_norm_b, ln_eps_, bs, h, x.data());
    }

    out.logits.resize((size_t) bs * n_labels_);
    qlinear_int8(x.data(), cls_q8, cls_q8_s, cls_b, bs, h, n_labels_, out.logits.data(), qlin);
    out.pooled = false;
    return out;
}

}  // namespace ru
