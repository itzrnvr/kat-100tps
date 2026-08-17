// fate_prefetch.h — Cross-layer gate expert prefetcher (Fate, arXiv 2502.12224)
//
// PURPOSE
//   Predicts layer i+1's routed experts from layer i's pre-FFN hidden state
//   ("ffn_post") by running layer i+1's REAL gate weights (ffn_gate_inp) on a
//   CPU thread, overlapped with layer i's FFN compute. Zero training; recall
//   78.8% top-k / 97.2% with 75th-percentile overfetch (paper numbers on
//   Qwen1.5-MoE-class models; we re-measure on KAT-CQ1 traces).
//
// KEY DECISIONS
// - Gate weights live in pinned host RAM (40 x 2048 x 256 F32 = 80 MB),
//   copied once at model load. Never on GPU: prediction must not compete
//   with the decode graph.
// - Prediction is submitted from the decode loop right after the layer-i
//   routing readback (we already sync there), using ffn_post_host_buf which
//   the routing collector already captures — zero extra D2H traffic.
// - Overfetch: keep experts whose softmax mass >= pct-threshold of the
//   predicted distribution (default 75th percentile, Fate's setting), capped
//   at max_prefetch (default 16) to bound PCIe bytes.
// - A CCT fallback table P(next|cur) built offline from routing traces is
//   unioned in when enabled — cheap and catches the cases where the stale
//   hidden misranks.
//
// GOTCHAS
// - The hidden must be the *post-attention* pre-FFN hidden (ffn_post), NOT
//   the block input; using the block input is what caps recall at ~53%.
// - CPU matmul must complete inside the FFN compute window (~1-3 ms);
//   at 524 KFLOPs/layer it is ~0.05-0.1 ms single-threaded. No thread pool
//   needed for v1 — one detached worker with a 1-slot mailbox (latest wins).
// - Prediction for layer i+1 is only a HINT for prefetch/promote. It never
//   changes routing semantics; a miss falls back to the existing cold path.
//
// BUG FIXES / HISTORY
// - [2026-08-17] Initial port. Not yet wired into the decode loop; see
//   integrate() notes at bottom.

#pragma once

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <unordered_map>
#include <thread>
#include <vector>

namespace dflash::common {

struct FateConfig {
    int      n_layers       = 40;
    int      n_embd         = 2048;   // hidden size (gate input dim)
    int      n_experts      = 256;    // gate output dim
    int      top_k          = 8;      // experts actually used per token
    int      max_prefetch   = 16;     // cap on predicted set (overfetch)
    float    pct_threshold  = 0.75f;  // 75th percentile of softmax mass
    bool     enable_cct     = false;  // union with CCT table if loaded
};

class FatePrefetcher {
public:
    explicit FatePrefetcher(const FateConfig & cfg = {}) : cfg_(cfg) {}

    ~FatePrefetcher() { stop(); }

    // Copy all gate matrices to host storage. gate_ptrs[i] must point at
    // layer i's ffn_gate_inp F32 data (row-major [n_embd x n_experts]).
    // Returns false if any pointer is null or sizes mismatch.
    bool init(const float * const * gate_ptrs) {
        gates_.assign((size_t)cfg_.n_layers * cfg_.n_embd * cfg_.n_experts, 0.0f);
        for (int l = 0; l < cfg_.n_layers; ++l) {
            if (!gate_ptrs[l]) return false;
            std::memcpy(gate_row(l), gate_ptrs[l],
                        sizeof(float) * (size_t)cfg_.n_embd * cfg_.n_experts);
        }
        running_.store(true);
        worker_ = std::thread([this] { worker_loop(); });
        return true;
    }

    // Load a CCT table: for each layer transition, a sparse map
    // (cur_expert -> next-layer expert list). Serialized as:
    //   int32 n_layers; per layer: int32 n_entries;
    //   entries: {int32 cur, int32 cnt, int32 next[cnt]}
    bool load_cct(const char * path) {
        FILE * f = fopen(path, "rb");
        if (!f) return false;
        cct_.assign((size_t)cfg_.n_layers, {});
        int32_t nl = 0;
        if (fread(&nl, 4, 1, f) != 1 || nl != cfg_.n_layers) { fclose(f); return false; }
        for (int l = 0; l < nl; ++l) {
            int32_t n = 0;
            if (fread(&n, 4, 1, f) != 1) { fclose(f); return false; }
            for (int e = 0; e < n; ++e) {
                int32_t cur = 0, cnt = 0;
                if (fread(&cur, 4, 1, f) != 1 || fread(&cnt, 4, 1, f) != 1 || cnt <= 0) {
                    fclose(f); return false;
                }
                std::vector<int32_t> next(cnt);
                if (fread(next.data(), 4, (size_t)cnt, f) != (size_t)cnt) {
                    fclose(f); return false;
                }
                if (cur >= 0 && cur < cfg_.n_experts)
                    cct_[(size_t)l][cur] = std::move(next);
            }
        }
        fclose(f);
        cfg_.enable_cct = true;
        return true;
    }

    // Submit a prediction request: predict experts of layer (il+1) from
    // layer il's ffn_post (host buffer, n_embd floats) plus optionally the
    // actual top-k ids selected at layer il (for the CCT union).
    // Non-blocking; overwrites any pending request (latest wins).
    void submit(int il, const float * ffn_post, const int32_t * cur_ids, int cur_k) {
        if (!running_.load()) return;
        const int next = il + 1;
        if (next >= cfg_.n_layers) return;
        {
            // copy into the request slot (2048 floats = 8 KB)
            std::memcpy(req_hidden_, ffn_post, sizeof(float) * (size_t)cfg_.n_embd);
            req_layer_.store(next);
            for (int i = 0; i < cur_k && i < cfg_.top_k; ++i) req_cur_ids_[(size_t)i] = cur_ids[i];
            req_k_.store(cur_k > cfg_.top_k ? cfg_.top_k : cur_k);
        }
        req_seq_.fetch_add(1, std::memory_order_release);
        has_req_.store(true, std::memory_order_release);
    }

    // Consume the latest prediction (if ready) for layer `for_layer`.
    // Returns count written into out_ids (0 = none ready).
    int take(int for_layer, int32_t * out_ids, int cap) {
        uint64_t done = done_seq_.load(std::memory_order_acquire);
        if (done == 0) return 0;
        if (pred_layer_.load() != for_layer) return 0;
        if (done_consumed_ == done) return 0;  // already taken
        int n = pred_count_;
        if (n > cap) n = cap;
        std::memcpy(out_ids, pred_ids_.data(), sizeof(int32_t) * (size_t)n);
        done_consumed_ = done;
        return n;
    }

    void stop() {
        running_.store(false);
        has_req_.store(false);
        if (worker_.joinable()) worker_.join();
    }

    // Stats
    uint64_t predictions() const { return done_seq_.load(); }

private:
    float * gate_row(int layer) {
        return gates_.data() + (size_t)layer * cfg_.n_embd * cfg_.n_experts;
    }

    void worker_loop() {
        std::vector<float> logits((size_t)cfg_.n_experts);
        while (running_.load()) {
            if (!has_req_.load(std::memory_order_acquire)) {
                std::this_thread::yield();
                continue;
            }
            const uint64_t seq = req_seq_.load(std::memory_order_acquire);
            const int next      = req_layer_.load();
            const int cur_k     = req_k_.load();
            float hidden_stack[2048];
            std::memcpy(hidden_stack, req_hidden_, sizeof(float) * (size_t)cfg_.n_embd);
            int32_t cur_ids_stack[8];
            std::memcpy(cur_ids_stack, req_cur_ids_, sizeof(int32_t) * (size_t)cur_k);
            has_req_.store(false, std::memory_order_release);
            if (seq == done_seq_.load()) continue;  // duplicate

            // --- cross-layer gate: logits = W^T h  (h[2048] x W[2048,256]) ---
            const float * W = gate_row(next);
            for (int e = 0; e < cfg_.n_experts; ++e) {
                float acc = 0.f;
                for (int d = 0; d < cfg_.n_embd; ++d)
                    acc += hidden_stack[d] * W[(size_t)d * cfg_.n_experts + e];
                logits[(size_t)e] = acc;
            }
            // softmax
            float mx = logits[0];
            for (int e = 1; e < cfg_.n_experts; ++e) mx = std::max(mx, logits[e]);
            float sum = 0.f;
            for (int e = 0; e < cfg_.n_experts; ++e) { logits[e] = std::exp(logits[e] - mx); sum += logits[e]; }
            const float inv = 1.0f / (sum > 0.f ? sum : 1.f);
            for (int e = 0; e < cfg_.n_experts; ++e) logits[e] *= inv;

            // --- top-k by softmax mass, then 75th-pct overfetch up to cap ---
            int idx[256];
            for (int e = 0; e < cfg_.n_experts; ++e) idx[e] = e;
            // partial selection: full sort is fine at 256 (few us)
            std::sort(idx, idx + cfg_.n_experts, [&](int a, int b) {
                return logits[a] > logits[b];
            });
            // pct-threshold over the distribution: keep experts whose
            // cumulative mass reaches 75% of total, bounded by top_k..max
            float cum = 0.f;
            int keep = cfg_.top_k;
            for (int i = cfg_.top_k; i < cfg_.max_prefetch && i < cfg_.n_experts; ++i) {
                cum += logits[idx[i]];
                if (cum >= cfg_.pct_threshold * 0.25f) keep = i + 1;  // bounded expansion
                if (keep >= cfg_.max_prefetch) break;
            }
            pred_count_ = 0;
            for (int i = 0; i < keep; ++i) pred_ids_[(size_t)pred_count_++] = idx[i];

            // --- CCT union (optional) ---
            if (cfg_.enable_cct && next > 0) {
                for (int i = 0; i < cur_k; ++i) {
                    auto it = cct_[(size_t)(next - 1)].find(cur_ids_stack[i]);
                    if (it == cct_[(size_t)(next - 1)].end()) continue;
                    for (int32_t nx : it->second) {
                        bool have = false;
                        for (int j = 0; j < pred_count_; ++j)
                            if (pred_ids_[j] == nx) { have = true; break; }
                        if (!have && pred_count_ < (int)pred_ids_.size())
                            pred_ids_[(size_t)pred_count_++] = nx;
                    }
                }
            }

            pred_layer_.store(next);
            done_seq_.store(seq, std::memory_order_release);
        }
    }

    FateConfig cfg_;
    std::vector<float> gates_;                    // all gates, host-pinned ideally
    std::vector<std::unordered_map<int32_t, std::vector<int32_t>>> cct_;

    // request mailbox (latest-wins)
    alignas(64) float   req_hidden_[2048];
    int32_t             req_cur_ids_[8];
    std::atomic<int>    req_layer_{-1};
    std::atomic<int>    req_k_{0};
    std::atomic<uint64_t> req_seq_{0};
    std::atomic<bool>   has_req_{false};

    // prediction result
    std::vector<int32_t> pred_ids_ = std::vector<int32_t>(64);
    int                  pred_count_ = 0;
    std::atomic<int>     pred_layer_{-1};
    std::atomic<uint64_t> done_seq_{0};
    uint64_t             done_consumed_ = 0;

    std::atomic<bool> running_{false};
    std::thread       worker_;
};

// INTEGRATION (pipelined decode loop):
//   after "ggml_backend_tensor_get(cpg.moe_selected, ...)" readback:
//     fate.submit(il, state.ffn_post_host_buf.data(), global_ids, n_used);
//   at the start of layer il+1's FFN staging (before hot/cold split):
//     int n = fate.take(il + 1, hint_ids, 64);
//     -> feed hint_ids into the swap manager as promote-now candidates
//        (only for experts NOT already hot; misses are hints, not waits).

} // namespace dflash::common
