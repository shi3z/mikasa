#!/usr/bin/env python3
"""CPU F32 reference of the whole Inkling 2 Mini graph (48 applications, looped core x3).

Follows src/models/inkling2.cpp. Tokens are processed together (prefill), like llama.cpp.
Writes per-application layer outputs (l_out) and final logits to a dict / npz.
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import layer0_ref as R  # noqa: E402

E = 2304
NH, NKV, HD, NREL = 24, 8, 128, 16
NSWA = 1024
N_EXP, N_USED, FFN = 128, 8, 1152
EPS = 1e-6
QK_CLIP, REL_CLIP = 22.627416610717773, 8.0
LOG_FLOOR, LOG_ALPHA = 8192, 0.10000000149011612
SITU_BETA, SITU_LB = 4.0, 25.0
W_SCALE = 8.0
SOFTCAP = 30.0
N_VOCAB = 200058
CORE_START, CORE_END, N_PASS = 6, 18, 3
N_APP = 48
BLOCK = 6


def loop_layer(il):
    """application index -> physical layer (hparams.loop_layer)."""
    n_core = CORE_END - CORE_START
    if il < CORE_START:
        return il
    if il < CORE_START + N_PASS * n_core:
        return CORE_START + (il - CORE_START) % n_core
    return il - (N_PASS - 1) * n_core


def is_swa(il):
    """sliding-window pattern from GGUF (True = SWA layer). Full layers are il % 6 == 5."""
    return (il % 6) != 5


def bf16_mat(model, name):
    t = model.by_name[name]
    ne0, ne1 = t["dims"][0], t["dims"][1]
    return R.deq_bf16(model.raw(name)[0], ne0 * ne1).reshape(ne1, ne0)


def f32(model, name):
    return model.tensor(name)


class Experts:
    """Expert tensor [K][N][128] (NVFP4 type 40 or Q8_0 type 8), dequantized per expert on demand."""

    def __init__(self, model, name, K, N):
        self.model = model
        self.name = name
        self.K, self.N = K, N
        t = model.by_name[name]
        self.type = t["type"]
        self.base = model.data_start + t["offset"]
        self.per = N * (K // 64) * 36 if self.type == 40 else N * (K // 32) * 34

    def mat(self, e):
        self.model.f.seek(self.base + e * self.per)
        raw = self.model.f.read(self.per)
        if self.type == 40:
            return R.deq_nvfp4(raw, self.N * self.K).reshape(self.N, self.K)
        return R.deq_q8_0(raw, self.N * self.K).reshape(self.N, self.K)


def mix_multi(states, queries):
    """states: list of [T,E]; queries: [nq, E]. Returns [nq][T,E] (rms-normalized states, softmax over states)."""
    S = np.stack([R.rms_norm(s, EPS) for s in states], axis=1)           # [T,S,E]
    scores = np.einsum("tse,qe->tqs", S, queries) / math.sqrt(E)          # [T,nq,S]
    p = R.softmax(scores, axis=-1)
    out = np.einsum("tqs,tse->qte", p, S)                                 # [nq,T,E]
    return [out[q] for q in range(queries.shape[0])]


def control(cur, msg, lm, module):
    """loop message controller: (1 - tanh(F h)) * cur + tanh(G h) * msg, h = D [rms(cur); rms(msg)]."""
    h = np.concatenate([R.rms_norm(cur, EPS), R.rms_norm(msg, EPS)], axis=-1)   # [T, 2E]
    h = h @ lm["down"][module].T                                                # [T, 64]
    gain = np.tanh(h @ lm["gain"][module].T)
    forget = np.tanh(h @ lm["forget"][module].T)
    return (1.0 - forget) * cur + gain * msg


def conv_static(x, w, n=4):
    """causal conv with zero history and residual (1/sqrt(n+1))."""
    return R.conv_causal(x, w, n)


def conv_dynamic(x, w, dyn_raw):
    """static conv + input-dependent taps per KV head (dyn_raw [T, 32] = 8 heads x 4 lags)."""
    T, C = x.shape
    xs = np.concatenate([np.zeros((3, C), np.float32), x], axis=0)
    y = np.zeros_like(x)
    for j in range(4):
        y += xs[j:j + T] * w[:, j][None, :]
    taps = dyn_raw.reshape(T, NKV, 4)                                     # [T, h, lag]
    per_chan = np.repeat(taps, C // NKV, axis=1)                          # [T, C, lag]
    for lag in range(4):
        shifted = np.zeros_like(x)
        if lag == 0:
            shifted = x
        else:
            shifted[lag:] = x[:-lag]
        y += per_chan[:, :, lag] * shifted
    return (x + y) / 3.0


def attention(Q, K, V, Rr, relW, sink, swa):
    """Q [T,H,D], K/V [T,KV,D], Rr [T,H,16] (already clipped and scaled), sink [T,H]. Causal, returns [T,H,D]."""
    T = Q.shape[0]
    rep = NH // NKV
    out = np.zeros((T, NH, HD), np.float32)
    for t in range(T):
        for h in range(NH):
            js = np.arange(0, t + 1)
            if swa:
                js = js[(t - js) < NSWA]
            d = t - js
            rel = np.where(d < NSWA, relW[np.minimum(d, NSWA - 1)] @ Rr[t, h], 0.0)
            bias = rel - sink[t, h]
            s = (K[js, h // rep] @ Q[t, h]) / math.sqrt(HD) + bias
            m = max(0.0, s.max())
            e = np.exp(s - m)
            den = math.exp(-m) + e.sum()
            out[t, h] = (e / den) @ V[js, h // rep]
    return out


def run(model, toks, log=None):
    T = len(toks)
    emb = model.mat("token_embd.weight")[toks]
    cur = R.rms_norm(emb, EPS, w=f32(model, "token_embd_norm.weight"))
    blocks = []
    prelude_blocks, prelude_cur = None, None
    bank, bank_last = None, None
    lout = {}
    relW_all = {}
    Wcache = {}

    def W(name):
        if name not in Wcache:
            Wcache[name] = model.mat(name)
        return Wcache[name]

    for il in range(N_APP):
        ip = loop_layer(il)
        p = f"blk.{ip}."
        swa = is_swa(il)
        pass_idx = (il - CORE_START) // (CORE_END - CORE_START) if CORE_START <= il < CORE_START + N_PASS * (CORE_END - CORE_START) else 0
        in_core = CORE_START <= ip < CORE_END and il >= CORE_START and il < CORE_START + N_PASS * (CORE_END - CORE_START)

        if il == CORE_START:
            prelude_blocks, prelude_cur = list(blocks), cur.copy()
        if in_core and pass_idx > 0 and ip == CORE_START:
            bank = blocks[len(prelude_blocks) + 1:]
            bank_last = cur.copy()
            blocks = list(prelude_blocks)
            cur = prelude_cur.copy()

        # loop messages for core layers of pass > 0
        msgs = None
        if in_core and pass_idx > 0:
            row = 2 * (ip - CORE_START)
            qa = model.tensor("loop_res_query.weight").reshape(24, E)[row:row + 2]
            qk = model.tensor("loop_kv_res_query.weight").reshape(24, E)[row:row + 2]
            ma = mix_multi(bank + [bank_last], qa)
            mk = mix_multi(bank + [bank_last], qk)
            msgs = [ma[0], mk[0], mk[1], ma[1]]
            lm = {k: [] for k in ("down", "gain", "forget")}
            for m_ in range(4):
                nm = ["attn", "k", "v", "mlp"][m_]
                lm["down"].append(W(p + f"loop_{nm}_down.weight"))
                lm["gain"].append(W(p + f"loop_{nm}_gain.weight"))
                lm["forget"].append(W(p + f"loop_{nm}_forget.weight"))
            loop_mats = lm

        inp_sa = cur
        if ip > 0:
            aq = model.mat(p + "attn_res_query.weight") if False else model.tensor(p + "attn_res_query.weight")
            nq = 3 if not swa else 1
            aq = aq.reshape(nq, E)
            mixed = mix_multi(blocks + [cur], aq)
            inputs = [mixed[j] if not swa else mixed[0] for j in range(3)]
        else:
            inputs = [cur, cur, cur]
        if msgs is not None:
            inputs = [control(inputs[j], msgs[j], loop_mats, j) for j in range(3)]
        if swa:
            x0 = R.rms_norm(inputs[0], EPS, w=f32(model, p + "attn_norm.weight"))
            x1 = R.rms_norm(inputs[1], EPS, w=f32(model, p + "attn_norm.weight"))
            x2 = R.rms_norm(inputs[2], EPS, w=f32(model, p + "attn_norm.weight"))
        else:
            x0 = R.rms_norm(inputs[0], EPS, w=f32(model, p + "attn_q_inp_norm.weight"))
            x1 = R.rms_norm(inputs[1], EPS, w=f32(model, p + "attn_k_inp_norm.weight"))
            x2 = R.rms_norm(inputs[2], EPS, w=f32(model, p + "attn_v_inp_norm.weight"))

        Qp = x0 @ W(p + "attn_q.weight").T
        K0 = x1 @ W(p + "attn_k.weight").T
        V0 = x2 @ W(p + "attn_v.weight").T
        Rr = x0 @ W(p + "attn_rel.weight").T
        G = x0 @ W(p + "attn_gate.weight").T
        sink = x0 @ bf16_mat(model, p + "attn_sink_proj.weight").T       # [T, 24]

        kw = f32(model, p + "attn_k_conv.weight").reshape(1024, 4)
        vw = f32(model, p + "attn_v_conv.weight").reshape(1024, 4)
        if swa:
            Kc = conv_static(K0, kw)
            Vc = conv_static(V0, vw)
        else:
            dk = x1 @ bf16_mat(model, p + "attn_k_conv_proj.weight").T
            dv = x2 @ bf16_mat(model, p + "attn_v_conv_proj.weight").T
            Kc = conv_dynamic(K0, kw, dk)
            Vc = conv_dynamic(V0, vw, dv)

        Q = clip_heads(Qp.reshape(T, NH, HD), QK_CLIP, f32(model, p + "attn_q_clip.weight").reshape(NH, HD))
        K = clip_heads(Kc.reshape(T, NKV, HD), QK_CLIP, None)
        V = R.rms_norm(Vc.reshape(T, NKV, HD), EPS)
        Rc = clip_heads(Rr.reshape(T, NH, NREL), REL_CLIP, f32(model, p + "attn_rel_clip.weight").reshape(NH, NREL))
        if not swa:
            tau = 1.0 + LOG_ALPHA * np.log(np.maximum(1.0, (np.arange(T) + 1.0) / LOG_FLOOR))
            Q = Q * tau[:, None, None]
            Rc = Rc * tau[:, None, None]
        relW = f32(model, p + "attn_rel_bias.weight").reshape(NSWA, NREL)
        O = attention(Q, K, V, Rc, relW, sink, swa).reshape(T, NH * HD)
        if log is not None:
            log(f"kqv_out-{il}", O)

        gated = O * (1.0 / (1.0 + np.exp(-G)))
        attn_out = gated @ W(p + "attn_output.weight").T
        attn_post = conv_static(attn_out, f32(model, p + "attn_post_conv.weight").reshape(E, 4))

        if ip % BLOCK == 0:
            blocks.append(inp_sa)
            cur = attn_post
        else:
            cur = inp_sa + attn_post
        ffn_inp = cur

        mf = mix_multi(blocks + [cur], f32(model, p + "ffn_res_query.weight").reshape(1, E))[0]
        if msgs is not None:
            mf = control(mf, msgs[3], loop_mats, 3)
        xf = R.rms_norm(mf, EPS, w=f32(model, p + "ffn_norm.weight"))
        logits = xf @ W(p + "ffn_gate_inp.weight").T
        probs = 1.0 / (1.0 + np.exp(-logits))
        sel = probs + f32(model, p + "exp_probs_b.bias")[None, :]
        order = np.argsort(-sel, axis=1, kind="stable")[:, :N_USED]
        wv = np.take_along_axis(probs, order, axis=1)
        wv = wv / (wv.sum(axis=1, keepdims=True) + 1e-20) * W_SCALE

        gu = Experts(model, p + "ffn_gate_up_exps.weight", E, 2304)
        dn = Experts(model, p + "ffn_down_exps.weight", FFN, E)
        gu_s = f32(model, p + "ffn_gate_up_exps.scale") if (p + "ffn_gate_up_exps.scale") in model.by_name else np.ones(N_EXP, np.float32)
        dn_s = f32(model, p + "ffn_down_exps.scale") if (p + "ffn_down_exps.scale") in model.by_name else np.ones(N_EXP, np.float32)
        moe = np.zeros((T, E), np.float32)
        cache_gu, cache_dn = {}, {}
        for t in range(T):
            for k in range(N_USED):
                e = int(order[t, k])
                if e not in cache_gu:
                    cache_gu[e] = gu.mat(e)
                    cache_dn[e] = dn.mat(e)
                g_u = (cache_gu[e] @ xf[t]) * gu_s[e]
                g, up = g_u[:FFN], g_u[FFN:]
                act = SITU_BETA * np.tanh(g / SITU_BETA) * (1.0 / (1.0 + np.exp(-g)))
                h = act * (SITU_LB * np.tanh(up / SITU_LB))
                moe[t] += wv[t, k] * ((cache_dn[e] @ h) * dn_s[e])
        ffn_out = conv_static(moe, f32(model, p + "ffn_post_conv.weight").reshape(E, 4))
        cur = ffn_inp + ffn_out
        lout[il] = cur.copy()
        if log is not None:
            log(f"l_out-{il}", cur)

    # output: mix over all blocks + cur, output norm, head, softcap
    fin = mix_multi(blocks + [cur], f32(model, "output_res_query.weight").reshape(1, E))[0]
    fin = R.rms_norm(fin, EPS, w=f32(model, "output_norm.weight"))
    head = model.mat("output.weight")                                  # [201024, E]
    lg = fin @ head[:N_VOCAB].T
    lg = SOFTCAP * np.tanh(lg / SOFTCAP)
    return lout, lg


def clip_heads(x, thr, scale):
    """x [T, H, D]; per-(t,h) rows over L2 norm thr shrink; optional per-(h,d) scale first."""
    if scale is not None:
        x = x * scale[None]
    n = np.sqrt((x * x).sum(-1, keepdims=True)) + 1e-6
    return x * np.clip(thr / n, 0.0, 1.0)


def main():
    model = R.Model(R.MODEL)
    outdir = os.path.join(os.path.dirname(__file__), "..", "out_full_ref")
    os.makedirs(outdir, exist_ok=True)
    logs = {}

    def log(name, arr):
        logs[name] = np.asarray(arr, np.float32)

    toks = R.TOKENS
    lout, lg = run(model, toks, log)
    for il, v in lout.items():
        v.astype("<f4").tofile(os.path.join(outdir, f"l_out-{il}.bin"))
    lg.astype("<f4").tofile(os.path.join(outdir, "logits.bin"))
    print("logits shape", lg.shape, "argmax last token:", int(np.argmax(lg[-1])))


if __name__ == "__main__":
    main()
