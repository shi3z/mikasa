#!/usr/bin/env python3
"""F32 CPU reference of Inkling 2 Mini layer 0 (prefill of a short prompt).

Follows src/models/inkling2.cpp (il = 0, SWA layer, no loop messages).
Dequantizes the weights in numpy and compares each intermediate tensor with the
dumps of the llama.cpp CPU build (engine/ref/dump_all).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from gguf_inspect import read_gguf  # noqa: E402

MODEL = os.path.expanduser("~/inkling-harness/models/Inkling-2-Mini-UD-Q4_K_XL.gguf")
DUMP = os.path.join(os.path.dirname(__file__), "dump_all")
TOKENS = [371, 151451, 149879, 151194, 149884]  # from the llama.cpp dump run
EPS = 1e-6
N_EMBD = 2304
N_HEAD = 24
N_KV = 8
HD = 128
N_REL = 16
N_SWA = 1024
QK_CLIP = 22.627416610717773
REL_CLIP = 8.0
ALPHA_SITU_BETA = 4.0
ALPHA_SITU_LB = 25.0
W_SCALE = 8.0
N_EXPERT = 128
N_USED = 8

KVAL = np.array([0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12], dtype=np.float32)


# ---------------------------------------------------------------- dequant ----

def ue4m3_to_f32(x):
    """ggml_ue4m3_to_fp32: unsigned E4M3, bias 7, returns value * 0.5."""
    x = np.asarray(x, dtype=np.uint8)
    exp = (x >> 3) & 0xF
    man = (x & 0x7).astype(np.float32)
    raw = np.where(exp == 0, man * 2.0 ** -9, (1.0 + man / 8.0) * np.exp2(exp.astype(np.float32) - 7.0))
    raw = np.where((x == 0) | (x == 0x7F), 0.0, raw)
    return (raw * 0.5).astype(np.float32)


def deq_f32(raw, shape):
    return np.frombuffer(raw, dtype="<f4").reshape(shape).astype(np.float32)


def deq_bf16(raw, shape):
    u = np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16
    return u.view(np.float32).reshape(shape)


def deq_q8_0(raw, n_elem):
    blocks = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 34)
    d = blocks[:, :2].copy().view("<f2").astype(np.float32)[:, 0]
    q = blocks[:, 2:].view(np.int8).astype(np.float32)
    return (q * d[:, None]).reshape(n_elem)


def deq_nvfp4(raw, n_elem):
    blocks = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 36)
    scales = ue4m3_to_f32(blocks[:, :4])            # [nb, 4]
    qs = blocks[:, 4:].reshape(-1, 4, 8)            # [nb, sub, j]
    lo = KVAL[qs & 0x0F]                            # elements s*16 + j
    hi = KVAL[qs >> 4]                              # elements s*16 + j + 8
    out = np.zeros((blocks.shape[0], 4, 16), dtype=np.float32)
    out[:, :, :8] = lo * scales[:, :, None]
    out[:, :, 8:] = hi * scales[:, :, None]
    return out.reshape(n_elem)


class Model:
    def __init__(self, path):
        self.g = read_gguf(path)
        self.data_start = self.g["data_start"]
        self.by_name = {t["name"]: t for t in self.g["tensors"]}
        self.f = open(path, "rb")

    def raw(self, name):
        t = self.by_name[name]
        n_elem = int(np.prod(t["dims"]))
        blck, nbytes = {0: (1, 4), 1: (1, 2), 8: (32, 34), 30: (1, 2), 40: (64, 36)}[t["type"]]
        size = n_elem // blck * nbytes
        self.f.seek(self.data_start + t["offset"])
        return self.f.read(size), n_elem, t

    def tensor(self, name):
        """Returns a flat float32 array in ggml order (ne0 fastest)."""
        raw, n, t = self.raw(name)
        if t["type"] == 0:
            return deq_f32(raw, n)
        if t["type"] == 30:
            return deq_bf16(raw, n).reshape(-1)
        if t["type"] == 8:
            return deq_q8_0(raw, n)
        if t["type"] == 40:
            return deq_nvfp4(raw, n)
        raise ValueError(f"unsupported type {t['type']} for {name}")

    def mat(self, name):
        """2D weight as [ne1, ne0] so that y = x @ W.T for x [.., ne0]."""
        t = self.by_name[name]
        ne0, ne1 = t["dims"][0], t["dims"][1]
        return self.tensor(name).reshape(ne1, ne0)


# ------------------------------------------------------------- primitives ----

def rms_norm(x, eps=EPS, w=None):
    y = x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)
    return y if w is None else y * w


def clip(x, thr, scale=None):
    """inkling2 clip(): optional per-channel scale, then rows over L2 norm thr are shrunk."""
    if scale is not None:
        x = x * scale
    n = np.sqrt(np.sum(x * x, axis=-1, keepdims=True)) + 1e-6
    f = np.clip(thr / n, 0.0, 1.0)
    return x * f


def conv_causal(x, w, n_conv=4):
    """convolve() without dynamic taps and with zero history.
    x [T, C], w [C, n_conv] (w[c, j] = ggml weight[j + n_conv*c]); returns (x + y)/sqrt(n_conv+1)."""
    T = x.shape[0]
    xs = np.concatenate([np.zeros((n_conv - 1, x.shape[1]), np.float32), x], axis=0)
    y = np.zeros_like(x)
    for j in range(n_conv):
        y += xs[j:j + T] * w[:, j][None, :]
    return (x + y) / np.sqrt(n_conv + 1.0)


def softmax(x, axis=-1):
    m = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / np.sum(e, axis=axis, keepdims=True)


# ------------------------------------------------------------- layer 0 -------

def layer0(m, log):
    T = len(TOKENS)
    emb = m.mat("token_embd.weight")[TOKENS]                     # [T, E]
    cur = rms_norm(emb, w=m.tensor("token_embd_norm.weight"))

    # attention input (SWA layer 0): one normalized input for Q, K, V
    attn_norm_w = m.tensor("blk.0.attn_norm.weight")
    x = rms_norm(cur, w=attn_norm_w)
    log("norm-0#1", rms_norm(cur))            # llama.cpp dumps the norm before the weight multiply
    inp_sa = cur                                                 # block start: this residual goes to blocks

    Q = x @ m.mat("blk.0.attn_q.weight").T                       # [T, 3072]
    Kp = x @ m.mat("blk.0.attn_k.weight").T                      # [T, 1024]
    Vp = x @ m.mat("blk.0.attn_v.weight").T
    Rr = x @ m.mat("blk.0.attn_rel.weight").T                    # [T, 384]
    kconv = m.tensor("blk.0.attn_k_conv.weight").reshape(1024, 4)
    vconv = m.tensor("blk.0.attn_v_conv.weight").reshape(1024, 4)
    K = conv_causal(Kp, kconv)
    V = conv_causal(Vp, vconv)

    Q = Q.reshape(T, N_HEAD, HD)
    Q = clip(Q, QK_CLIP, scale=m.tensor("blk.0.attn_q_clip.weight").reshape(N_HEAD, HD))
    K = clip(K.reshape(T, N_KV, HD), QK_CLIP)
    V = rms_norm(V.reshape(T, N_KV, HD), EPS)
    R = clip(Rr.reshape(T, N_HEAD, N_REL), REL_CLIP,
             scale=m.tensor("blk.0.attn_rel_clip.weight").reshape(N_HEAD, N_REL))
    log("Qcur-0", Q)
    log("Kcur-0", K)
    log("Vcur-0", V)
    log("Rcur-0", R)

    # relative bias table: table[t, h, k] = sum_r W[k, r] R[t, h, r], row N_SWA is zero
    W = m.tensor("blk.0.attn_rel_bias.weight").reshape(N_SWA, N_REL)  # W[k, r]
    W = np.concatenate([W, np.zeros((1, N_REL), np.float32)], axis=0)
    table = np.einsum("kr,thr->thk", W, R)                       # [T, H, 1025]
    # the sinks are subtracted from the bias and the sink logit is 0
    sinks = x @ m.mat("blk.0.attn_sink_proj.weight").T           # [T, H]
    table = table - sinks[:, :, None]
    # relative distance index per (query t, key j): distance if 0 <= d < N_SWA else N_SWA
    d = np.arange(T)[:, None] - np.arange(T)[None, :]            # [t, j]
    relidx = np.where((d >= 0) & (d < N_SWA), d, N_SWA)
    causal = (d >= 0)
    bias = np.zeros((T, N_HEAD, T), np.float32)                  # [t, h, j]
    for t in range(T):
        bias[t] = table[t][:, relidx[t]]                         # [H, T]->[h, j]
    # full 256-cell view for the comparison with kq_b-0: empty cells use index N_SWA
    NKV = 256
    rel_full = np.full((T, NKV), N_SWA, dtype=np.int64)
    rel_full[:, :T] = relidx
    bias_full = np.stack([table[t][:, rel_full[t]] for t in range(T)], axis=0)  # [t, h, NKV]
    log("kq_b-0", np.transpose(bias_full, (1, 0, 2)))            # ne [NKV, T, H]
    scores = np.einsum("thd,jhd->thj", Q, np.repeat(K, N_HEAD // N_KV, axis=1)) / np.sqrt(HD)
    scores = scores + bias
    scores = np.where(causal[:, None, :], scores, -np.inf)
    p = softmax(np.concatenate([np.zeros((T, N_HEAD, 1), np.float32), scores], axis=-1), axis=-1)
    p = p[..., 1:]                                               # drop the zero sink logit
    Vr = np.repeat(V, N_HEAD // N_KV, axis=1)                    # [T, H, D]
    O = np.einsum("thj,jhd->thd", p, Vr)                         # [T, H, D]
    log("kqv_out-0", O.reshape(T, N_HEAD * HD))

    gate = x @ m.mat("blk.0.attn_gate.weight").T                 # [T, 3072]
    O = O.reshape(T, N_HEAD * HD) * (1.0 / (1.0 + np.exp(-gate)))
    out = O @ m.mat("blk.0.attn_output.weight").T                # [T, E]
    attn = conv_causal(out, m.tensor("blk.0.attn_post_conv.weight").reshape(N_EMBD, 4))

    # block start: no residual add, the block input goes to blocks
    ffn_inp = attn
    log("ffn_inp-0", ffn_inp)
    blocks = [inp_sa]

    # FFN input: mix(blocks, cur, ffn_res_query)
    states = np.stack([blocks[0], ffn_inp], axis=1)              # [T, S=2, E]
    states_n = rms_norm(states, EPS)
    q = m.tensor("blk.0.ffn_res_query.weight")
    sc = np.einsum("tse,e->ts", states_n, q) / np.sqrt(N_EMBD)
    ps = softmax(sc, axis=-1)
    mixed = np.einsum("ts,tse->te", ps, states_n)

    xf = rms_norm(mixed, EPS, w=m.tensor("blk.0.ffn_norm.weight"))
    log("norm-0#2", rms_norm(mixed, EPS))

    # MoE
    Wr = m.mat("blk.0.ffn_gate_inp.weight")                      # [128, E]
    logits = xf @ Wr.T                                           # [T, 128]
    log("ffn_moe_logits-0", logits)
    probs = 1.0 / (1.0 + np.exp(-logits))
    sel_p = probs + m.tensor("blk.0.exp_probs_b.bias").reshape(1, N_EXPERT)
    order = np.argsort(-sel_p, axis=1, kind="stable")[:, :N_USED]  # [T, 8]
    w = np.take_along_axis(probs, order, axis=1)
    w = w / (w.sum(axis=1, keepdims=True) + 1e-20)
    w = w * W_SCALE                                              # [T, 8]

    gu_w = m.tensor("blk.0.ffn_gate_up_exps.weight").reshape(N_EXPERT, 2 * 1152, N_EMBD)
    gu_s = m.tensor("blk.0.ffn_gate_up_exps.scale")
    dn_w = m.tensor("blk.0.ffn_down_exps.weight").reshape(N_EXPERT, N_EMBD, 1152)
    dn_s = m.tensor("blk.0.ffn_down_exps.scale")

    moe = np.zeros((T, N_EMBD), np.float32)
    for t in range(T):
        for k in range(N_USED):
            e = order[t, k]
            gu = (gu_w[e] @ xf[t]) * gu_s[e]                     # [2304]
            g, u = gu[:1152], gu[1152:]
            act = (ALPHA_SITU_BETA * np.tanh(g / ALPHA_SITU_BETA)) * (1.0 / (1.0 + np.exp(-g)))
            up = ALPHA_SITU_LB * np.tanh(u / ALPHA_SITU_LB)
            h = act * up
            dn = (dn_w[e] @ h) * dn_s[e]                         # [E]
            moe[t] += w[t, k] * dn
    log("ffn_moe_out-0", moe)
    ffn_out = conv_causal(moe, m.tensor("blk.0.ffn_post_conv.weight").reshape(N_EMBD, 4))
    log("ffn_out-0", ffn_out)
    l_out = ffn_inp + ffn_out
    log("l_out-0", l_out)
    return l_out


def compare(name, ours, ref_path, shape, report):
    ref = np.fromfile(ref_path, dtype="<f4").reshape(shape)
    ours = np.asarray(ours, np.float32).reshape(shape)
    err = np.max(np.abs(ours - ref))
    scale = np.max(np.abs(ref)) + 1e-12
    report.append((name, shape, err, err / scale))


def main():
    m = Model(MODEL)
    report = []
    values = {}

    def log(name, arr):
        values[name] = np.asarray(arr, np.float32)

    layer0(m, log)

    # compare against the llama.cpp dumps: shapes from index.tsv (ggml order reversed)
    index = {}
    with open(os.path.join(DUMP, "index.tsv")) as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            index.setdefault(p[0], []).append((p[1], [int(v) for v in p[2:6]]))
    for name, arr in values.items():
        # dump keys: "norm-0" (first occurrence), "norm-0#2" (second), ...
        key = name[:-2] if name.endswith("#1") else name
        if key not in index:
            print(f"no dump for {name}")
            continue
        typ, ne = index[key][0]
        ref = np.fromfile(os.path.join(DUMP, key + ".bin"), dtype="<f4" if typ == "f32" else "<i4")
        ours = arr.reshape(-1)
        if ours.size != ref.size:
            print(f"{name}: size mismatch ours={ours.size} ref={ref.size} (ne={ne})")
            continue
        ref = ref.reshape(tuple(reversed(ne)))
        ours = ours.reshape(ref.shape)
        err = float(np.max(np.abs(ours - ref)))
        rel = err / (float(np.max(np.abs(ref))) + 1e-12)
        report.append((name, str(tuple(ref.shape)), err, rel))
    print(f"{'tensor':24s} {'shape':22s} {'max abs err':>12s} {'rel err':>10s}")
    for name, shape, err, rel in report:
        print(f"{name:24s} {shape:22s} {err:12.4e} {rel:10.3e}")


if __name__ == "__main__":
    main()
