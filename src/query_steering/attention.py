"""Query steering on Qwen3 attention layers, at the last token only.

query steering: add a fixed vector to the last token's query, before RoPE; the head then reads this prompt's K, V as usual
    q_last += α · q*,   q* = mean(q_pos − q_neg) from contrast pairs
residual steering (baseline): h_last += α · r* at the input of each steered layer
"""
from dataclasses import dataclass, field

import json
import re
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention, apply_rotary_pos_emb

@dataclass
class State:
    mode: str = "normal"  # normal | qsteer | rsteer | both (qsteer at alpha + rsteer at r_alpha) | capture
    layers: set = field(default_factory=set)
    alpha: float = 0.0  # qsteer / rsteer
    r_alpha: float = 0.0  # mode "both": the residual dose
    start: int = -1  # qsteer: steer query positions start: (default: the last token only)
    q_star: dict = field(default_factory=dict)  # layer -> [H, d]
    r_star: dict = field(default_factory=dict)  # layer -> [D]
    q_cap: dict = field(default_factory=dict)  # capture: layer -> last-token query [H, d]
    h_cap: dict = field(default_factory=dict)  # capture: layer -> last-token residual [D]
    keep_attn: bool = False  # store the last token's attention weights, any mode
    a_cap: dict = field(default_factory=dict)  # layer -> [H, T]
    kv: dict = field(default_factory=dict)  # generate(): layer -> (k, v) of the prompt without its last token, never steered
    kv_record: bool = False


S = State()


def attn_forward(self, hidden_states, position_embeddings, attention_mask, past_key_values=None, **kw):
    """Qwen3Attention.forward plus query steering. No HF KV cache; generate() keeps the unsteered prompt prefix in S.kv."""
    B, T, _ = hidden_states.shape
    on = self.layer_idx in S.layers
    hs = (B, T, -1, self.head_dim)
    q = self.q_norm(self.q_proj(hidden_states).view(hs)).transpose(1, 2)  # [B,H,T,d] before RoPE
    if S.mode == "capture" and on:
        S.q_cap[self.layer_idx] = q[0, :, -1].float()
    if S.mode in ("qsteer", "both") and on:
        q = q.clone()
        q[0, :, S.start:] += S.alpha * S.q_star[self.layer_idx].to(q.dtype)[:, None]
    k = self.k_norm(self.k_proj(hidden_states).view(hs)).transpose(1, 2)
    v = self.v_proj(hidden_states).view(hs).transpose(1, 2)
    cos, sin = position_embeddings
    q, k = apply_rotary_pos_emb(q, k, cos, sin)
    mask, causal = None, True
    if S.kv_record:
        S.kv[self.layer_idx] = (k, v)
    elif S.kv:  # queries are positions n_c.. of the sequence; key j is visible to query i if j <= n_c + i
        k0, v0 = S.kv[self.layer_idx]
        n_c = k0.shape[2]
        k, v = torch.cat([k0, k], 2), torch.cat([v0, v], 2)
        mask = torch.arange(n_c + T, device=q.device)[None] <= torch.arange(T, device=q.device)[:, None] + n_c
        causal = False
    g = self.num_key_value_groups
    k, v = k.repeat_interleave(g, 1), v.repeat_interleave(g, 1)
    if S.keep_attn and on:
        S.a_cap[self.layer_idx] = (q[0, :, -1:] @ k[0].transpose(-1, -2) * self.scaling).softmax(-1)[:, 0].float()
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=causal, scale=self.scaling)
    out = out.transpose(1, 2).reshape(B, T, -1)
    return self.o_proj(out), None


def _resid_hook(layer_idx):
    def hook(module, args, kwargs):
        if layer_idx not in S.layers or S.mode not in ("capture", "rsteer", "both"):
            return None
        h = args[0]  # decoder layers get hidden_states positionally
        if S.mode == "capture":
            S.h_cap[layer_idx] = h[0, -1].float()
            return None
        h = h.clone()
        h[0, -1] += (S.r_alpha if S.mode == "both" else S.alpha) * S.r_star[layer_idx].to(h.dtype)
        return (h, *args[1:]), kwargs
    return hook


def load(model_name="Qwen/Qwen3-4B", device="cuda"):
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16, device_map=device).eval()
    ids = tok("The quick brown fox jumps over the lazy dog because", return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        ref = model(ids).logits.float()
        Qwen3Attention.forward = attn_forward
        err = (model(ids).logits.float() - ref).abs().max().item()
    assert err < 1.0, f"patched forward differs from original: max|Δlogit| {err}"
    for i, layer in enumerate(model.model.layers):
        layer.register_forward_pre_hook(_resid_hook(i), with_kwargs=True)
    return tok, model


@torch.no_grad()
def last_logprobs(tok, model, text):
    ids = tok(text, return_tensors="pt").input_ids.to(model.device)
    return model(ids).logits[0, -1].float().log_softmax(-1)


@torch.no_grad()
def generate(tok, model, text, n=40, stop_ids=(), cache=True):
    """greedy; each step recomputes the reply so the intervention hits only the newest token; stops after a stop_ids token.
    The prompt without its last token is never steered, so its keys and values are computed once (S.kv); cache=False recomputes all."""
    ids = tok(text, return_tensors="pt").input_ids.to(model.device)
    n0 = ids.shape[1]
    n_c = n0 - 1 if cache else 0
    if n_c:
        mode, S.mode, S.kv_record = S.mode, "normal", True
        model(ids[:, :n_c])
        S.mode, S.kv_record = mode, False
    try:
        for _ in range(n):
            pos = torch.arange(n_c, ids.shape[1], device=ids.device)[None]
            ids = torch.cat([ids, model(ids[:, n_c:], position_ids=pos).logits[0, -1].argmax().view(1, 1)], 1)
            if ids[0, -1].item() in stop_ids:
                break
    finally:
        S.kv = {}
    return tok.decode(ids[0, n0:])


def repetition(g):
    """1 - distinct word trigrams / trigrams; fluent text is near 0, a loop near 1"""
    w = re.findall(r"\w+|[^\w\s]", g)
    grams = [tuple(w[i:i + 3]) for i in range(len(w) - 2)]
    return 1 - len(set(grams)) / len(grams) if grams else 0.0


def finished(g):
    return g.rstrip().endswith(("<|im_end|>", "<|endoftext|>"))


def json_ok(g):
    m = re.search(r"\{.*\}", g, re.S)
    try:
        return m is not None and isinstance(json.loads(m.group()), dict)
    except json.JSONDecodeError:
        return False


def broken(g, g0):
    """broken vs the unsteered answer g0 to the same prompt: did not finish within the token budget, or repetition up 50%"""
    return not finished(g) or repetition(g) > 1.5 * repetition(g0) + 0.05  # +0.05: a base of 0 would flag any repeat


JSON_PROBE = 'Reply with only this JSON, filled in: {"done": true or false, "note": "<one short sentence>"}'


def calibrate(tok, model, msgs, q_star, n_gen=200, stop_ids=(), log=print):
    """α = 2/3 of the first dose where the answer breaks (steering-concepts: dose_walk_calibration).
    One chat (msgs), two probes: the chat itself (finishes in n_gen tokens, repetition up < 50%) and the chat with the
    last user message replaced by JSON_PROBE (parseable JSON). α rises in third-octave steps from 1/8; stop at 2 broken in a row."""
    chat = lambda m: tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    task, probe = chat(msgs), chat(msgs[:-1] + [{"role": "user", "content": JSON_PROBE}])
    S.mode = "normal"
    g0, j0 = generate(tok, model, task, n_gen, stop_ids), generate(tok, model, probe, 60, stop_ids)
    assert finished(g0) and json_ok(j0), f"unsteered probe already fails, raise n_gen: {g0[-80:]!r} {j0!r}"
    S.q_star, S.mode = q_star, "qsteer"
    fails, a_fail = 0, None
    for a in [2 ** (k / 3) / 8 for k in range(40)]:  # 1/8 .. ~800
        S.alpha = a
        g, j = generate(tok, model, task, n_gen, stop_ids), generate(tok, model, probe, 60, stop_ids)
        bad = broken(g, g0) or not json_ok(j)
        log(f"α={a:.3g} broken={bad} finished={finished(g)} rep={repetition(g):.2f} (base {repetition(g0):.2f}) json={json_ok(j)} | {g[:80]!r}")
        fails = fails + 1 if bad else 0  # one broken rung then a healthy one is not the boundary
        a_fail = (a_fail if fails > 1 else a) if bad else None
        if fails == 2:
            break
    S.mode = "normal"
    assert a_fail, "never broke"
    return 2 / 3 * a_fail


def orthonormalise(vecs, eps=1e-6):
    """Per layer and head, make the concept directions orthogonal without choosing an order (V (VᵀV)^-1/2, symmetric
    Gram-Schmidt); each keeps its own per-head norm. vecs: name -> {layer: [H, d]}. steering-concepts: multi_vector_steering."""
    names, layers = list(vecs), list(next(iter(vecs.values())))
    out = {k: {} for k in names}
    for L in layers:
        V = torch.stack([vecs[k][L] for k in names]).float().transpose(0, 1)  # [H, k, d]
        G = V @ V.transpose(1, 2) + eps * torch.eye(len(names), device=V.device)
        e, U = torch.linalg.eigh(G)
        Q = U @ torch.diag_embed(e.clamp_min(eps) ** -0.5) @ U.transpose(1, 2) @ V  # [H, k, d], unit rows
        Q = Q * V.norm(dim=-1, keepdim=True)
        for i, k in enumerate(names):
            out[k][L] = Q[:, i].to(vecs[k][L].dtype)
    return out


def extract(tok, model, pairs, layers):
    """diff of means at the last token over (pos_text, neg_text) pairs -> q* per layer [H,d], r* per layer [D]"""
    last = lambda t: tok(t).input_ids[-1]
    bad = [(a[-30:], b[-30:]) for a, b in pairs if last(a) != last(b)]
    assert not bad, f"pos and neg must end on the same token (space included), else q* encodes the token: {bad[:2]}"
    S.mode, S.layers = "capture", set(layers)
    dq, dr = {L: [] for L in layers}, {L: [] for L in layers}
    for pos, neg in pairs:
        last_logprobs(tok, model, pos)
        qp, hp = dict(S.q_cap), dict(S.h_cap)
        last_logprobs(tok, model, neg)
        for L in layers:
            dq[L].append(qp[L] - S.q_cap[L])
            dr[L].append(hp[L] - S.h_cap[L])
    S.mode = "normal"
    return {L: torch.stack(dq[L]).mean(0) for L in layers}, {L: torch.stack(dr[L]).mean(0) for L in layers}


def load_vector(repo="wassname/query-steering", model_dir="qwen3-4b", name="super_q", device="cpu"):
    """-> ({layer: [H, d]}, {demo: calibrated alpha}) from a Hugging Face repo (or a local dir with the same layout)"""
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    get = lambda f: f"{repo}/{model_dir}/{name}/{f}" if Path(repo).is_dir() else hf_hub_download(repo, f"{model_dir}/{name}/{f}")
    t = load_file(get(f"{name}.safetensors"))
    return {int(k.split(".")[1]): v.to(device) for k, v in t.items()}, json.loads(Path(get("config.json")).read_text())["alpha"]


def parse_layers(spec, model):
    """"all" | "late" (second half) | "19,23" -> layer indices"""
    n = model.config.num_hidden_layers
    return {"all": list(range(n)), "late": list(range(n // 2, n))}.get(spec) or [int(x) for x in spec.split(",")]
