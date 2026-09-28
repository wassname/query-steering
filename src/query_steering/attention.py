"""Query steering on Qwen3 attention layers, at the last token only.

query steering: add a fixed vector to the last token's query, before RoPE; the head then reads this prompt's K, V as usual
    q_last += α · q*,   q* = mean(q_pos − q_neg) from contrast pairs
residual steering (baseline): h_last += α · r* at the input of each steered layer
"""
from dataclasses import dataclass, field

import re

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention, apply_rotary_pos_emb

@dataclass
class State:
    mode: str = "normal"  # normal | qsteer | rsteer | capture
    layers: set = field(default_factory=set)
    alpha: float = 0.0  # qsteer / rsteer
    start: int = -1  # qsteer: steer query positions start: (default: the last token only)
    q_star: dict = field(default_factory=dict)  # layer -> [H, d]
    r_star: dict = field(default_factory=dict)  # layer -> [D]
    q_cap: dict = field(default_factory=dict)  # capture: layer -> last-token query [H, d]
    h_cap: dict = field(default_factory=dict)  # capture: layer -> last-token residual [D]
    keep_attn: bool = False  # store the last token's attention weights, any mode
    a_cap: dict = field(default_factory=dict)  # layer -> [H, T]


S = State()


def attn_forward(self, hidden_states, position_embeddings, attention_mask, past_key_values=None, **kw):
    """Qwen3Attention.forward without KV cache (full recompute each step), plus query steering."""
    B, T, _ = hidden_states.shape
    on = self.layer_idx in S.layers
    hs = (B, T, -1, self.head_dim)
    q = self.q_norm(self.q_proj(hidden_states).view(hs)).transpose(1, 2)  # [B,H,T,d] before RoPE
    if S.mode == "capture" and on:
        S.q_cap[self.layer_idx] = q[0, :, -1].float()
    if S.mode == "qsteer" and on:
        q = q.clone()
        q[0, :, S.start:] += S.alpha * S.q_star[self.layer_idx].to(q.dtype)[:, None]
    k = self.k_norm(self.k_proj(hidden_states).view(hs)).transpose(1, 2)
    v = self.v_proj(hidden_states).view(hs).transpose(1, 2)
    cos, sin = position_embeddings
    q, k = apply_rotary_pos_emb(q, k, cos, sin)
    g = self.num_key_value_groups
    k, v = k.repeat_interleave(g, 1), v.repeat_interleave(g, 1)
    if S.keep_attn and on:
        S.a_cap[self.layer_idx] = (q[0, :, -1:] @ k[0].transpose(-1, -2) * self.scaling).softmax(-1)[:, 0].float()
    out = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=self.scaling)
    out = out.transpose(1, 2).reshape(B, T, -1)
    return self.o_proj(out), None


def _resid_hook(layer_idx):
    def hook(module, args, kwargs):
        if layer_idx not in S.layers or S.mode not in ("capture", "rsteer"):
            return None
        h = args[0]  # decoder layers get hidden_states positionally
        if S.mode == "capture":
            S.h_cap[layer_idx] = h[0, -1].float()
            return None
        h = h.clone()
        h[0, -1] += S.alpha * S.r_star[layer_idx].to(h.dtype)
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
def generate(tok, model, text, n=40, stop_ids=()):
    """greedy; full recompute each step, so the intervention hits every new last token; stops after a stop_ids token"""
    ids = tok(text, return_tensors="pt").input_ids.to(model.device)
    n0 = ids.shape[1]
    for _ in range(n):
        ids = torch.cat([ids, model(ids).logits[0, -1].argmax().view(1, 1)], 1)
        if ids[0, -1].item() in stop_ids:
            break
    return tok.decode(ids[0, n0:])


def loop_share(g):
    """share of repeated word 4-grams; a looping answer ("flag received. flag received. …") is near 1"""
    w = re.findall(r"\w+|[^\w\s]", g)
    grams = [tuple(w[i:i + 4]) for i in range(len(w) - 3)]
    return 1 - len(set(grams)) / len(grams) if grams else 0.0


@torch.no_grad()
def nll(tok, model, text, g):
    """median -log p per answer token under the unsteered model; fluent answers are low, word salad high (median: one surprising token, e.g. the steered first word, does not count)"""
    mode, S.mode = S.mode, "normal"
    n0 = tok(text, return_tensors="pt").input_ids.shape[1]
    ids = tok(text + g, return_tensors="pt").input_ids.to(model.device)
    lp = model(ids).logits[0, n0 - 1:-1].float().log_softmax(-1)
    S.mode = mode
    return -lp.gather(-1, ids[0, n0:, None]).median().item()


FAIL_NLL, FAIL_LOOP = 1.0, 0.4  # TODO check against the answers in outputs/04_walk3_*.log


def broken(tok, model, text, g):
    """word salad (high NLL) or a loop (a loop can have low NLL: the model copies itself)"""
    return nll(tok, model, text, g) > FAIL_NLL or loop_share(g) > FAIL_LOOP


def calibrate(tok, model, text, q_star, n_gen=100, stop_ids=(), log=print):
    """α = 2/3 of the dose where the answer breaks. Doubles α from 1/8 on one prompt; judged on fluency only."""
    S.q_star, S.mode = q_star, "qsteer"
    fails, a_fail = 0, None
    for a in [2.0 ** k for k in range(-3, 11)]:
        S.alpha = a
        g = generate(tok, model, text, n_gen, stop_ids)
        bad = broken(tok, model, text, g)
        log(f"α={a:g} broken={bad} NLL={nll(tok, model, text, g):.2f} loops={loop_share(g):.2f} | {g[:90]!r}")
        a_fail = a_fail or (a if bad else None)
        fails = fails + 1 if bad else 0
        if fails == 2:  # two in a row: one broken answer can be chance
            break
    S.mode = "normal"
    assert a_fail, "never broke up to α=1024"
    return 2 / 3 * a_fail


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


def parse_layers(spec, model):
    """"all" | "late" (second half) | "19,23" -> layer indices"""
    n = model.config.num_hidden_layers
    return {"all": list(range(n)), "late": list(range(n // 2, n))}.get(spec) or [int(x) for x in spec.split(",")]
