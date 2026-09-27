"""Query steering on the softmax-attention layers of Qwen3.5 (8 of 32) or Qwen3 (all), at the last token (or every position with S.all_pos).

query steering: add a fixed vector to the last token's query, before RoPE; the head then reads this prompt's K, V as usual
    q_last += α · q*,   q* = mean(q_pos − q_neg) from contrast pairs
residual steering (baseline): h_last += α · r* at the input of each steered layer
"""
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Attention, apply_rotary_pos_emb

@dataclass
class State:
    mode: str = "normal"  # normal | qsteer | rsteer | capture
    layers: set = field(default_factory=set)
    alpha: float = 0.0  # qsteer / rsteer
    q_star: dict = field(default_factory=dict)  # layer -> [H, d]
    r_star: dict = field(default_factory=dict)  # layer -> [D]
    q_cap: dict = field(default_factory=dict)  # capture: layer -> last-token query [H, d]
    h_cap: dict = field(default_factory=dict)  # capture: layer -> last-token residual [D]
    all_pos: bool = False  # steer every position, not only the last token
    record_attn: bool = False  # store the last token's attention probs per layer in attn_cap
    attn_cap: dict = field(default_factory=dict)  # layer -> [H, T]
    q_delta: dict = field(default_factory=dict)  # qgrad: layer -> zero leaf [1,H,1,d] added to every query position


S = State()


def attn_forward(self, hidden_states, position_embeddings, attention_mask, past_key_values=None, **kw):
    """Qwen3_5Attention.forward without KV cache (full recompute each step), plus query steering."""
    B, T, _ = hidden_states.shape
    on = self.layer_idx in S.layers
    hs = (B, T, -1, self.head_dim)
    qg = self.q_proj(hidden_states)
    gated = qg.shape[-1] == 2 * self.config.num_attention_heads * self.head_dim  # Qwen3.5 has an output gate, Qwen3 not
    q, gate = torch.chunk(qg.view(B, T, -1, self.head_dim * 2), 2, dim=-1) if gated else (qg, None)
    q = self.q_norm(q.reshape(hs)).transpose(1, 2)  # [B,H,T,d] before RoPE
    if S.mode == "capture" and on:
        S.q_cap[self.layer_idx] = q[0, :, -1].float()
    if S.mode == "qgrad" and on:
        q = q + S.q_delta[self.layer_idx].to(q.dtype)
    if S.mode == "qsteer" and on:
        q = q.clone()
        pos = slice(None) if S.all_pos else slice(-1, None)
        q[0, :, pos] += S.alpha * S.q_star[self.layer_idx].to(q.dtype)[:, None]  # [H,1,d] broadcast over positions
    k = self.k_norm(self.k_proj(hidden_states).view(hs)).transpose(1, 2)
    v = self.v_proj(hidden_states).view(hs).transpose(1, 2)
    cos, sin = position_embeddings
    q, k = apply_rotary_pos_emb(q, k, cos, sin)
    g = self.num_key_value_groups
    k, v = k.repeat_interleave(g, 1), v.repeat_interleave(g, 1)
    out = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=self.scaling)
    if S.record_attn and on:
        S.attn_cap[self.layer_idx] = (q[0, :, -1:] @ k[0].transpose(-1, -2) * self.scaling).float().softmax(-1)[:, 0]
    out = out.transpose(1, 2).reshape(B, T, -1)
    if gated:
        out = out * torch.sigmoid(gate.reshape(B, T, -1))
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
        h[0, slice(None) if S.all_pos else -1] += S.alpha * S.r_star[layer_idx].to(h.dtype)
        return (h, *args[1:]), kwargs
    return hook


def load(model_name="Qwen/Qwen3.5-4B", device="cuda"):
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16, device_map=device).eval()
    cfg = getattr(model.config, "text_config", model.config)
    full = [i for i, t in enumerate(cfg.layer_types) if t == "full_attention"]
    ids = tok("The quick brown fox jumps over the lazy dog because", return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        ref = model(ids).logits.float()
        Qwen3_5Attention.forward = Qwen3Attention.forward = attn_forward
        err = (model(ids).logits.float() - ref).abs().max().item()
    assert err < 1.0, f"patched forward differs from original: max|Δlogit| {err}"
    decoder = model.model.language_model.layers if hasattr(model.model, "language_model") else model.model.layers
    for i in full:
        decoder[i].register_forward_pre_hook(_resid_hook(i), with_kwargs=True)
    return tok, model, full


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


def extract(tok, model, pairs, layers):
    """diff of means at the last token over (pos_text, neg_text) pairs -> q* per layer [H,d], r* per layer [D]"""
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


def extract_qvjp(tok, model, pairs, layers, q_ref):
    """Query directions chosen by output effect (VJP), per layer [H,d], rescaled to |q_ref[L]| (so only direction differs).

    c = mean(h_pos) − mean(h_neg) at the last token of the last decoder layer
    g(x)_L = ∂⟨c, h_last(x)⟩/∂δ_L, δ_L a query shift shared by every position (the steering we apply with all_pos)
    qvjp_mean = mean over all prompts of g;  qvjp_delta = mean_pos g − mean_neg g (vjp-steering's vjp_delta estimator)
    """
    decoder = model.model.language_model.layers if hasattr(model.model, "language_model") else model.model.layers
    found = {}
    h = decoder[-1].register_forward_hook(lambda m, a, out: found.__setitem__("h", out[0] if isinstance(out, tuple) else out))
    model.requires_grad_(False)

    def run(text, grad):
        ids = tok(text, return_tensors="pt").input_ids.to(model.device)
        if not grad:
            with torch.no_grad():
                model(ids)
            return found["h"][0, -1].float()
        H, d = model.config.get_text_config().num_attention_heads, q_ref[layers[0]].shape[-1]
        S.q_delta = {L: torch.zeros(1, H, 1, d, device=model.device, dtype=torch.float32, requires_grad=True) for L in layers}
        S.mode, S.layers = "qgrad", set(layers)
        with torch.enable_grad():
            model(ids)
            (found["h"][0, -1].float() @ c).backward()
        S.mode = "normal"
        return {L: S.q_delta[L].grad[0, :, 0].clone() for L in layers}

    try:
        c = torch.stack([run(p, False) for p, _ in pairs]).mean(0) - torch.stack([run(n, False) for _, n in pairs]).mean(0)
        gp = [run(p, True) for p, _ in pairs]
        gn = [run(n, True) for _, n in pairs]
    finally:
        h.remove()
    out = {}
    for name, v in (("qvjp_mean", {L: torch.stack([g[L] for g in gp + gn]).mean(0) for L in layers}),
                    ("qvjp_delta", {L: torch.stack([g[L] for g in gp]).mean(0) - torch.stack([g[L] for g in gn]).mean(0) for L in layers})):
        out[name] = {L: v[L] / v[L].norm() * q_ref[L].norm() for L in layers}
    return out
