"""Draw attention maps (token highlighting) from the JSON written by scripts/05_attention_map.py, as PNG and as one HTML page.

uv run python -m query_steering.render outputs/05_attention_map.json   # re-render without the GPU
"""
import html
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, Normalize, TwoSlopeNorm, to_hex

# capped colour ranges: the strongest red is still light enough for black text
CMAPS = {"diverging": ListedColormap(plt.get_cmap("RdBu_r")(np.linspace(0.25, 0.75, 256))),
         "sequential": ListedColormap(plt.get_cmap("Reds")(np.linspace(0.0, 0.4, 256)))}


def norm(sec):
    m = sec["vmax"]
    return TwoSlopeNorm(0, -m, m) if sec["kind"] == "diverging" else Normalize(0, m, clip=True)


def colour(sec, v):
    return to_hex(CMAPS[sec["kind"]](norm(sec)(v)))


def png(sections, path, width=11.0):
    """one box per token, wrapped like text, on one baseline; red underline where sec["under"]"""
    fig = plt.figure(figsize=(width, 1))
    r = fig.canvas.get_renderer()
    W, line_h, y, items, bars = width * fig.dpi, 17.0, 0.0, [], []
    for sec in sections:
        bars.append((y, sec))
        items.append((0, y, sec["heading"], "none", True, 0, False))
        y += 42.0
        x = 0.0
        for s, v, ul in zip(sec["pieces"], sec["values"], sec["under"]):
            for j, part in enumerate(s.split("\n")):
                if j:
                    x, y = 0.0, y + line_h
                if not part:
                    continue
                t = fig.text(0, 0, part, fontsize=9, family="DejaVu Sans")
                w = t.get_window_extent(r).width
                t.remove()
                if x + w > W and x > 0:
                    x, y = 0.0, y + line_h
                items.append((x, y, part, colour(sec, v), False, w, ul))
                x += w
        y += 2 * line_h
    H = y + 8
    fig.set_size_inches(width, H / fig.dpi)
    for x, y, part, c, bold, w, ul in items:  # y = top of the line
        base = 1 - (y + 4 + 12) / H
        if bold:
            fig.text(0, base, part, fontsize=10, weight="bold", va="baseline")
            continue
        fig.add_artist(plt.Rectangle((x / W, 1 - (y + 4 + line_h - 1) / H), w / W, (line_h - 1) / H, fc=c, ec="none", transform=fig.transFigure))
        fig.text(x / W, base, part, fontsize=9, family="DejaVu Sans", va="baseline", ha="left")
        if ul:
            yl = base - 5 / H  # below descenders
            fig.add_artist(plt.Line2D([x / W, (x + w) / W], [yl, yl], color="red", lw=2.2, transform=fig.transFigure))
    for y, sec in bars:
        cax = fig.add_axes([0.78, 1 - (y + 14) / H, 0.2, 6 / H])
        cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm(sec), cmap=CMAPS[sec["kind"]]), cax=cax, orientation="horizontal")
        m = sec["vmax"]
        ticks = [-m, 0, m] if sec["kind"] == "diverging" else [0, m]
        cb.set_ticks(ticks, labels=[f"{t:+.3g}" if t else "0" for t in ticks])
        cb.ax.tick_params(labelsize=7, pad=1)
        cb.set_label(sec["label"], fontsize=7, labelpad=1)
    fig.savefig(path, dpi=fig.dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def legend_html(sec):
    stops = np.linspace(-1, 1, 9) if sec["kind"] == "diverging" else np.linspace(0, 1, 9)
    grad = ", ".join(colour(sec, s * sec["vmax"]) for s in stops)
    lo = f"{-sec['vmax']:+.3g}" if sec["kind"] == "diverging" else "0"
    return (f'<span class="legend"><span class="bar" style="background: linear-gradient(to right, {grad})"></span>'
            f'<span class="ticks"><span>{lo}</span><span>{sec["vmax"]:+.3g}</span></span>{html.escape(sec["label"])}</span>')


def section_html(sec):
    spans = []
    for s, v, ul in zip(sec["pieces"], sec["values"], sec["under"]):
        cls = ' class="fact"' if ul else ""
        spans.append(f'<span{cls} style="background:{colour(sec, v)}" title="{v:+.4f}">{html.escape(s)}</span>')
    return f'<div class="sec"><div class="head"><b>{html.escape(sec["heading"])}</b>{legend_html(sec)}</div><div class="text">{"".join(spans)}</div></div>'


PAGE = """<!doctype html><meta charset="utf-8"><title>Query steering: attention maps</title>
<style>
body {{ font: 15px/1.5 system-ui, sans-serif; max-width: 1000px; margin: 2em auto; padding: 0 1em; color: #111; }}
h2 {{ margin-top: 2.5em; }}
.sec {{ margin: 1.2em 0; }}
.head {{ display: flex; justify-content: space-between; align-items: flex-end; gap: 1em; margin-bottom: .3em; }}
.text {{ white-space: pre-wrap; font: 14px/1.9 ui-monospace, monospace; border: 1px solid #ddd; padding: .6em; }}
.fact {{ text-decoration: underline 2.5px red; text-underline-offset: 4px; }}
.legend {{ font-size: 11px; color: #555; text-align: center; white-space: nowrap; }}
.bar {{ display: block; width: 180px; height: 8px; border: 1px solid #aaa; }}
.ticks {{ display: flex; justify-content: space-between; }}
</style>
<h1>Query steering: attention maps</h1>
<p>From <a href="https://github.com/wassname/query-steering">github.com/wassname/query-steering</a>. Each demo shows the transcript and two answers.
Transcript: each token's colour is how much more (red) or less (blue) the model looked at it with steering than without, while writing the steered answer.
Answers: each token's colour is how much the model looked at the hidden fact (underlined in red) while writing that word; both answers use the same scale.
Hover a token for its value. How these are computed: <a href="https://github.com/wassname/query-steering#attention-maps">README, Attention maps</a>.</p>
{body}
"""


def main(json_path, img_dir="docs/img", html_path="docs/index.html"):
    demos = json.loads(Path(json_path).read_text())
    body = []
    for name, demo in demos.items():
        png(demo["sections"], Path(img_dir) / f"attn_{name}.png")
        body.append(f'<h2>{html.escape(demo["title"])}</h2>' + "".join(section_html(s) for s in demo["sections"]))
    Path(html_path).write_text(PAGE.format(body="\n".join(body)))


if __name__ == "__main__":
    main(*sys.argv[1:])
