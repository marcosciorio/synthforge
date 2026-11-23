#!/usr/bin/env python3
"""Design of the space dimension of the captioner (reverb, delay, chorus)
from the distribution of the effect parameters on the reference presets.
"""

import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy import stats as sps

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
JSON_DIR = ROOT / "data" / "Dataset_TAL_48kHz" / "not_in_pool" / "json"
EVAL_DIR = Path(__file__).resolve().parent
PLOT_DIR = EVAL_DIR / "space_plots"
REPORT_PATH = EVAL_DIR / "space_dimension_analysis_report.md"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

SPACE_PARAMS = [
    "reverbwet", "reverbdecay", "reverbpredelay", "reverbhighcut",
    "reverblowcut",
    "delaywet", "delaytime", "delaysync", "delayfeedback",
    "delayhighshelf", "delaylowshelf", "delayx2l", "delayx2r",
    "chorus1enable", "chorus2enable",
]


def space_from_params(p: Dict) -> Optional[str]:
    rev_wet = p.get("reverbwet", 0); rev_decay = p.get("reverbdecay", 0)
    dly_wet = p.get("delaywet", 0); dly_time = p.get("delaytime", 0)
    dly_sync = p.get("delaysync", 0)
    chorus_on = (p.get("chorus1enable", 0) > 0.5 or
                 p.get("chorus2enable", 0) > 0.5)

    cavernous = rev_wet > 0.4 and rev_decay > 0.65
    echoey = dly_wet > 0.25 and (dly_sync > 0.5 or dly_time > 0.05)
    spacious = rev_wet > 0.25
    dry = rev_wet <= 0.05 and dly_wet <= 0.05 and not chorus_on

    if cavernous:  return "cavernous"
    if echoey:     return "echoey"
    if spacious:   return "spacious"
    if chorus_on:  return "chorused"
    if dry:        return "dry"
    return None


def space_attempt1(p):
    rev_wet = p.get("reverbwet", 0); rev_decay = p.get("reverbdecay", 0)
    dly_wet = p.get("delaywet", 0); dly_fb = p.get("delayfeedback", 0)
    chorus_on = (p.get("chorus1enable", 0) > 0.5 or
                 p.get("chorus2enable", 0) > 0.5)
    if rev_wet > 0.4 and rev_decay > 0.65:      return "cavernous"
    if dly_wet > 0.25 and dly_fb > 0.3:         return "echoey"
    if rev_wet > 0.25:                          return "spacious"
    if chorus_on:                               return "chorused"
    if rev_wet <= 0.05 and dly_wet <= 0.05:     return "dry"
    return None


def make_attempt2(dw_echo):
    def rule(p):
        rev_wet = p.get("reverbwet", 0); rev_decay = p.get("reverbdecay", 0)
        dly_wet = p.get("delaywet", 0)
        chorus_on = (p.get("chorus1enable", 0) > 0.5 or
                     p.get("chorus2enable", 0) > 0.5)
        if rev_wet > 0.4 and rev_decay > 0.65:  return "cavernous"
        if dly_wet > dw_echo:                   return "echoey"
        if rev_wet > 0.25:                      return "spacious"
        if chorus_on:                           return "chorused"
        if rev_wet <= 0.05 and dly_wet <= 0.05: return "dry"
        return None
    return rule


ATTEMPTS = [
    ("Attempt 1 — echoey = wet>0.25 and feedback>0.3", space_attempt1),
    ("Attempt 2a — echoey = wet>0.20 (no feedback)", make_attempt2(0.20)),
    ("Attempt 2b — echoey = wet>0.25 (no feedback)", make_attempt2(0.25)),
    ("Attempt 2c — echoey = wet>0.30 (no feedback)", make_attempt2(0.30)),
    ("Final — 2b with anti-comb guard (sync or time>0.05)", space_from_params),
]

LABEL_ORDER = ["dry", "cavernous", "spacious", "echoey", "chorused", "none"]

_LFO_DEST = {0: "osc1+2", 1: "osc1", 2: "osc2", 3: "pw1+2",
             4: "fm", 5: "cutoff", 6: "volume", 7: "pan"}


def _lfo_dest(val: float) -> str:
    return _LFO_DEST[min(int(round(val * 7)), 7)]


def is_pulsing(p):
    return (
        (p.get("lfo1amount", 0) > 0.3 and
         _lfo_dest(p.get("lfo1destination", 0)) == "volume" and
         p.get("lfo1rate", 0) > 0.4) or
        (p.get("lfo2amount", 0) > 0.3 and
         _lfo_dest(p.get("lfo2destination", 0)) == "volume" and
         p.get("lfo2rate", 0) > 0.4) or
        (p.get("delaysync", 0) > 0.5 and p.get("delayfeedback", 0) > 0.35 and
         p.get("delaywet", 0) > 0.1)
    )


def load_dataset():
    rows = []
    for jf in sorted(JSON_DIR.glob("*.json")):
        params = json.load(open(jf))["parameters"]
        row = {"name": jf.stem, "pulsing": is_pulsing(params),
               "_params": params}
        for k in SPACE_PARAMS:
            row[k] = params.get(k, 0.0)
        rows.append(row)
    return pd.DataFrame(rows)


SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

LABEL_COLORS = {
    "dry":       "#eda100",
    "cavernous": "#4a3aa7",
    "spacious":  "#2a78d6",
    "echoey":    "#1baf7a",
    "chorused":  "#008300",
    "none":      "#898781",
}

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.family": "sans-serif",
    "font.size": 9,
    "axes.edgecolor": BASELINE,
    "axes.labelcolor": INK2,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "text.color": INK,
})


def _style_ax(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_linewidth(0.8)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def hist_with_thresholds(vals, param, thresholds, out_path, bins=40):
    fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=150)
    _style_ax(ax)
    ax.hist(vals, bins=bins, range=(0, 1), color="#2a78d6", alpha=0.55,
            edgecolor=SURFACE, linewidth=0.5)
    for thr, note in thresholds:
        ax.axvline(thr, color=INK2, linewidth=1.0)
        ax.annotate(f"{thr:g} · {note}", xy=(thr, 1.0),
                    xycoords=("data", "axes fraction"),
                    xytext=(3, -2), textcoords="offset points",
                    va="top", ha="left", fontsize=7.5, color=INK2,
                    rotation=90)
    ax.set_xlim(-0.02, 1.02)
    ax.set_xlabel(param, fontsize=9)
    ax.set_ylabel("n preset", fontsize=9)
    ax.set_title(f"{param} — distribution on the corpus (n={len(vals)})",
                 fontsize=10, color=INK, loc="left", pad=10)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def violin_by_label(df, labels, param, thresholds, out_path):
    fig, ax = plt.subplots(figsize=(8.2, 4.2), dpi=150)
    _style_ax(ax)
    rng = np.random.default_rng(42)
    lab_series = pd.Series([l or "none" for l in labels], index=df.index)
    positions, tick_labels = [], []
    for i, lab in enumerate(LABEL_ORDER):
        vals = df.loc[lab_series == lab, param].values
        positions.append(i)
        tick_labels.append(f"{lab}\n(n={len(vals)})")
        if len(vals) == 0:
            continue
        col = LABEL_COLORS[lab]
        if len(vals) >= 8 and np.ptp(vals) > 1e-9:
            vp = ax.violinplot([vals], positions=[i], widths=0.78,
                               showextrema=False)
            for body in vp["bodies"]:
                body.set_facecolor(col)
                body.set_alpha(0.40)
                body.set_edgecolor(col)
                body.set_linewidth(1.2)
        jitter = rng.uniform(-0.10, 0.10, size=len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals, s=7, color=col,
                   alpha=0.55, linewidths=0, zorder=3)
        med = float(np.median(vals))
        ax.hlines(med, i - 0.22, i + 0.22, color=INK, linewidth=1.6,
                  zorder=4)
    for thr, note in thresholds:
        ax.axhline(thr, color=INK2, linewidth=1.0, zorder=2)
        ax.annotate(f"{thr:g} · {note}", xy=(1.0, thr),
                    xycoords=("axes fraction", "data"),
                    xytext=(4, 0), textcoords="offset points",
                    va="center", ha="left", fontsize=7.5, color=INK2)
    ax.set_xticks(positions)
    ax.set_xticklabels(tick_labels, fontsize=8)
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel(param, fontsize=9)
    ax.set_title(f"{param} by proposed space label", fontsize=10,
                 color=INK, loc="left", pad=10)
    fig.subplots_adjust(right=0.84, left=0.08, bottom=0.13, top=0.90)
    fig.savefig(out_path)
    plt.close(fig)


def counts_bar(counts, n, title, out_path):
    fig, ax = plt.subplots(figsize=(6.4, 3.4), dpi=150)
    _style_ax(ax)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.grid(axis="y", visible=False)
    y = np.arange(len(LABEL_ORDER))[::-1]
    for yi, lab in zip(y, LABEL_ORDER):
        c = counts.get(lab, 0)
        ax.barh(yi, c, height=0.55, color=LABEL_COLORS[lab])
        ax.annotate(f"{c}  ({100*c/n:.1f}%)", xy=(c, yi), xytext=(5, 0),
                    textcoords="offset points", va="center",
                    fontsize=8, color=INK2)
    ax.set_yticks(y)
    ax.set_yticklabels(LABEL_ORDER, fontsize=9)
    ax.set_xlim(0, max(counts.values()) * 1.25)
    ax.set_title(title, fontsize=10, loc="left", color=INK, pad=10)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def iteration_panel(all_counts, n, out_path):
    k = len(all_counts)
    fig, axes = plt.subplots(1, k, figsize=(3.0 * k, 3.2), dpi=150,
                             sharey=True)
    for ax, (name, counts) in zip(np.atleast_1d(axes), all_counts):
        _style_ax(ax)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.grid(axis="y", visible=False)
        y = np.arange(len(LABEL_ORDER))[::-1]
        for yi, lab in zip(y, LABEL_ORDER):
            c = counts.get(lab, 0)
            ax.barh(yi, 100 * c / n, height=0.55, color=LABEL_COLORS[lab])
            ax.annotate(f"{100*c/n:.0f}%", xy=(100 * c / n, yi),
                        xytext=(3, 0), textcoords="offset points",
                        va="center", fontsize=7, color=INK2)
        ax.set_yticks(y)
        ax.set_yticklabels(LABEL_ORDER, fontsize=8)
        ax.set_xlim(0, 45)
        short = name.split("—")[0].strip()
        ax.set_title(short, fontsize=8.5, loc="left", color=INK, pad=6)
    fig.suptitle("Threshold iterations (share % by label)",
                 fontsize=10, x=0.01, ha="left", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_path)
    plt.close(fig)


KDE_BW = 0.05


def kde_valleys(vals):
    vals = np.asarray(vals, dtype=float)
    if len(vals) < 10 or np.ptp(vals) < 1e-9:
        return []
    try:
        kde = sps.gaussian_kde(vals, bw_method=KDE_BW / max(vals.std(), 1e-9))
        grid = np.linspace(0, 1, 501)
        dens = kde(grid)
    except Exception:
        return []
    valleys = []
    dmax = dens.max()
    for i in range(1, len(grid) - 1):
        if dens[i] < dens[i - 1] and dens[i] <= dens[i + 1]:
            left_peak = dens[:i].max()
            right_peak = dens[i:].max()
            depth = (min(left_peak, right_peak) - dens[i]) / dmax
            if depth > 0.02:
                valleys.append((float(grid[i]), float(depth)))
    return valleys


def simulate(df, rule):
    labels = [rule(p) for p in df["_params"]]
    counts = {lab: 0 for lab in LABEL_ORDER}
    for l in labels:
        counts[l or "none"] += 1
    return labels, counts


def main():
    df = load_dataset()
    n = len(df)
    lines = []

    def emit(s=""):
        print(s)
        lines.append(s)

    emit("# New caption dimension \"space\": analysis and proposal "
         "(517 handcrafted presets)")
    emit()
    emit(f"Dataset: `{JSON_DIR}`, {n} presets loaded. No existing label "
         f"for this dimension: the simulated distributions below are "
         f"non-circular predictions.")
    emit()

    emit("## 1. Distributions of the reverb, delay and chorus parameters")
    emit()
    emit("| parameter | mean | median | std | p25–p75 | zeros (=0.0) | "
         "unique values | notes |")
    emit("|---|---:|---:|---:|---|---:|---:|---|")
    notes = {
        "delayx2l": "**unused: always 0.0 on the whole corpus**",
        "delayx2r": "**unused: always 0.0 on the whole corpus**",
        "chorus1enable": "almost binary (0.0 or ~1.0; 7 intermediate values)",
        "chorus2enable": "almost binary (0.0 or ~1.0; 3 intermediate values)",
        "reverblowcut": "median 1.0: default \"pass everything\"",
        "delaytime": "peak at 0.5 (default); with sync>0.5 the value is a rhythmic division",
    }
    for k in SPACE_PARAMS:
        v = df[k].values
        q25, q75 = np.percentile(v, [25, 75])
        emit(f"| {k} | {v.mean():.3f} | {np.median(v):.3f} | {v.std():.3f} "
             f"| {q25:.2f}–{q75:.2f} | {(v == 0).sum()} "
             f"({100*(v == 0).mean():.0f}%) | {len(np.unique(v))} | "
             f"{notes.get(k, '')} |")
    emit()
    emit("Summary: reverbwet mean "
         f"{df.reverbwet.mean():.3f}, median {df.reverbwet.median():.3f}, "
         f"{(df.reverbwet > 0.3).sum()}/517 above 0.3 "
         f"({100*(df.reverbwet > 0.3).mean():.1f}%); chorus on (either "
         f"enable > 0.5): "
         f"{((df.chorus1enable > 0.5) | (df.chorus2enable > 0.5)).sum()}/517 "
         f"({100*((df.chorus1enable > 0.5) | (df.chorus2enable > 0.5)).mean():.1f}%).")
    emit()

    emit("## 2. Density valleys (KDE) and threshold placement")
    emit()
    kv_rw = kde_valleys(df.reverbwet.values)
    kv_rd = kde_valleys(df.loc[df.reverbwet > 0.25, "reverbdecay"].values)
    kv_dw = kde_valleys(df.loc[df.delaywet > 0, "delaywet"].values)
    emit(f"- `reverbwet` (all): valleys at "
         f"{', '.join(f'{v:.2f} (depth {d:.2f})' for v, d in kv_rw) or 'none'}. "
         f"The valley at ~0.13 separates the zero cluster (226 presets at 0.0) "
         f"from actual use. The \"spacious\" threshold is set at 0.25, above the "
         f"valley: between 0.13 and 0.25 the reverb is barely audible "
         f"({((df.reverbwet > 0.13) & (df.reverbwet <= 0.25)).sum()} presets, "
         f"better left as None).")
    emit(f"- `reverbdecay` given reverbwet>0.25: valleys at "
         f"{', '.join(f'{v:.2f} (depth {d:.2f})' for v, d in kv_rd) or 'none'}. "
         f"The \"cavernous\" threshold reverbdecay>0.65 falls on this valley.")
    emit(f"- `delaywet` (non-zero only): valleys at "
         f"{', '.join(f'{v:.2f} (depth {d:.2f})' for v, d in kv_dw) or 'none'}. "
         f"No useful valley below 0.8: the mass is continuous. The 0.25 "
         f"threshold for \"echoey\" balances the labels (see section 4) rather "
         f"than following a natural valley: above it the echo is clearly "
         f"in the foreground.")
    emit(f"- `delayfeedback` given delaywet>0.15: median "
         f"{df.loc[df.delaywet > 0.15, 'delayfeedback'].median():.2f}, "
         f"p75 {np.percentile(df.loc[df.delaywet > 0.15, 'delayfeedback'], 75):.2f} "
         f"The audible delay in this corpus is a low-feedback slap-back, "
         f"almost never a regenerative echo (the reason Attempt 1 fails, "
         f"see section 4). corr(delaywet, delayfeedback) for "
         f"delaywet>0: "
         f"{np.corrcoef(df.loc[df.delaywet > 0, 'delaywet'], df.loc[df.delaywet > 0, 'delayfeedback'])[0, 1]:+.2f}.")
    emit()

    hist_with_thresholds(df.reverbwet.values, "reverbwet",
                         [(0.05, "dry: wet <= 0.05"),
                          (0.25, "spacious: wet > 0.25"),
                          (0.4, "cavernous: wet > 0.4")],
                         PLOT_DIR / "hist_reverbwet.png")
    hist_with_thresholds(df.delaywet.values, "delaywet",
                         [(0.05, "dry: wet <= 0.05"),
                          (0.25, "echoey: wet > 0.25")],
                         PLOT_DIR / "hist_delaywet.png")
    hist_with_thresholds(df.loc[df.reverbwet > 0.25, "reverbdecay"].values,
                         "reverbdecay | reverbwet>0.25",
                         [(0.65, "cavernous: decay > 0.65")],
                         PLOT_DIR / "hist_reverbdecay_cond.png", bins=30)
    emit(f"Histograms with thresholds: `space_plots/hist_reverbwet.png`, "
         f"`hist_delaywet.png`, `hist_reverbdecay_cond.png`.")
    emit()

    emit("## 3. Co-occurrence of the effects (audibility thresholds)")
    emit()
    R = df.reverbwet > 0.25
    D = df.delaywet > 0.25
    C = (df.chorus1enable > 0.5) | (df.chorus2enable > 0.5)
    emit("| reverb (wet>0.25) | delay (wet>0.25) | chorus (>0.5) | n | % |")
    emit("|:---:|:---:|:---:|---:|---:|")
    import itertools
    for r, d, c in itertools.product([False, True], repeat=3):
        m = (R == r) & (D == d) & (C == c)
        emit(f"| {'x' if r else ''} | {'x' if d else ''} | "
             f"{'x' if c else ''} | {m.sum()} | {100*m.mean():.1f}% |")
    emit()
    emit("All eight combinations occur with non-trivial counts: the "
         "priority of the rule (large reverb > delay > reverb > chorus) "
         "chooses the word when effects coexist. Chorus is the most "
         "secondary effect (often on together with others), so it is "
         "chosen only when it is the only audible effect.")
    emit()

    emit("## 4. Iterations (each attempt simulated on the corpus)")
    emit()
    emit("Acceptance criteria: no label above ~45%, no near-empty label "
         "below ~4%, None below ~35%.")
    emit()
    all_counts = []
    for name, rule in ATTEMPTS:
        labels, counts = simulate(df, rule)
        all_counts.append((name, counts))
        emit(f"### {name}")
        emit()
        emit("| label | n | % |")
        emit("|---|---:|---:|")
        for lab in LABEL_ORDER:
            emit(f"| {lab} | {counts[lab]} | {100*counts[lab]/n:.1f}% |")
        emit()
        if name.startswith("Attempt 1"):
            emit("**Rejected**: \"echoey\" at 2.9% is a near-empty "
                 "label. Cause (see section 2): the feedback of the audible delay "
                 "in this corpus is low (slap-back), and the condition "
                 "`delayfeedback > 0.3`, taken from the delay branch of "
                 "\"pulsing\", selects a kind of delay that is almost "
                 "absent here. Fix: \"echoey\" must depend on the wet level "
                 "(how audible the echo is), not on the feedback (how many "
                 "repetitions).")
            emit()
        elif name.startswith("Attempt 2a"):
            emit("Balanced, but \"echoey\" (26.5%) exceeds \"spacious\" "
                 "(21.9%) although delay is used less than reverb in the "
                 "corpus (34% against 43% above the respective audibility "
                 "thresholds): wet>0.20 is too inclusive and takes too many "
                 "presets with both reverb and delay.")
            emit()
        elif name.startswith("Attempt 2b"):
            emit("Balanced (max 23.6%, None 4.8%). One semantic issue "
                 "remains: 17 of the 106 \"echoey\" presets have an "
                 "unsynchronised delay with `delaytime < 0.05` (a few "
                 "milliseconds: a comb filter or flanger, no audible echo).")
            emit()
        elif name.startswith("Attempt 2c"):
            emit("More conservative alternative (echoey 15.7%): rejected "
                 "in favour of 2b with the anti-comb guard, which removes the "
                 "false positives directly instead of raising the threshold "
                 "for all presets.")
            emit()
        else:
            emit("**Adopted**: the guard `(delaysync > 0.5 or delaytime > "
                 "0.05)` moves the 17 false echoey presets to "
                 "spacious, chorused or None according to their other "
                 "parameters. With sync on, delaytime is a rhythmic division, "
                 "so even a value near 0 is an audible echo and is kept.")
            emit()

    iteration_panel(all_counts, n, PLOT_DIR / "iteration_path.png")
    emit(f"Small multiples of the iterations: `space_plots/iteration_path.png`.")
    emit()

    emit("## 5. Final proposal")
    emit()
    emit("Dimension name: **`space`** (key `\"space\"` in `choices`). "
         "Vocabulary: **dry, cavernous, spacious, echoey, chorused** "
         "(single perceptual adjectives, as in the other dimensions).")
    emit()
    final_labels, final_counts = simulate(df, space_from_params)
    counts_bar(final_counts, n,
               "Presets by proposed space label (n=517)",
               PLOT_DIR / "category_counts.png")
    emit("| label | n | % | meaning |")
    emit("|---|---:|---:|---|")
    meanings = {
        "dry": "no space effect (wet ≤ 0.05, chorus off)",
        "cavernous": "large reverb: wet > 0.4 and decay > 0.65",
        "spacious": "audible reverb: wet > 0.25",
        "echoey": "audible delay: wet > 0.25 (with anti-comb guard)",
        "chorused": "chorus on, without a dominant reverb or delay",
        "none": "effects present but below the thresholds",
    }
    for lab in LABEL_ORDER:
        emit(f"| {lab} | {final_counts[lab]} | "
             f"{100*final_counts[lab]/n:.1f}% | {meanings[lab]} |")
    emit()
    emit("Violin plots by label: `space_plots/violin_<param>.png`; counts: "
         "`space_plots/category_counts.png`.")
    emit()
    for param, thr in [("reverbwet", [(0.25, "spacious"), (0.4, "cavernous")]),
                       ("reverbdecay", [(0.65, "cavernous")]),
                       ("delaywet", [(0.25, "echoey")]),
                       ("delayfeedback", [])]:
        violin_by_label(df, final_labels, param, thr,
                        PLOT_DIR / f"violin_{param}.png")

    emit("### Proposed code (for captioner.py)")
    emit()
    emit("```python")
    import inspect
    emit(inspect.getsource(space_from_params).rstrip())
    emit("```")
    emit()
    emit("Integration: add `\"space\": space_from_params(params)` "
         "to the `choices` dict.")
    emit()

    emit("## 6. Overlap with \"pulsing\" (timeShape)")
    emit()
    pul_mask = df.pulsing.values
    npul = int(pul_mask.sum())
    lab_series = pd.Series([l or "none" for l in final_labels])
    emit(f"Presets labelled \"pulsing\" by the timeShape rule "
         f"(delaysync>0.5 & feedback>0.35 & wet>0.1, or a fast LFO): **{npul}**.")
    emit()
    emit("| space label | n pulsing | n total | % of pulsing | % of the "
         "label |")
    emit("|---|---:|---:|---:|---:|")
    for lab in LABEL_ORDER:
        both = int(((lab_series == lab) & pul_mask).sum())
        tot = final_counts[lab]
        emit(f"| {lab} | {both} | {tot} | "
             f"{100*both/max(npul, 1):.0f}% | {100*both/max(tot, 1):.1f}% |")
    emit()
    ech_pul = int(((lab_series == "echoey") & pul_mask).sum())
    emit(f"Assessment: the two dimensions are almost uncorrelated. Only {ech_pul} "
         f"of the {npul} pulsing presets are also \"echoey\" ({ech_pul}/"
         f"{final_counts['echoey']} = "
         f"{100*ech_pul/final_counts['echoey']:.1f}% of the echoey presets): "
         f"\"pulsing\" comes mostly from the LFO branch and from the "
         f"synchronised regenerative delay (high feedback), while \"echoey\" "
         f"captures the low-feedback slap-back delay, which is the norm in the "
         f"corpus. There is no redundancy: a preset with both labels "
         f"pulses in time and echoes in space.")
    emit()

    emit("## Output files")
    emit()
    emit("- Script: `space_dimension_analysis.py`")
    emit("- Report: `space_dimension_analysis_report.md`")
    emit("- Plots: `space_plots/` (category_counts.png, "
         "iteration_path.png, hist_reverbwet.png, hist_delaywet.png, "
         "hist_reverbdecay_cond.png, violin_reverbwet.png, "
         "violin_reverbdecay.png, violin_delaywet.png, "
         "violin_delayfeedback.png)")

    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nReport written to {REPORT_PATH}")


if __name__ == "__main__":
    main()
