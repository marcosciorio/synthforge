#!/usr/bin/env python3
"""Statistical analysis of the thresholds used by the timeShape rules
of the captioner, on the reference presets.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
JSON_DIR = ROOT / "data" / "Dataset_TAL_48kHz" / "not_in_pool" / "json"
CAP_DIR = ROOT / "data" / "Dataset_TAL_48kHz" / "not_in_pool" / "captions"
EVAL_DIR = Path(__file__).resolve().parent
PLOT_DIR = EVAL_DIR / "timeshape_plots"
REPORT_PATH = EVAL_DIR / "timeshape_threshold_analysis_report.md"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

_LFO_DEST = {0: "osc1+2", 1: "osc1", 2: "osc2", 3: "pw1+2",
             4: "fm", 5: "cutoff", 6: "volume", 7: "pan"}


def _lfo_dest(val: float) -> str:
    return _LFO_DEST[min(int(round(val * 7)), 7)]


def rule_branches(p):
    A = p.get("ampattack", 0); D = p.get("ampdecay", 0)
    S = p.get("ampsustain", 0); R = p.get("amprelease", 0)
    lfo1_amt = p.get("lfo1amount", 0); lfo1_rate = p.get("lfo1rate", 0)
    lfo1_d = _lfo_dest(p.get("lfo1destination", 0))
    lfo2_amt = p.get("lfo2amount", 0); lfo2_rate = p.get("lfo2rate", 0)
    lfo2_d = _lfo_dest(p.get("lfo2destination", 0))
    delay_sync = p.get("delaysync", 0); delay_fb = p.get("delayfeedback", 0)
    fad_amt = p.get("freeadamount", 0); fad_dest = p.get("freeaddestination", 0)

    pulsing = (
        (lfo1_amt > 0.3 and lfo1_d == "volume" and lfo1_rate > 0.4) or
        (lfo2_amt > 0.3 and lfo2_d == "volume" and lfo2_rate > 0.4) or
        (delay_sync > 0.5 and delay_fb > 0.3)
    )
    evolving = (
        (lfo1_amt > 0.2 and lfo1_rate < 0.3 and lfo1_d == "cutoff") or
        (lfo2_amt > 0.2 and lfo2_rate < 0.3 and lfo2_d == "cutoff") or
        (fad_amt > 0.5 and fad_dest < 0.65)
    )
    fading = S < 0.2 and D > 0.4 and R > 0.3

    if pulsing:
        label = "pulsing"
    elif evolving:
        label = "evolving"
    elif A > 0.4:
        label = "slow"
    elif R > 0.6 or S > 0.6:
        label = "long"
    elif fading:
        label = "fading"
    elif R < 0.25 and S < 0.3:
        label = "short"
    else:
        label = None
    flags = dict(pulsing=pulsing, evolving=evolving, fading=fading,
                 lfo1_d=lfo1_d, lfo2_d=lfo2_d)
    return label, flags


USED_PARAMS = ["ampattack", "ampdecay", "ampsustain", "amprelease",
               "lfo1amount", "lfo1rate", "lfo2amount", "lfo2rate",
               "delaysync", "delayfeedback", "freeadamount",
               "freeaddestination"]
EXCLUDED_PARAMS = ["filterattack", "filterdecay", "filtersustain",
                   "filterrelease", "filtercontour", "freeadattack",
                   "freeaddecay", "delaywet"]
ALL_PARAMS = USED_PARAMS + EXCLUDED_PARAMS + ["lfo1destination",
                                              "lfo2destination"]

LABEL_ORDER = ["pulsing", "evolving", "slow", "long", "fading", "short",
               "none"]


def load_dataset():
    rows = []
    mismatches = 0
    for jf in sorted(JSON_DIR.glob("*.json")):
        cf = CAP_DIR / jf.name
        if not cf.exists():
            print(f"WARNING: missing caption for {jf.name}", file=sys.stderr)
            continue
        params = json.load(open(jf))["parameters"]
        cap = json.load(open(cf))
        ts_caption = cap.get("choices", {}).get("timeShape") or "none"
        ts_rule, flags = rule_branches(params)
        ts_rule = ts_rule or "none"
        if ts_rule != ts_caption:
            mismatches += 1
        row = {"name": jf.stem, "timeShape": ts_caption,
               "timeShape_recomputed": ts_rule}
        row.update(flags)
        for k in ALL_PARAMS:
            row[k] = params.get(k, 0.0)
        rows.append(row)
    df = pd.DataFrame(rows)
    return df, mismatches


SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

LABEL_COLORS = {
    "pulsing":  "#2a78d6",
    "evolving": "#1baf7a",
    "slow":     "#eda100",
    "long":     "#008300",
    "fading":   "#4a3aa7",
    "short":    "#e34948",
    "none":     "#898781",
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


def violin_by_label(df, param, thresholds, out_path, title_extra=""):
    fig, ax = plt.subplots(figsize=(8.2, 4.2), dpi=150)
    _style_ax(ax)
    rng = np.random.default_rng(42)
    positions, tick_labels = [], []
    for i, lab in enumerate(LABEL_ORDER):
        vals = df.loc[df.timeShape == lab, param].values
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
        branch = note.split(":")[0]
        ax.axhline(thr, color=INK2, linewidth=1.0, zorder=2)
        ax.annotate(f"{thr:g} · {branch}", xy=(1.0, thr),
                    xycoords=("axes fraction", "data"),
                    xytext=(4, 0), textcoords="offset points",
                    va="center", ha="left", fontsize=7.5, color=INK2)
    ax.set_xticks(positions)
    ax.set_xticklabels(tick_labels, fontsize=8)
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel(param, fontsize=9)
    ax.set_title(f"{param} per timeShape{title_extra}", fontsize=10,
                 color=INK, loc="left", pad=10)
    fig.subplots_adjust(right=0.86, left=0.08, bottom=0.13, top=0.90)
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


def nearest_valley(valleys, thr, max_dist=0.25):
    cand = [(abs(v - thr), v, d) for v, d in valleys if abs(v - thr) <= max_dist]
    if not cand:
        return None
    cand.sort()
    return cand[0][1], cand[0][2]


def cohens_d(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or len(b) < 2:
        return np.nan
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) /
                 (na + nb - 2))
    if sp < 1e-12:
        return np.nan
    return (a.mean() - b.mean()) / sp


def eta_squared(groups):
    allv = np.concatenate(groups)
    gm = allv.mean()
    ss_b = sum(len(g) * (g.mean() - gm) ** 2 for g in groups if len(g) > 0)
    ss_t = ((allv - gm) ** 2).sum()
    return ss_b / ss_t if ss_t > 1e-12 else np.nan


THRESHOLDS = {
    "ampattack":     [(0.4, "slow: A > 0.4")],
    "ampdecay":      [(0.4, "fading: D > 0.4")],
    "ampsustain":    [(0.2, "fading: S < 0.2"), (0.3, "short: S < 0.3"),
                      (0.6, "long: S > 0.6")],
    "amprelease":    [(0.25, "short: R < 0.25"), (0.3, "fading: R > 0.3"),
                      (0.6, "long: R > 0.6")],
    "lfo1amount":    [(0.2, "evolving: amt > 0.2"), (0.3, "pulsing: amt > 0.3")],
    "lfo1rate":      [(0.3, "evolving: rate < 0.3"), (0.4, "pulsing: rate > 0.4")],
    "lfo2amount":    [(0.2, "evolving: amt > 0.2"), (0.3, "pulsing: amt > 0.3")],
    "lfo2rate":      [(0.3, "evolving: rate < 0.3"), (0.4, "pulsing: rate > 0.4")],
    "delaysync":     [(0.5, "pulsing: sync > 0.5")],
    "delayfeedback": [(0.3, "pulsing: fb > 0.3")],
    "freeadamount":  [(0.5, "evolving: amt > 0.5")],
    "freeaddestination": [(0.65, "evolving: dest < 0.65")],
}

CONDITIONAL_SUBSETS = {
    ("lfo1amount", 0.3): ("lfo1_d", "volume"),
    ("lfo1rate", 0.4): ("lfo1_d", "volume"),
    ("lfo1amount", 0.2): ("lfo1_d", "cutoff"),
    ("lfo1rate", 0.3): ("lfo1_d", "cutoff"),
    ("lfo2amount", 0.3): ("lfo2_d", "volume"),
    ("lfo2rate", 0.4): ("lfo2_d", "volume"),
    ("lfo2amount", 0.2): ("lfo2_d", "cutoff"),
    ("lfo2rate", 0.3): ("lfo2_d", "cutoff"),
}


def main():
    df, mismatches = load_dataset()
    n = len(df)
    lines = []

    def emit(s=""):
        print(s)
        lines.append(s)

    emit("# TimeShape threshold analysis (517 handcrafted presets)")
    emit()
    emit(f"Dataset: `{JSON_DIR}` and `{CAP_DIR}`, {n} presets loaded.")
    emit(f"Sanity check: label recomputed by the rule against `choices.timeShape` "
         f"in the captions: **{mismatches} mismatches** out of {n}.")
    emit()

    emit("## 1. Label distribution")
    emit()
    counts = df.timeShape.value_counts().reindex(LABEL_ORDER).fillna(0).astype(int)
    emit("| timeShape | n | % |")
    emit("|---|---:|---:|")
    for lab in LABEL_ORDER:
        emit(f"| {lab} | {counts[lab]} | {100*counts[lab]/n:.1f}% |")
    emit()

    fig, ax = plt.subplots(figsize=(6.4, 3.4), dpi=150)
    _style_ax(ax)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.grid(axis="y", visible=False)
    y = np.arange(len(LABEL_ORDER))[::-1]
    for yi, lab in zip(y, LABEL_ORDER):
        ax.barh(yi, counts[lab], height=0.55, color=LABEL_COLORS[lab])
        ax.annotate(f"{counts[lab]}  ({100*counts[lab]/n:.1f}%)",
                    xy=(counts[lab], yi), xytext=(5, 0),
                    textcoords="offset points", va="center",
                    fontsize=8, color=INK2)
    ax.set_yticks(y)
    ax.set_yticklabels(LABEL_ORDER, fontsize=9)
    ax.set_xlim(0, counts.max() * 1.22)
    ax.set_title("Presets by timeShape label (n=517)", fontsize=10,
                 loc="left", color=INK, pad=10)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "category_counts.png")
    plt.close(fig)

    emit("## 2. Parameters used by the rule: thresholds and distributions")
    emit()
    emit(f"Violin plots by parameter in `timeshape_plots/used_<param>.png` "
         f"(thresholds as horizontal lines). Density valleys "
         f"estimated with a Gaussian KDE, fixed bandwidth {KDE_BW}.")
    emit()
    emit("| parameter | threshold (branch) | % beyond threshold (all) | % beyond "
         "(relevant subset) | median | p25–p75 | nearest KDE valley | "
         "assessment |")
    emit("|---|---|---:|---:|---:|---|---|---|")

    for param in USED_PARAMS:
        vals_all = df[param].values
        valleys = kde_valleys(vals_all)
        violin_by_label(df, param, THRESHOLDS[param],
                        PLOT_DIR / f"used_{param}.png")
        for thr, note in THRESHOLDS[param]:
            direction = ">" if ">" in note.split(":")[1] else "<"
            if direction == ">":
                frac_all = float((vals_all > thr).mean())
            else:
                frac_all = float((vals_all < thr).mean())
            key = (param, thr)
            frac_sub_s = "—"
            if key in CONDITIONAL_SUBSETS:
                col, want = CONDITIONAL_SUBSETS[key]
                sub = df.loc[df[col] == want, param].values
                if len(sub) > 0:
                    fs = float((sub > thr).mean() if direction == ">"
                               else (sub < thr).mean())
                    frac_sub_s = f"{100*fs:.1f}% (n={len(sub)})"
                else:
                    frac_sub_s = "n=0"
            med = float(np.median(vals_all))
            q25, q75 = np.percentile(vals_all, [25, 75])
            nv = nearest_valley(valleys, thr)
            if nv is None:
                nv_s = "none within ±0.25"
                verdict = ("threshold in a region without valleys: arbitrary "
                           "with respect to the distribution")
            else:
                v, depth = nv
                nv_s = f"{v:.2f} (depth {depth:.2f})"
                if abs(v - thr) <= 0.05:
                    verdict = "well placed (close to a valley)"
                else:
                    verdict = f"the data suggest ≈{v:.2f}"
            emit(f"| {param} | {thr:g} ({note.split(':')[0]}, {direction}) | "
                 f"{100*frac_all:.1f}% | {frac_sub_s} | {med:.2f} | "
                 f"{q25:.2f}–{q75:.2f} | {nv_s} | {verdict} |")
    emit()

    emit("### LFO subsets by destination")
    emit()
    emit("| LFO | destination | n presets |")
    emit("|---|---|---:|")
    for lfo, col in [("lfo1", "lfo1_d"), ("lfo2", "lfo2_d")]:
        for dest, cnt in df[col].value_counts().items():
            emit(f"| {lfo} | {dest} | {cnt} |")
    emit()

    not_pe = df[(~df.pulsing) & (~df.evolving)]
    emit("### Cascade of the amplitude-envelope branches (after pulsing/evolving)")
    emit()
    n_pe = n - len(not_pe)
    emit(f"- Presets captured by pulsing/evolving: {n_pe} "
         f"({100*n_pe/n:.1f}%); the other {len(not_pe)} reach the amplitude "
         f"branches.")
    elig = not_pe
    a_gt = elig[elig.ampattack > 0.4]
    emit(f"- slow branch (A>0.4): {len(a_gt)}/{len(elig)} eligible "
         f"({100*len(a_gt)/len(elig):.1f}%)")
    elig = elig[elig.ampattack <= 0.4]
    l_hit = elig[(elig.amprelease > 0.6) | (elig.ampsustain > 0.6)]
    emit(f"- long branch (R>0.6 or S>0.6): {len(l_hit)}/{len(elig)} "
         f"({100*len(l_hit)/len(elig):.1f}%)")
    elig = elig[~((elig.amprelease > 0.6) | (elig.ampsustain > 0.6))]
    f_hit = elig[(elig.ampsustain < 0.2) & (elig.ampdecay > 0.4) &
                 (elig.amprelease > 0.3)]
    emit(f"- fading branch (S<0.2, D>0.4, R>0.3): {len(f_hit)}/{len(elig)} "
         f"({100*len(f_hit)/len(elig):.1f}%)")
    elig = elig.drop(f_hit.index)
    s_hit = elig[(elig.amprelease < 0.25) & (elig.ampsustain < 0.3)]
    emit(f"- short branch (R<0.25, S<0.3): {len(s_hit)}/{len(elig)} "
         f"({100*len(s_hit)/len(elig):.1f}%)")
    emit(f"- remaining None: {len(elig) - len(s_hit)}")
    emit()

    emit("### Breakdown of the pulsing/evolving triggers")
    emit()

    def g(row, k):
        return row[k]

    ev_lfo = ev_fad = ev_both = ev_fad_dead = 0
    pul_lfo = pul_dly = pul_dly_wet0 = 0
    for _, r in df.iterrows():
        pul_l = ((r.lfo1amount > 0.3 and r.lfo1_d == "volume" and
                  r.lfo1rate > 0.4) or
                 (r.lfo2amount > 0.3 and r.lfo2_d == "volume" and
                  r.lfo2rate > 0.4))
        pul_d = (r.delaysync > 0.5 and r.delayfeedback > 0.3)
        if pul_l or pul_d:
            if pul_l:
                pul_lfo += 1
            if pul_d:
                pul_dly += 1
                if r.delaywet <= 0.05:
                    pul_dly_wet0 += 1
            continue
        lfo_cut = ((r.lfo1amount > 0.2 and r.lfo1rate < 0.3 and
                    r.lfo1_d == "cutoff") or
                   (r.lfo2amount > 0.2 and r.lfo2rate < 0.3 and
                    r.lfo2_d == "cutoff"))
        fad = (r.freeadamount > 0.5 and r.freeaddestination < 0.65)
        if lfo_cut and fad:
            ev_both += 1
        elif lfo_cut:
            ev_lfo += 1
        elif fad:
            ev_fad += 1
            if r.freeadattack <= 0.05 and r.freeaddecay <= 0.05:
                ev_fad_dead += 1
    emit(f"- evolving: {ev_lfo} LFO→cutoff only, {ev_fad} free envelope only, "
         f"{ev_both} both; the free-envelope branch produces "
         f"{100*(ev_fad+ev_both)/max(ev_lfo+ev_fad+ev_both,1):.0f}% "
         f"of the evolving labels.")
    emit(f"  Of the free-envelope-only presets, **{ev_fad_dead} have a flat envelope** "
         f"(freeadattack ≤ 0.05 and freeaddecay ≤ 0.05: no audible "
         f"change): false positives, removed by adding "
         f"`freeaddecay > 0.05` (or attack) to the branch.")
    emit(f"- pulsing: {pul_lfo} from the LFO, {pul_dly} from the delay; "
         f"**{pul_dly_wet0} of the {pul_dly} delay cases have delaywet ≤ 0.05** "
         f"(inaudible delay): false positives, removed by adding "
         f"`delaywet > 0.1` to the delay branch.")
    emit()

    emit("## 3. Parameters not used by the rule: separation between labels")
    emit()
    emit("Statistics computed on the labels assigned by the rule "
         "(non-circular evidence: the sound designer chose these values "
         "independently of the rule). eta² = SS_between/SS_total; "
         "Kruskal–Wallis as a non-parametric check; Cohen's d for the "
         "targeted comparisons.")
    emit()
    emit("| parameter | eta² (6 labels + none) | ANOVA F (p) | "
         "Kruskal-Wallis H (p) | targeted comparison | Cohen's d |")
    emit("|---|---:|---|---|---|---:|")

    TARGETED = {
        "filterattack":  ("evolving", "vs rest"),
        "filtercontour": ("evolving", "vs rest"),
        "filterdecay":   ("fading", "vs rest"),
        "filtersustain": ("long", "vs rest"),
        "filterrelease": ("long", "vs rest"),
        "freeadattack":  ("evolving", "vs rest"),
        "freeaddecay":   ("evolving", "vs rest"),
        "delaywet":      ("pulsing", "vs rest"),
    }

    sep_rows = []
    for param in EXCLUDED_PARAMS:
        groups = [df.loc[df.timeShape == lab, param].values
                  for lab in LABEL_ORDER]
        groups_nz = [g for g in groups if len(g) >= 2]
        e2 = eta_squared(groups_nz)
        try:
            F, pF = sps.f_oneway(*groups_nz)
        except Exception:
            F, pF = np.nan, np.nan
        try:
            H, pH = sps.kruskal(*groups_nz)
        except Exception:
            H, pH = np.nan, np.nan
        tgt_lab, tgt_desc = TARGETED[param]
        a = df.loc[df.timeShape == tgt_lab, param].values
        b = df.loc[df.timeShape != tgt_lab, param].values
        d = cohens_d(a, b)
        emit(f"| {param} | {e2:.3f} | {F:.1f} ({pF:.1e}) | "
             f"{H:.1f} ({pH:.1e}) | {tgt_lab} {tgt_desc} | {d:+.2f} |")
        sep_rows.append((param, e2, F, pF, H, pH, tgt_lab, d))
        violin_by_label(df, param, [], PLOT_DIR / f"excl_{param}.png",
                        title_extra="  (parameter not used by the rule)")
    emit()

    fig, ax = plt.subplots(figsize=(6.4, 3.2), dpi=150)
    _style_ax(ax)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.grid(axis="y", visible=False)
    order = sorted(sep_rows, key=lambda r: r[1])
    y = np.arange(len(order))
    for yi, r in zip(y, order):
        ax.barh(yi, r[1], height=0.55, color="#2a78d6")
        ax.annotate(f"{r[1]:.3f}", xy=(r[1], yi), xytext=(5, 0),
                    textcoords="offset points", va="center", fontsize=8,
                    color=INK2)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in order], fontsize=8.5)
    ax.set_xlim(0, max(r[1] for r in order) * 1.25)
    ax.set_title("Separation between labels (eta²), parameters not used by the rule",
                 fontsize=10, loc="left", color=INK, pad=10)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "excluded_eta_squared.png")
    plt.close(fig)

    emit("### Medians by label (parameters not used by the rule)")
    emit()
    header = "| parameter | " + " | ".join(LABEL_ORDER) + " |"
    emit(header)
    emit("|---|" + "---:|" * len(LABEL_ORDER))
    for param in EXCLUDED_PARAMS:
        meds = [f"{np.median(df.loc[df.timeShape == lab, param]):.2f}"
                if (df.timeShape == lab).sum() > 0 else "—"
                for lab in LABEL_ORDER]
        emit(f"| {param} | " + " | ".join(meds) + " |")
    emit()

    emit("## Output files")
    emit()
    emit(f"- Script: `timeshape_threshold_analysis.py`")
    emit(f"- Report: `timeshape_threshold_analysis_report.md`")
    emit(f"- Plots: `timeshape_plots/` "
         f"(category_counts.png, used_<param>.png ×{len(USED_PARAMS)}, "
         f"excl_<param>.png ×{len(EXCLUDED_PARAMS)}, "
         f"excluded_eta_squared.png)")

    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nReport written to {REPORT_PATH}")


if __name__ == "__main__":
    main()
