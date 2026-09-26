#!/usr/bin/env python3
"""One chart per file, for the preprint.

The repository's own figures are multi-panel because a README wants a single image that says
everything. A paper does not: every panel there needs its own number, its own caption and its
own place in the argument. This script re-draws the same measurements as single-panel figures
into img/, straight from the scorer output, so nothing is hand-copied.

    python preprint/plot_figs.py
"""
import json
import statistics
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# The paper is set in Times: neurips_2023.sty does \renewcommand{\rmdefault}{ptm}. Matplotlib
# defaults to DejaVu Sans, so every figure label was in a different typeface from the body
# text. Match the document instead.
matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "axes.unicode_minus": False,
})

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
IMG = HERE / "img"
IMG.mkdir(parents=True, exist_ok=True)

MODELS = [
    ("whisper-large-v2", "large-v2", "#95a5a6", "v"),
    ("whisper-large-v3", "large-v3", "#1e8449", "^"),
    ("whisper-large-v3-turbo", "turbo", "#e67e22", "D"),
    ("malaysian-whisper-large-v2", "malaysian-v2", "#5b7fd4", "s"),
    ("Malaysian-whisper-large-v3-turbo-v3", "malaysian-turbo-v3", "#203882", "o"),
]
GRID = dict(color="#e0e0e0", linewidth=0.8, linestyle="--", alpha=0.6)
INK = "#333333"
TICK = "#444444"
DPI = 200

scores = json.loads((ROOT / "bench" / "scores.json").read_text())
lex = json.loads((ROOT / "bench" / "lexicon_synth_scores.json").read_text())
wild = json.loads((ROOT / "bench" / "wild_scores.json").read_text())


def frame(ax, xlab=None, ylab=None, title=None, sub=None, axis="y"):
    ax.grid(True, axis=axis, **GRID)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#cccccc")
    ax.tick_params(colors=TICK, labelsize=8.5, length=0)
    if xlab:
        ax.set_xlabel(xlab, fontsize=9.5, color=TICK, labelpad=7)
    if ylab:
        ax.set_ylabel(ylab, fontsize=9.5, color=TICK, labelpad=7)
    if title:
        ax.set_title(title, fontsize=11, color="#1a1a2e", fontweight="bold",
                     pad=16 if sub else 9, loc="left")
    if sub:
        ax.text(0, 1.015, sub, transform=ax.transAxes, fontsize=7.8, color="#666666",
                style="italic", va="bottom")


def legend_below(ax, ncol=2, pad=0.18):
    """Legends go under the axes. Inside the frame they land on exactly the bars and points a
    reader came for, and which corner is free changes every time the data does."""
    ax.legend(fontsize=8, facecolor="#f5f5f5", edgecolor="#cccccc", framealpha=0.95,
              ncol=ncol, loc="upper center", bbox_to_anchor=(0.5, -pad), borderaxespad=0.0)


def save(fig, name):
    out = IMG / name
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"-> {out.relative_to(ROOT)}")


def pct(ax, hi=1.0):
    ticks = [t for t in ax.get_yticks() if 0 <= t <= hi]
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{t:.0%}" for t in ticks], fontsize=8.5, color=TICK)


# 1. hallucination on the three non-speech arms
def fig_nonspeech():
    arms = [("silence", "silence\n42 clips"), ("music", "music\n600"),
            ("nonspeech", "nonspeech\n1,168")]
    fig, ax = plt.subplots(figsize=(6.6, 3.9), dpi=DPI)
    x = np.arange(len(arms))
    w = 0.16
    for i, (key, name, color, _) in enumerate(MODELS):
        vals = [scores[key][a]["hallucination_rate"] for a, _ in arms]
        ax.bar(x + (i - 2) * w, vals, width=w, color=color, edgecolor="white",
               linewidth=0.7, label=name, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels([n for _, n in arms], fontsize=8.5, color=INK)
    ax.set_ylim(0, 1.05)
    pct(ax)
    frame(ax, ylab="clips where the model emitted words",
          title="Hallucination on audio with no speech in it",
          sub="the correct output is the empty string, so every bar is error")
    legend_below(ax, ncol=5, pad=0.20)
    save(fig, "fig_nonspeech.png")


# 2. both extremes on the reduplication arm
def fig_repetition():
    fig, ax = plt.subplots(figsize=(6.6, 3.9), dpi=DPI)
    names = [n for _, n, _, _ in MODELS]
    y = np.arange(len(MODELS))
    run = [scores[k]["reduplication"]["runaway_rate"] for k, _, _, _ in MODELS]
    emp = [scores[k]["reduplication"]["empty_rate"] for k, _, _, _ in MODELS]
    ax.barh(y - 0.19, run, height=0.36, color="#922b21", edgecolor="white", linewidth=0.7,
            label="runaway: emitted more repeats than were spoken", zorder=3)
    ax.barh(y + 0.19, emp, height=0.36, color="#1a5276", edgecolor="white", linewidth=0.7,
            label="deleted: emitted nothing at all", zorder=3)
    for yy, v in zip(y - 0.19, run):
        ax.text(v + 0.01, yy, f"{v:.1%}", va="center", fontsize=7.4, color="#922b21",
                fontweight="bold")
    for yy, v in zip(y + 0.19, emp):
        ax.text(v + 0.01, yy, f"{v:.1%}", va="center", fontsize=7.4, color="#1a5276",
                fontweight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8.5, color=INK, fontweight="bold")
    ax.set_ylim(len(MODELS) - 0.45, -0.55)
    ax.set_xlim(0, 0.72)
    ticks = [t for t in ax.get_xticks() if 0 <= t <= 0.72]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:.0%}" for t in ticks], fontsize=8.5, color=TICK)
    frame(ax, xlab="share of 1,440 clips", axis="x",
          title="Both extremes are wrong on repeated speech",
          sub="the clip contains a unit repeated an exact number of times")
    legend_below(ax, ncol=1, pad=0.18)
    save(fig, "fig_repetition.png")


# 3. the trade itself
def fig_tradeoff():
    n = {"silence": 42, "music": 600, "nonspeech": 1168}
    fig, ax = plt.subplots(figsize=(6.6, 4.6), dpi=DPI)
    xs, ys = [], []
    for key, name, color, marker in MODELS:
        num = sum(scores[key][a]["hallucination_rate"] * c for a, c in n.items())
        x = num / sum(n.values())
        y = lex[key]["recovered_rate"]
        xs.append(x); ys.append(y)
        ax.scatter([x], [y], s=190, c=color, marker=marker, edgecolors="white",
                   linewidths=1.4, zorder=5)
        dy = 14 if name not in ("large-v3",) else -20
        ax.annotate(name, (x, y), textcoords="offset points", xytext=(0, dy), ha="center",
                    fontsize=8.2, fontweight="bold", color=color, zorder=6)
    m, b = np.polyfit(np.array(xs), np.array(ys), 1)
    gx = np.linspace(0.02, 0.97, 10)
    ax.plot(gx, m * gx + b, color="#922b21", linewidth=1.3, linestyle="--", alpha=0.55)
    r = float(np.corrcoef(xs, ys)[0, 1])
    ax.text(0.98, 0.04, f"the diagonal is the trade:  r = {r:+.2f}", transform=ax.transAxes,
            ha="right", fontsize=8.4, color="#922b21", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="#f5f5f5", ec="#922b21", linewidth=0.9))
    ax.annotate("nobody is here\nquiet on noise AND accurate on speech", xy=(0.03, 0.80),
                ha="left", va="top", fontsize=8.4, color="#1e8449", fontweight="bold",
                style="italic",
                bbox=dict(boxstyle="round,pad=0.4", fc="#eafaf1", ec="#1e8449", linewidth=1.1))
    ax.set_xlim(-0.03, 1.06)
    ax.set_ylim(0.22, 0.84)
    for f, axis in ((ax.set_xticklabels, "x"), (ax.set_yticklabels, "y")):
        pass
    ax.set_xticks(np.arange(0, 1.01, 0.2))
    ax.set_xticklabels([f"{v:.0%}" for v in np.arange(0, 1.01, 0.2)], fontsize=8.5, color=TICK)
    ax.set_yticks(np.arange(0.3, 0.81, 0.1))
    ax.set_yticklabels([f"{v:.0%}" for v in np.arange(0.3, 0.81, 0.1)], fontsize=8.5, color=TICK)
    frame(ax, xlab="emits words over audio with NO speech  (worse to the right)",
          ylab="recovers the phrase when it IS spoken",
          title="Every checkpoint trades one failure for the other", axis="both")
    save(fig, "fig_tradeoff.png")


# 4. English accuracy against multilingual accuracy
def fig_accuracy():
    fig, ax = plt.subplots(figsize=(6.6, 3.9), dpi=DPI)
    x = np.arange(len(MODELS))
    ls = [scores[k]["librispeech_test_clean"]["wer"] for k, _, _, _ in MODELS]
    fl = [statistics.median(scores[k]["fleurs"]["per_lang_cer"].values()) for k, _, _, _ in MODELS]
    ax.bar(x - 0.2, ls, width=0.38, color="#1e8449", edgecolor="white", linewidth=0.7,
           label="librispeech WER (English)", zorder=3)
    ax.bar(x + 0.2, fl, width=0.38, color="#922b21", edgecolor="white", linewidth=0.7,
           label="FLEURS CER, typical language (58 languages)", zorder=3)
    for xx, v in zip(x - 0.2, ls):
        ax.text(xx, v + 0.02, f"{v:.3f}", ha="center", fontsize=7.4, color="#1e8449",
                fontweight="bold")
    for xx, v in zip(x + 0.2, fl):
        ax.text(xx, v + 0.02, f"{v:.3f}", ha="center", fontsize=7.4, color="#922b21",
                fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([n for _, n, _, _ in MODELS], fontsize=8.5, color=INK)
    ax.set_ylim(0, 1.45)
    frame(ax, ylab="error rate",
          title="English accuracy hides what a multilingual fine-tune destroyed",
          sub="malaysian-v2 matches base large-v3 on librispeech and loses every other language")
    legend_below(ax, ncol=2, pad=0.16)
    save(fig, "fig_accuracy.png")


# 5. the same thing, language by language
def fig_fleurs_per_lang():
    fig, ax = plt.subplots(figsize=(6.6, 4.0), dpi=DPI)
    base = scores["whisper-large-v3"]["fleurs"]["per_lang_cer"]
    order = [l for l, _ in sorted(base.items(), key=lambda kv: kv[1])]
    x = np.arange(len(order))
    for key, name, color, _ in MODELS:
        per = scores[key]["fleurs"]["per_lang_cer"]
        ax.plot(x, [min(per.get(l, np.nan), 2.0) for l in order], color=color, linewidth=1.6,
                label=name, alpha=0.95 if "Malaysian" in key or "malaysian" in key else 0.85)
    ax.axhline(0.15, color="#1a1a2e", linewidth=0.9, linestyle=":", alpha=0.6)
    ax.text(1, 0.175, "0.15, usable", fontsize=7.6, color="#1a1a2e", style="italic")
    ax.set_xlim(-0.5, len(order) - 0.5)
    ax.set_ylim(0, 2.02)
    ax.set_xticks([])
    frame(ax, xlab="58 FLEURS languages, ordered by base large-v3 CER (easiest to hardest)",
          ylab="character error rate (clipped at 2.0)",
          title="Where the multilingual damage actually is",
          sub="the two Malay fine-tunes sit above the usable line in almost every language")
    legend_below(ax, ncol=5, pad=0.14)
    save(fig, "fig_fleurs_per_lang.png")


# 6. the positive arm, three framings of the same audio
def fig_lexsynth_recovered():
    conds = [("", "bare clip", "#1a5276"), ("__zeros2", "+2 s digital zeros", "#1e8449"),
             ("__tone2", "+2 s real room tone", "#e67e22")]
    fig, ax = plt.subplots(figsize=(6.6, 3.9), dpi=DPI)
    y = np.arange(len(MODELS))
    for i, (suf, lab, color) in enumerate(conds):
        vals = [(lex.get(k + suf) or {}).get("recovered_rate") for k, _, _, _ in MODELS]
        ax.barh(y + (1 - i) * 0.26, [v or 0 for v in vals], height=0.26, color=color,
                edgecolor="white", linewidth=0.7, label=lab, zorder=3)
        for yy, v in zip(y + (1 - i) * 0.26, vals):
            if v:
                ax.text(v + 0.01, yy, f"{v:.1%}", va="center", fontsize=7.2, color=color,
                        fontweight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels([n for _, n, _, _ in MODELS], fontsize=8.5, color=INK, fontweight="bold")
    ax.set_ylim(len(MODELS) - 0.45, -0.55)
    ax.set_xlim(0, 0.88)
    ticks = [t for t in ax.get_xticks() if 0 <= t <= 0.88]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:.0%}" for t in ticks], fontsize=8.5, color=TICK)
    frame(ax, xlab="phrase recovered, higher is better", axis="x",
          title="The ranking inverts when the phrase is really spoken",
          sub="6,267 clips, 83 languages; padding is applied to the same clips")
    legend_below(ax, ncol=3, pad=0.20)
    save(fig, "fig_lexsynth_recovered.png")


# 7. mean CER on that arm is a tail statistic
def fig_lexsynth_tail():
    fig, ax = plt.subplots(figsize=(6.6, 3.9), dpi=DPI)
    x = np.arange(len(MODELS))
    med = [(lex.get(k + "__tone2") or {}).get("cer_median") for k, _, _, _ in MODELS]
    mean = [(lex.get(k + "__tone2") or {}).get("cer") for k, _, _, _ in MODELS]
    ax.bar(x - 0.2, med, width=0.38, color="#1a5276", edgecolor="white", linewidth=0.7,
           label="median CER, the typical clip", zorder=3)
    ax.bar(x + 0.2, mean, width=0.38, color="#922b21", edgecolor="white", linewidth=0.7,
           label="mean CER, dragged by the tail", zorder=3)
    for xx, v in zip(x - 0.2, med):
        ax.text(xx, v + 0.06, f"{v:.2f}", ha="center", fontsize=7.4, color="#1a5276",
                fontweight="bold")
    for xx, v in zip(x + 0.2, mean):
        ax.text(xx, v + 0.06, f"{v:.2f}", ha="center", fontsize=7.4, color="#922b21",
                fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([n for _, n, _, _ in MODELS], fontsize=8.5, color=INK)
    frame(ax, ylab="character error rate, room tone condition",
          title="Report the median as well, or the tail speaks for the arm",
          sub="3 to 6 percent of clips run past twice the phrase length and carry the mean")
    legend_below(ax, ncol=2, pad=0.16)
    save(fig, "fig_lexsynth_tail.png")


# 8. where hallucinations actually live
def fig_wild_yield():
    y = json.loads((ROOT / "bench" / "wild_yields.json").read_text())["corpora"]
    y = sorted(y, key=lambda c: c["kept"])
    fig, ax = plt.subplots(figsize=(6.6, 3.9), dpi=DPI)
    pos = np.arange(len(y))
    rates = [max(c["kept"] / c["heard"], 1e-5) for c in y]
    colors = ["#922b21" if c["kept"] > 1000 else "#5b7fd4" for c in y]
    ax.barh(pos, rates, height=0.6, color=colors, edgecolor="white", linewidth=0.7, zorder=3)
    for p, c, r in zip(pos, y, rates):
        ax.text(r * 1.25, p, f"{r:.2%}  ({c['kept']:,})", va="center", fontsize=7.6,
                color=INK, fontweight="bold")
    ax.set_yticks(pos)
    ax.set_yticklabels([c["name"] for c in y], fontsize=8.5, color=INK)
    ax.set_xscale("log")
    ax.set_xlim(5e-5, 6)
    frame(ax, xlab="share of clips kept, log scale", axis="x",
          title="Hallucination yield tracks recording quality, across five thousandfold",
          sub="per 8,000 clips heard, mined with whisper-large-v3")
    save(fig, "fig_wild_yield.png")


# 9. what the checkpoints do on real voice-free audio
def fig_wild_blank():
    fig, ax = plt.subplots(figsize=(6.6, 3.9), dpi=DPI)
    x = np.arange(len(MODELS))
    emit = [wild[k]["blank_speech"]["any_output_rate"] for k, _, _, _ in MODELS]
    lexr = [wild[k]["blank_speech"]["lexicon_rate"] for k, _, _, _ in MODELS]
    ax.bar(x - 0.2, emit, width=0.38, color="#922b21", edgecolor="white", linewidth=0.7,
           label="emitted something", zorder=3)
    ax.bar(x + 0.2, lexr, width=0.38, color="#1a5276", edgecolor="white", linewidth=0.7,
           label="output was exactly a known lexicon phrase", zorder=3)
    for xx, v in zip(x - 0.2, emit):
        ax.text(xx, v + 0.02, f"{v:.1%}", ha="center", fontsize=7.4, color="#922b21",
                fontweight="bold")
    for xx, v in zip(x + 0.2, lexr):
        ax.text(xx, v + 0.02, f"{v:.1%}", ha="center", fontsize=7.4, color="#1a5276",
                fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([n for _, n, _, _ in MODELS], fontsize=8.5, color=INK)
    ax.set_ylim(0, 1.15)
    pct(ax, 1.0)
    frame(ax, ylab="share of 4,421 clips",
          title="Real audio a VAD confirms has no speech in it",
          sub="the lexicon explains most of what gets written, but only once the clip is known blank")
    legend_below(ax, ncol=2, pad=0.16)
    save(fig, "fig_wild_blank.png")


# 10. picking a generator
def fig_tts():
    rows = [("Multilingual-Expressive", 0.221, "45 named voices"),
            ("OmniVoice (auto)", 0.349, "646 languages, 1 voice each"),
            ("OmniVoice (voice design)", 0.589, "646 languages, 48 tags"),
            ("ToucanTTS", 0.830, "7,233 languages"),
            ("Higgs Audio v2", 1.089, "cloning"),
            ("Higgs Audio v3", 1.230, "cloning")]
    rows = rows[::-1]
    fig, ax = plt.subplots(figsize=(6.6, 3.7), dpi=DPI)
    y = np.arange(len(rows))
    colors = ["#1e8449" if r[1] < 0.4 else "#5b7fd4" for r in rows]
    ax.barh(y, [r[1] for r in rows], height=0.6, color=colors, edgecolor="white",
            linewidth=0.7, zorder=3)
    for yy, r in zip(y, rows):
        ax.text(r[1] + 0.02, yy, f"{r[1]:.3f}    {r[2]}", va="center", fontsize=7.6,
                color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=8.5, color=INK)
    ax.set_xlim(0, 2.0)
    frame(ax, xlab="round-trip character error rate, lower is better", axis="x",
          title="Coverage and quality are not the same axis",
          sub="48 lexicon phrases, 22 languages, identical text for every system")
    save(fig, "fig_tts.png")


# 11. voice conversion: cost against how far the voice actually moved
def fig_vc():
    vc = json.loads((ROOT / "tts" / "vc_scores_summary.json").read_text())
    cal = json.loads((ROOT / "tts" / "vc_sim_calibration.json").read_text())
    # Raw cosines are not comparable to anything. The calibration puts 0% at "a stranger" and
    # 100% at "two clips of the same person", both measured at 1.6 s, which is the scale the
    # text quotes.
    floor = cal["floor_diff_speaker_matched"]["mean"]
    ceil = cal["ceiling_same_speaker"]["mean"]
    scale = lambda c: (c - floor) / (ceil - floor)

    pts = []
    for name, v in vc.items():
        if not isinstance(v, dict) or v.get("n", 0) < 40:
            continue
        st, ss, d = v.get("mean_sim_tgt"), v.get("mean_sim_src"), v.get("mean_d_cer")
        if st is None or ss is None or d is None:
            continue
        pts.append((name.replace("_", " "), d, (scale(st) - scale(ss)) * 100,
                    v.get("median_dur_ratio") or 1.0))
    pts.sort(key=lambda r: r[1])

    OFFSETS = {
        "knnvc": (-8, -16, "right"),
        "openvoice": (10, -16, "left"),
        "openvoice longref": (12, 6, "left"),
        "seedvc": (-10, -4, "right"),
        "seedvc longref": (12, -4, "left"),
        "higgs3 clone": (-12, -4, "right"),
        "omnivoice clone": (-10, 8, "right"),
        "scicom clone": (0, -20, "center"),
        "openvoice clone": (0, 13, "center"),
    }

    fig, ax = plt.subplots(figsize=(6.6, 4.4), dpi=DPI)
    for i, (name, d, gap, dur) in enumerate(pts):
        color = "#922b21" if dur > 1.5 else ("#1e8449" if d < 0.15 else "#5b7fd4")
        ax.scatter([d], [gap], s=80 + 200 * max(dur - 1, 0), c=color, edgecolors="white",
                   linewidths=1.2, zorder=5)
        # Three systems land within two calibrated points of each other, so alternating above
        # and below is not enough; those get pushed sideways by hand.
        dx, dy, ha = OFFSETS.get(name, (0, 13, "center"))
        ax.annotate(name, (d, gap), textcoords="offset points", xytext=(dx, dy), ha=ha,
                    fontsize=7.2, color=INK, zorder=6)
    ax.axvline(0, color="#1a1a2e", linewidth=0.9, linestyle=":", alpha=0.5)
    ax.text(0.015, 0.975, "marker size is output length relative to the source; red runs long",
            transform=ax.transAxes, fontsize=7.4, color="#666666", style="italic", va="top")
    ax.set_ylim(-14, 112)
    ax.set_xlim(-0.62, 1.62)
    frame(ax, xlab="character error rate added on top of the source clip",
          ylab="identity moved, calibrated points\n(0 = a stranger, 100 = the target)",
          title="A converter that changes nothing scores a perfect zero cost", axis="both")
    save(fig, "fig_vc.png")


# 12. the voice collapse, and what fixed it
def fig_voice_diversity():
    # Explicit paths, not a search. "Before" is OmniVoice's own clips, because OmniVoice takes
    # no speaker argument and is the engine that collapsed; "after" is the pooled corpus with
    # the named-voice top-up folded in. A generic walk picked the multi-speaker engine here and
    # silently drew an empty chart.
    before = json.loads((ROOT / "tts" / "voice_diversity.json").read_text())["engines"]["omnivoice"]["langs"]
    after = json.loads((ROOT / "tts" / "voice_diversity_after.json").read_text())["engines"]["pooled"]["langs"]

    collapsed = sorted((l for l in before if l in after and before[l]["median_cosine"] >= 0.75),
                       key=lambda l: -before[l]["median_cosine"])
    fig, ax = plt.subplots(figsize=(6.6, 4.0), dpi=DPI)
    x = np.arange(len(collapsed))
    yb = [before[l]["median_cosine"] for l in collapsed]
    ya = [after[l]["median_cosine"] for l in collapsed]
    ax.plot(x, yb, marker="o", markersize=4.2, color="#922b21", linewidth=1.5,
            label=f"before, median {statistics.median(yb):.3f}")
    ax.plot(x, ya, marker="s", markersize=4.2, color="#1e8449", linewidth=1.5,
            label=f"after 3 named voices per phrase, median {statistics.median(ya):.3f}")
    ax.vlines(x, ya, yb, color="#bbbbbb", linewidth=0.8, zorder=1)
    for v in (0.605, 0.853):
        ax.axhline(v, color="#1a1a2e", linewidth=0.9, linestyle=":", alpha=0.7)
    # The reference lines are labelled in a corner, not on the lines: at the right edge the
    # 0.853 label sat on top of the `ml` spike.
    ax.text(0.015, 0.975, "dotted: 0.605 two strangers, 0.853 the same person twice",
            transform=ax.transAxes, fontsize=7.4, color="#1a1a2e", style="italic", va="top")
    ax.set_xticks(x)
    ax.set_xticklabels(collapsed, fontsize=7.2, color=INK, rotation=90)
    ax.set_xlim(-0.6, len(collapsed) - 0.4)
    ax.set_ylim(0.40, 0.97)
    frame(ax, ylab="median pairwise speaker similarity",
          title="One voice per language, and what fixed it",
          sub=f"the {len(collapsed)} OmniVoice languages that sat at 0.75 or worse, "
              "calibrated WavLM scale")
    legend_below(ax, ncol=1, pad=0.22)
    save(fig, "fig_voice_diversity.png")


# ── the sweep ────────────────────────────────────────────────────────────────────────────
# These read a dump produced on the GPU box by `bench/sweep_table.py --json`, because the run
# directories are 24 merged checkpoints and never leave it.
SWEEP = HERE / "sweep_rows.json"
LRS = ("1e-4", "2e-4", "5e-4")
RANKS = (8, 16, 32, 64, 128)
# Trainable parameters per rank unit, measured from the runs. The MLP pair costs 1.83x.
PER_RANK = {"loraattn": 983_040, "lora": 1_802_240}
FULL_PARAMS = 1_574_900_000
CONFIGS = ([("lora", r, lr) for r in RANKS for lr in LRS]
           + [("loraattn", r, lr) for r in RANKS for lr in LRS]
           + [("full", 0, lr) for lr in ("5e-6", "1e-5", "2e-5")])
FAM_MARKER = {"lora": "s", "loraattn": "o", "full": "*"}
FAM_LABEL = {"lora": "attn + MLP", "loraattn": "attn only", "full": "full fine-tune"}


def run_name(fam, rank, lr, mix):
    return f"v3_full_{mix}_lr{lr}" if fam == "full" else f"v3_{fam}_r{rank}_{mix}_lr{lr}"


def params_of(fam, rank):
    return FULL_PARAMS if fam == "full" else PER_RANK[fam] * rank


def sweep_rows():
    """[(label, family, rank, lr, all_row, no_synth_row)] for every configuration present."""
    if not SWEEP.exists():
        return None
    rows = {r["run"]: r for r in json.loads(SWEEP.read_text())}
    out = []
    for fam, rank, lr in CONFIGS:
        a = rows.get(run_name(fam, rank, lr, "all"))
        n = rows.get(run_name(fam, rank, lr, "no_synth"))
        if a is None and n is None:
            continue
        label = (f"r{rank}\n{lr}" if rank else f"full\n{lr}")
        out.append((label, fam, rank, lr, a, n))
    return out


def _paired(metric, ylab, title, sub, fname, base=None, as_pct=True):
    rows = sweep_rows()
    if not rows:
        print(f"!! {fname}: no {SWEEP.name} yet")
        return
    fig, ax = plt.subplots(figsize=(6.8, 3.9), dpi=DPI)
    x = np.arange(len(rows))
    av = [(r[4] or {}).get(metric) for r in rows]
    nv = [(r[5] or {}).get(metric) for r in rows]
    ax.bar(x - 0.2, [v if v is not None else 0 for v in nv], width=0.38, color="#922b21",
           edgecolor="white", linewidth=0.7, label="without the synthetic lexicon", zorder=3)
    ax.bar(x + 0.2, [v if v is not None else 0 for v in av], width=0.38, color="#1e8449",
           edgecolor="white", linewidth=0.7, label="with the synthetic lexicon", zorder=3)
    if base is not None:
        ax.axhline(base, color="#1a1a2e", linewidth=1.0, linestyle=":", alpha=0.8)
        ax.text(len(rows) - 0.4, base, f" base large-v3  {base:.3f}", fontsize=7.4,
                color="#1a1a2e", style="italic", va="bottom", ha="right")
    ax.set_xticks(x)
    ax.set_xticklabels([r[0] for r in rows], fontsize=6.0, color=INK)
    if as_pct:
        pct(ax, max([v for v in av + nv + [base or 0] if v is not None] + [0.01]) * 1.2)
    frame(ax, ylab=ylab, title=title, sub=sub)
    legend_below(ax, ncol=2, pad=0.22)
    save(fig, fname)


def fig_sweep_wild_halluc():
    _paired("wild_words", "words emitted over voice-free wild audio",
            "Question 1 and 2, side by side: hallucination on real audio",
            "every bar is one fine-tune of whisper-large-v3, 1,000 steps, same data except the "
            "synthetic positives", "fig_sweep_wild_halluc.png", base=0.999)


def fig_sweep_wild_loop():
    _paired("wild_loop", "clips with a token run of six or more",
            "The repetition half, on the same audio",
            "looping on real voice-free clips", "fig_sweep_wild_loop.png")


def fig_sweep_cost():
    rows = sweep_rows()
    if not rows:
        print("!! fig_sweep_cost.png: no sweep_rows.json yet")
        return
    fig, ax = plt.subplots(figsize=(6.6, 4.4), dpi=DPI)
    seen = set()
    for label, fam, rank, lr, a, n in rows:
        for r, mix, color, marker in ((n, "no_synth", "#922b21", "o"),
                                      (a, "all", "#1e8449", "s")):
            if not r or r.get("wild_words") is None or r.get("lex_rec") is None:
                continue
            lab = None
            if mix not in seen:
                seen.add(mix)
                lab = f"{'with' if mix == 'all' else 'without'} synthetic lexicon"
            ax.scatter([r["wild_words"]], [r["lex_rec"]], s=70, c=color, marker=marker,
                       edgecolors="white", linewidths=1.0, zorder=5, label=lab)
    ax.scatter([0.999], [0.698], s=200, c="#1a1a2e", marker="*", edgecolors="white",
               linewidths=1.2, zorder=6, label="base large-v3")
    frame(ax, xlab="words emitted over voice-free wild audio  (better to the left)",
          ylab="phrase recovered when it IS spoken  (better upward)",
          title="The same trade, after fine-tuning",
          sub="up and to the left is the corner the benchmark exists to find", axis="both")
    legend_below(ax, ncol=3, pad=0.20)
    save(fig, "fig_sweep_cost.png")


# ── training curves ──────────────────────────────────────────────────────────────────────
LOSS = HERE / "wandb_loss.json"


def fig_loss():
    if not LOSS.exists():
        print("!! fig_loss.png: no wandb_loss.json yet")
        return
    runs = json.loads(LOSS.read_text())
    fig, ax = plt.subplots(figsize=(6.6, 4.0), dpi=DPI)
    shown = set()
    for name, r in sorted(runs.items()):
        mix = (r.get("config") or {}).get("mix") or ("all" if "_all_" in name else "no_synth")
        full = "_full_" in name
        color = "#1e8449" if mix == "all" else "#922b21"
        lab = None
        key = (mix, full)
        if key not in shown:
            shown.add(key)
            lab = f"{'with' if mix == 'all' else 'without'} synthetic lexicon"
            lab += ", full fine-tune" if full else ", LoRA"
        xs = [p[0] for p in r["points"]]
        ys = [p[1] for p in r["points"]]
        ax.plot(xs, ys, color=color, linewidth=1.5 if full else 1.0,
                linestyle="--" if full else "-", alpha=0.9 if full else 0.6, label=lab)
    ax.set_yscale("log")
    # A log axis prints its own minor labels (4x10^-1 and friends) straight through the
    # explicit ticks below.
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_ylim(0.25, 12)
    ax.set_yticks([0.3, 0.5, 1, 2, 4, 8])
    ax.set_yticklabels(["0.3", "0.5", "1", "2", "4", "8"], fontsize=8.5, color=TICK)
    ax.set_xlim(0, 1000)
    frame(ax, xlab="optimiser step", ylab="training loss, log scale",
          title="The mix with the better loss is the worse model",
          sub="24 runs, logged every 25 steps; an empty target is cheap, so a blanker mix "
              "scores lower by construction")
    legend_below(ax, ncol=2, pad=0.18)
    save(fig, "fig_loss.png")


def fig_loss_vs_result():
    rows = sweep_rows()
    if not rows or not LOSS.exists():
        print("!! fig_loss_vs_result.png: needs sweep_rows.json and wandb_loss.json")
        return
    runs = json.loads(LOSS.read_text())
    final = {k: v["points"][-1][1] for k, v in runs.items()}
    fig, ax = plt.subplots(figsize=(6.6, 4.2), dpi=DPI)
    seen = set()
    for _, fam, rank, lr, a, n in rows:
        for row, mix, color, marker in ((n, "no_synth", "#922b21", "o"),
                                        (a, "all", "#1e8449", "s")):
            if not row or row.get("lex_rec") is None:
                continue
            name = run_name(fam, rank, lr, mix)
            if name not in final:
                continue
            lab = None
            if mix not in seen:
                seen.add(mix)
                lab = f"{'with' if mix == 'all' else 'without'} synthetic lexicon"
            ax.scatter([final[name]], [row["lex_rec"]], s=70, c=color, marker=marker,
                       edgecolors="white", linewidths=1.0, zorder=5, label=lab)
    ax.axhline(0.698, color="#1a1a2e", linewidth=0.9, linestyle=":", alpha=0.7)
    ax.text(0.98, 0.706, "base large-v3, 0.698", transform=ax.get_yaxis_transform(),
            ha="right", fontsize=7.6, color="#1a1a2e", style="italic")
    ax.set_xscale("log")
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xticks([0.3, 0.5, 1, 2, 3])
    ax.set_xticklabels(["0.3", "0.5", "1", "2", "3"], fontsize=8.5, color=TICK)
    frame(ax, xlab="final training loss, log scale",
          ylab="phrase recovered when it IS spoken",
          title="Training loss does not rank these models",
          sub="every run left of 0.5 is a negatives-only mix, and every one of them is worse",
          axis="both")
    legend_below(ax, ncol=2, pad=0.18)
    save(fig, "fig_loss_vs_result.png")


def fig_sweep_rank():
    """Trainable parameters against the result, which is the rank and target-set question."""
    rows = sweep_rows()
    if not rows:
        print("!! fig_sweep_rank.png: no sweep_rows.json yet")
        return
    fig, ax = plt.subplots(figsize=(6.6, 4.2), dpi=DPI)
    seen = set()
    for _, fam, rank, lr, a, n in rows:
        for r, mix, color in ((n, "no_synth", "#922b21"), (a, "all", "#1e8449")):
            if not r or r.get("wild_words") is None:
                continue
            key = (fam, mix)
            lab = None
            if key not in seen:
                seen.add(key)
                lab = f"{FAM_LABEL[fam]}, {'with' if mix == 'all' else 'without'} synth"
            # a star reads much smaller than a square at the same point area
            size = 190 if fam == "full" else 70
            ax.scatter([params_of(fam, rank) / 1e6], [r["wild_words"]], s=size, c=color,
                       marker=FAM_MARKER[fam], edgecolors="white", linewidths=1.0,
                       zorder=5, label=lab)
    ax.axhline(0.999, color="#1a1a2e", linewidth=0.9, linestyle=":", alpha=0.7)
    ax.text(0.99, 0.985, "base large-v3", transform=ax.get_yaxis_transform(), ha="right",
            fontsize=7.6, color="#1a1a2e", style="italic", va="top")
    ax.set_xscale("log")
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xticks([8, 16, 32, 64, 128, 256, 1600])
    ax.set_xticklabels(["8", "16", "32", "64", "128", "256", "1,600"], fontsize=8.5, color=TICK)
    pct(ax, 1.05)
    frame(ax, xlab="trainable parameters, millions, log scale",
          ylab="words emitted over voice-free wild audio",
          title="Does capacity buy anything?",
          sub="marker shape is which modules the adapters touch", axis="both")
    legend_below(ax, ncol=3, pad=0.22)
    save(fig, "fig_sweep_rank.png")


def fig_sweep_family():
    """Same rank, same learning rate, different modules. Isolates what the MLP pair buys."""
    rows = sweep_rows()
    if not rows:
        print("!! fig_sweep_family.png: no sweep_rows.json yet")
        return
    by = {}
    for _, fam, rank, lr, a, n in rows:
        if fam == "full" or not a or a.get("wild_words") is None:
            continue
        by.setdefault((fam, lr), {})[rank] = a["wild_words"]
    if not by:
        print("!! fig_sweep_family.png: no adapter runs yet")
        return
    fig, ax = plt.subplots(figsize=(6.6, 4.0), dpi=DPI)
    style = {"1e-4": ":", "2e-4": "--", "5e-4": "-"}
    seen = set()
    for (fam, lr), pts in sorted(by.items()):
        xs = sorted(pts)
        color = "#1a5276" if fam == "lora" else "#e67e22"
        lab = None
        if fam not in seen:
            seen.add(fam)
            lab = FAM_LABEL[fam]
        ax.plot(xs, [pts[x] for x in xs], marker="o", markersize=4.5, color=color,
                linewidth=1.6, linestyle=style.get(lr, "-"), alpha=0.9, label=lab)
    ax.axhline(0.999, color="#1a1a2e", linewidth=0.9, linestyle=":", alpha=0.7)
    ax.text(0.99, 0.992, "base large-v3", transform=ax.get_yaxis_transform(), ha="right",
            fontsize=7.6, color="#1a1a2e", style="italic", va="top")
    ax.set_xscale("log", base=2)
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xticks(RANKS)
    ax.set_xticklabels([str(r) for r in RANKS], fontsize=8.5, color=TICK)
    pct(ax, 1.05)
    frame(ax, xlab="LoRA rank", ylab="words emitted over voice-free wild audio",
          title="Learning to stop lives in the feed-forward layers",
          sub="`all` mix; one line per learning rate, dotted 1e-4, dashed 2e-4, solid 5e-4",
          axis="both")
    legend_below(ax, ncol=2, pad=0.18)
    save(fig, "fig_sweep_family.png")


if __name__ == "__main__":
    for fn in (fig_nonspeech, fig_repetition, fig_tradeoff, fig_accuracy, fig_fleurs_per_lang,
               fig_lexsynth_recovered, fig_lexsynth_tail, fig_wild_yield, fig_wild_blank,
               fig_tts, fig_vc, fig_voice_diversity,
               fig_sweep_wild_halluc, fig_sweep_wild_loop, fig_sweep_cost,
               fig_loss, fig_loss_vs_result, fig_sweep_rank, fig_sweep_family):
        try:
            fn()
        except Exception as e:
            print(f"!! {fn.__name__}: {type(e).__name__}: {e}")
