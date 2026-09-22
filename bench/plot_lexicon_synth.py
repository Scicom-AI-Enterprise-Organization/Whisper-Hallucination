"""
plot_lexicon_synth.py
─────────────────────
Generates a PNG of the false-positive side of the benchmark: the same five checkpoints on
`lexicon_synth`, where the hallucination phrase is ACTUALLY SPOKEN, under three framings of
the same audio.

The negative arms ask what a model invents over silence. This asks what it does when the
phrase is real — and whether wrapping it in silence changes the answer, because production
audio arrives as VAD chunks with a noise floor around the speech, not as a bare 0.9 s clip.

  bare     the clip as published. Whisper zero-pads every input to 30 s anyway, so this is
           already 0.9 s of speech inside ~29 s of digital zeros
  zeros    2 s of digital zeros prepended and appended — the CONTROL. Trailing zeros change
           nothing by construction; only the leading 2 s shifts where the speech begins
  tone     2 s of real room tone each side, drawn from the `silence` arm — a noise floor the
           model is known to hallucinate over

Numbers come from bench/lexicon_synth_scores.json (bench/score_lexicon_synth.py), so a
re-score updates the figure with no edit here.
"""

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT = Path(__file__).resolve().parent.parent
SCORES = ROOT / 'bench' / 'lexicon_synth_scores.json'
BENCH = ROOT / 'bench' / 'scores.json'

MODELS = [
    ('whisper-large-v2',                    'whisper-large-v2'),
    ('whisper-large-v3',                    'whisper-large-v3'),
    ('whisper-large-v3-turbo',              'whisper-large-v3-turbo'),
    ('malaysian-whisper-large-v2',          'malaysian-whisper-v2'),
    ('Malaysian-whisper-large-v3-turbo-v3', 'Malaysian-turbo-v3'),
]
HIGHLIGHT = {'malaysian-whisper-v2', 'Malaysian-turbo-v3'}

CONDITIONS = [
    ('',         'bare clip',        '#1a5276'),
    ('__zeros2', '+2 s zeros',       '#1e8449'),
    ('__tone2',  '+2 s room tone',   '#e67e22'),
]

# (metric, title, subtitle, formatter). The MEDIAN length ratio is 1.00 for every model in
# every condition — the over-generation lives entirely in a tail, which is why `over_2x_rate`
# is plotted instead and the mean CER it drags is annotated rather than charted (one model at
# 3.7 would flatten the other four to invisible slivers).
PANELS = [
    ('recovered_rate', 'Phrase recovered  ↑ higher is better',
     'the clip says the phrase; did it come back?', 'pct'),
    ('cer_median', 'Median CER  ↓ lower is better',
     'the typical clip, unmoved by the runaway tail', 'num'),
    ('over_2x_rate', 'Output over 2× the phrase  ↓ lower is better',
     'the tail that drags the mean — text the model added', 'pct'),
    ('empty_rate', 'Emitted nothing  ↓ lower is better',
     'real speech deleted outright', 'pct'),
]

STYLE = dict(
    bg_color        = '#ffffff',
    grid_color      = '#e0e0e0',
    ours_color      = '#203882',
    title_color     = '#1a1a2e',
    label_color     = '#333333',
    tick_color      = '#444444',
    caption_color   = '#666666',
    anno_bg         = '#f5f5f5',
    title_fontsize  = 11,
    axis_fontsize   = 9.5,
    tick_fontsize   = 8.5,
    value_fontsize  = 7.4,
    dpi             = 200,
    output_file     = str(ROOT / 'bench' / 'lexicon_synth_results.png'),
)


def main():
    s = STYLE
    sc = json.loads(SCORES.read_text())
    bench = json.loads(BENCH.read_text()) if BENCH.exists() else {}

    fig, axes = plt.subplots(1, len(PANELS), figsize=(19.5, 5.8), dpi=s['dpi'])
    fig.patch.set_facecolor(s['bg_color'])
    fig.subplots_adjust(wspace=0.16, left=0.105, right=0.988, top=0.73, bottom=0.12)

    names = [disp for _, disp in MODELS]
    y = np.arange(len(MODELS))
    h = 0.26

    for ax, (metric, title, subtitle, fmt) in zip(axes, PANELS):
        ax.set_facecolor(s['bg_color'])
        ax.grid(True, axis='x', color=s['grid_color'], linewidth=0.8, linestyle='--', alpha=0.6)
        ax.set_axisbelow(True)

        vals_all = []
        for ci, (suffix, clabel, color) in enumerate(CONDITIONS):
            vals = []
            for key, _ in MODELS:
                row = sc.get(f"{key}{suffix}") or {}
                vals.append(row.get(metric))
            vals_all += [v for v in vals if v is not None]
            offset = (1 - ci) * h
            ax.barh(y + offset, [v or 0 for v in vals], height=h, color=color,
                    edgecolor='white', linewidth=0.7, zorder=3,
                    label=clabel if ax is axes[0] else None)

        span = max(vals_all) if vals_all else 1.0
        for ci, (suffix, _, color) in enumerate(CONDITIONS):
            offset = (1 - ci) * h
            for i, (key, _) in enumerate(MODELS):
                v = (sc.get(f"{key}{suffix}") or {}).get(metric)
                if v is None:
                    continue
                label = f'{v:.3f}' if fmt == 'num' else f'{v:.1%}'
                ax.text(v + span * 0.015, i + offset, label,
                        va='center', ha='left', fontsize=s['value_fontsize'],
                        color=color, fontweight='bold')

        ax.set_yticks(y)
        # Rows are aligned across panels, so only the leftmost panel carries the model names;
        # repeating them put a label on top of the next panel's bars.
        first = ax is axes[0]
        ax.set_yticklabels(names if first else [], fontsize=s['tick_fontsize'],
                           fontweight='bold')
        if first:
            for tick, name in zip(ax.get_yticklabels(), names):
                tick.set_color(s['ours_color'] if name in HIGHLIGHT else s['label_color'])
        ax.set_ylim(len(MODELS) - 0.45, -0.55)
        ax.set_xlim(0, span * 1.26)
        ax.tick_params(colors=s['tick_color'], labelsize=s['tick_fontsize'], length=0)
        if fmt == 'pct':
            # Pin the locator before relabelling, or matplotlib warns and may mislabel.
            ticks = [tk for tk in ax.get_xticks() if 0 <= tk <= span * 1.26]
            ax.set_xticks(ticks)
            ax.set_xticklabels([f'{v:.0%}' for v in ticks],
                               fontsize=s['tick_fontsize'], color=s['tick_color'])
        ax.set_title(title, fontsize=s['title_fontsize'], color=s['title_color'],
                     fontweight='bold', pad=16, loc='left')
        ax.text(0, 1.015, subtitle, transform=ax.transAxes, fontsize=7.8,
                color=s['caption_color'], style='italic', va='bottom')
        for spine in ('top', 'right'):
            ax.spines[spine].set_visible(False)
        for spine in ('left', 'bottom'):
            ax.spines[spine].set_color('#cccccc')

    worst = max(sc.values(), key=lambda v: v.get("cer") or 0)
    base = sc.get(f'{worst["model"]}') or {}
    disp = dict(MODELS).get(worst["model"], worst["model"])
    # Top-right of the median-CER panel: the three base checkpoints' bars end near 0.14, so
    # this corner is empty. Lower placements sat on malaysian-whisper-v2's value labels.
    axes[1].text(0.97, 0.97,
                 f'mean CER hides the tail:\n{disp}  '
                 f'{base.get("cer", 0):.2f} bare → {worst["cer"]:.2f} {worst["condition"]}',
                 transform=axes[1].transAxes, ha='right', va='top',
                 fontsize=7.4, color='#922b21', fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.3', fc=s['anno_bg'], ec='#922b21',
                           alpha=0.95, linewidth=0.9), zorder=6)

    # The contrast that gives the arm its meaning: on the SAME room tone with no speech in it,
    # every OpenAI checkpoint emits something on 100% of clips.
    sil = {m: (bench.get(k, {}).get('silence', {}) or {}).get('hallucination_rate_any_output')
           for k, m in MODELS}
    if any(v is not None for v in sil.values()):
        txt = '  •  '.join(f'{m} {v:.0%}' for m, v in sil.items() if v is not None)
        fig.text(0.5, 0.845,
                 'For contrast — on room tone with NO speech, these same checkpoints emit '
                 f'something on: {txt}',
                 ha='center', fontsize=8.0, color=s['ours_color'], fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.35', fc=s['anno_bg'], ec=s['ours_color'],
                           alpha=0.95, linewidth=0.9))

    handles = [mpatches.Patch(facecolor=c, edgecolor='white', label=l)
               for _, l, c in CONDITIONS]
    fig.legend(handles=handles, fontsize=8.5, facecolor='#f5f5f5', edgecolor='#cccccc',
               labelcolor=s['label_color'], ncol=3, framealpha=0.95,
               loc='upper center', bbox_to_anchor=(0.5, 0.925))

    n = (sc.get('whisper-large-v3') or {}).get('n', 0)
    fig.suptitle('The false-positive side — the hallucination phrase is actually spoken',
                 fontsize=13.5, color=s['title_color'], fontweight='bold', y=0.985)
    fig.text(0.5, 0.945,
             f'`lexicon_synth` test split, {n:,} synthetic clips, 83 languages  •  greedy, '
             'auto language  •  padding is applied to the SAME clips, so each triple is a '
             'within-clip comparison',
             ha='center', fontsize=8.5, color=s['caption_color'], style='italic')

    out = s['output_file']
    plt.savefig(out, dpi=s['dpi'], bbox_inches='tight',
                facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close(fig)
    print(f'Saved → {out}')


if __name__ == '__main__':
    main()
