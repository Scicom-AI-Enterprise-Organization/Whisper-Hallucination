"""
plot_wild.py
────────────
Generates a PNG of the wild arm: real recordings that made a model hallucinate or loop,
as opposed to the built stimuli the other eight arms use.

Three panels, because the arm answers three separate questions:

  left    where the failures live. Yield per 8,000 clips heard, by corpus, on a log axis
          because the spread is five thousandfold -- AudioSet 54.7%, curated read 0.01%.
  middle  what each checkpoint does on audio a VAD confirms has NO speech. Correct output is
          nothing, so the bar is the hallucination rate on real audio.
  right   whether a text blocklist could separate the human verdicts on HALAS. If it could,
          the two bars per model would differ.

Numbers come from bench/wild_scores.json (bench/score_wild.py) and bench/wild_yields.json.
"""

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT = Path(__file__).resolve().parent.parent
SCORES = ROOT / 'bench' / 'wild_scores.json'
YIELDS = ROOT / 'bench' / 'wild_yields.json'

MODELS = [
    ('whisper-large-v2',                    'whisper-large-v2',       False),
    ('whisper-large-v3',                    'whisper-large-v3',       False),
    ('whisper-large-v3-turbo',              'whisper-large-v3-turbo', False),
    ('malaysian-whisper-large-v2',          'malaysian-whisper-v2',   True),
    ('Malaysian-whisper-large-v3-turbo-v3', 'Malaysian-turbo-v3',     True),
]

STYLE = dict(
    bg_color      = '#ffffff',
    grid_color    = '#e0e0e0',
    ours_color    = '#203882',
    bar_color     = '#1a5276',
    alt_color     = '#e67e22',
    good_color    = '#1e8449',
    bad_color     = '#922b21',
    title_color   = '#1a1a2e',
    label_color   = '#333333',
    tick_color    = '#444444',
    caption_color = '#666666',
    anno_bg       = '#f5f5f5',
    title_size    = 11.5,
    tick_size     = 8.5,
    value_size    = 7.8,
    dpi           = 200,
    output_file   = str(ROOT / 'bench' / 'wild_results.png'),
)


def main():
    s = STYLE
    sc = json.loads(SCORES.read_text())
    yl = json.loads(YIELDS.read_text())["corpora"]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18.5, 6.2), dpi=s['dpi'],
                                        gridspec_kw=dict(width_ratios=[1.15, 1.0, 1.15],
                                                         wspace=0.30))
    fig.patch.set_facecolor(s['bg_color'])
    fig.subplots_adjust(left=0.115, right=0.985, top=0.76, bottom=0.13)

    # ── left: yield by corpus, log axis ──────────────────────────────────────────────
    ax1.set_facecolor(s['bg_color'])
    ax1.grid(True, axis='x', color=s['grid_color'], linewidth=0.8, linestyle='--', alpha=0.6)
    ax1.set_axisbelow(True)
    names = [c["name"] for c in yl]
    rates = [100 * c["kept"] / c["heard"] for c in yl]
    y = np.arange(len(names))
    colors = [s['bad_color'] if r > 10 else (s['alt_color'] if r > 1 else s['bar_color'])
              for r in rates]
    ax1.barh(y, rates, color=colors, edgecolor='white', linewidth=0.8, height=0.62, zorder=3)
    for i, (r, c) in enumerate(zip(rates, yl)):
        ax1.text(r * 1.15, i, f'{r:.2f}%  ({c["kept"]:,})', va='center', ha='left',
                 fontsize=s['value_size'], color=s['label_color'], fontweight='bold')
    ax1.set_xscale('log')
    ax1.set_xlim(0.008, 400)
    ax1.set_yticks(y)
    ax1.set_yticklabels([f'{c["name"]}\n{c["detail"]}' for c in yl],
                        fontsize=7.6, color=s['label_color'])
    ax1.invert_yaxis()
    ax1.set_xlabel('clips kept per 8,000 heard  (log scale)', fontsize=9.5,
                   color=s['tick_color'], labelpad=6)
    ax1.tick_params(colors=s['tick_color'], labelsize=s['tick_size'], length=0)
    ax1.set_title('Where the failures live', fontsize=s['title_size'],
                  color=s['title_color'], fontweight='bold', pad=10, loc='left')
    ax1.text(0.98, 0.04, 'noisy YouTube hallucinates\n5,000× more than curated read speech',
             transform=ax1.transAxes, ha='right', va='bottom', fontsize=8.0,
             color=s['bad_color'], fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.35', fc=s['anno_bg'], ec=s['bad_color'],
                       alpha=0.95, linewidth=0.9))
    for sp in ('top', 'right'):
        ax1.spines[sp].set_visible(False)

    # ── middle: any-output on VAD-confirmed voice-free clips ─────────────────────────
    ax2.set_facecolor(s['bg_color'])
    ax2.grid(True, axis='y', color=s['grid_color'], linewidth=0.8, linestyle='--', alpha=0.6)
    ax2.set_axisbelow(True)
    labels, vals, n_clips = [], [], 0
    for key, disp, ours in MODELS:
        b = (sc.get(key) or {}).get('blank_speech') or {}
        if not b:
            continue
        labels.append(disp)
        vals.append(b['any_output_rate'])
        n_clips = b['n']
    x = np.arange(len(labels))
    bar_colors = [s['ours_color'] if l.lower().startswith('malaysian') else s['bar_color']
                  for l in labels]
    ax2.bar(x, vals, color=bar_colors, edgecolor='white', linewidth=0.8, width=0.62, zorder=3)
    for i, v in enumerate(vals):
        ax2.text(i, v + 0.02, f'{v:.1%}', ha='center', va='bottom',
                 fontsize=s['value_size'] + 0.6, color=s['label_color'], fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=7.6, rotation=20, ha='right', color=s['label_color'])
    ax2.set_ylim(0, 1.14)
    ax2.set_yticks(np.arange(0, 1.01, 0.2))
    ax2.set_yticklabels([f'{v:.0%}' for v in np.arange(0, 1.01, 0.2)],
                        fontsize=s['tick_size'], color=s['tick_color'])
    ax2.tick_params(colors=s['tick_color'], length=0)
    ax2.set_title(f'Emits words over real silence  ↓ lower is better',
                  fontsize=s['title_size'], color=s['title_color'], fontweight='bold',
                  pad=24, loc='left')
    ax2.text(0, 1.015, f'{n_clips:,} clips a VAD confirms contain no speech — correct output '
             f'is nothing', transform=ax2.transAxes, fontsize=7.8,
             color=s['caption_color'], style='italic', va='bottom')
    for sp in ('top', 'right'):
        ax2.spines[sp].set_visible(False)

    # ── right: can a blocklist tell the human verdicts apart? ────────────────────────
    ax3.set_facecolor(s['bg_color'])
    ax3.grid(True, axis='y', color=s['grid_color'], linewidth=0.8, linestyle='--', alpha=0.6)
    ax3.set_axisbelow(True)
    labels, hall, clean = [], [], []
    for key, disp, ours in MODELS:
        h = (sc.get(key) or {}).get('halas_hallucination') or {}
        c = (sc.get(key) or {}).get('halas_clean') or {}
        if not h or not c:
            continue
        labels.append(disp)
        hall.append(h['lexicon_rate'])
        clean.append(c['lexicon_rate'])
    x = np.arange(len(labels))
    w = 0.36
    ax3.bar(x - w/2, hall, width=w, color=s['bad_color'], edgecolor='white', linewidth=0.8,
            label='humans said: hallucinated', zorder=3)
    ax3.bar(x + w/2, clean, width=w, color=s['good_color'], edgecolor='white', linewidth=0.8,
            label='humans said: clean', zorder=3)
    for i, (a, b) in enumerate(zip(hall, clean)):
        ax3.text(i - w/2, a + 0.008, f'{a:.1%}', ha='center', va='bottom', fontsize=7.2,
                 color=s['bad_color'], fontweight='bold')
        ax3.text(i + w/2, b + 0.008, f'{b:.1%}', ha='center', va='bottom', fontsize=7.2,
                 color=s['good_color'], fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels, fontsize=7.6, rotation=20, ha='right', color=s['label_color'])
    ax3.set_ylim(0, max(hall + clean) * 1.32)
    ticks = [tk for tk in ax3.get_yticks() if 0 <= tk <= max(hall + clean) * 1.32]
    ax3.set_yticks(ticks)   # pin before relabelling, or matplotlib warns and may mislabel
    ax3.set_yticklabels([f'{v:.0%}' for v in ticks], fontsize=s['tick_size'],
                        color=s['tick_color'])
    ax3.tick_params(colors=s['tick_color'], length=0)
    ax3.set_title('Could a blocklist tell these apart?', fontsize=s['title_size'],
                  color=s['title_color'], fontweight='bold', pad=24, loc='left')
    ax3.text(0, 1.015, 'share of outputs that are a known hallucination phrase, on HALAS',
             transform=ax3.transAxes, fontsize=7.8, color=s['caption_color'],
             style='italic', va='bottom')
    ax3.legend(fontsize=8.0, facecolor='#f5f5f5', edgecolor='#cccccc', framealpha=0.95,
               loc='upper right')
    if hall and clean:
        gap = hall[1] - clean[1] if len(hall) > 1 else hall[0] - clean[0]
        ax3.text(0.02, 0.86, f'large-v3: {gap*100:.1f} points apart.\nNot a detector.',
                 transform=ax3.transAxes, ha='left', va='top', fontsize=8.0,
                 color=s['bad_color'], fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.35', fc=s['anno_bg'], ec=s['bad_color'],
                           alpha=0.95, linewidth=0.9))
    for sp in ('top', 'right'):
        ax3.spines[sp].set_visible(False)

    fig.suptitle('The wild arm — real audio that actually made a model fail',
                 fontsize=13.5, color=s['title_color'], fontweight='bold', y=0.965)
    fig.text(0.5, 0.885,
             'mined from streamed corpora with no human labelling (token run ≥ 6, or words '
             'over VAD-confirmed silence), plus 3,607 Earnings-22 clips carrying HALAS human '
             'span annotations',
             ha='center', fontsize=8.5, color=s['caption_color'], style='italic')

    out = s['output_file']
    plt.savefig(out, dpi=s['dpi'], bbox_inches='tight',
                facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close(fig)
    print(f'Saved → {out}')


if __name__ == '__main__':
    main()
