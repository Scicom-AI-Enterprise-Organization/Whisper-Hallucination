"""
plot_reduplication.py
─────────────────────
Generates a PNG of small multiples showing what actually triggers repetition runaway on the
`reduplication` arm: failure rate per model as a function of each stimulus knob the arm was
built with — the unit pattern, how many times it repeats, how fast, and how much silence
follows it.

Numbers are read from bench/reduplication_profile.json (built on the box by
bench/reduplication_profile.py from the per-clip results), so a re-run updates the figure
with no edit here.

To add a new model:   benchmark it; profile it; add a row to MODELS with its display style.
To change style:      edit STYLE.
"""

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

ROOT = Path(__file__).resolve().parent.parent

# ══════════════════════════════════════════════════════════════════════════════
#  MODELS  (key in reduplication_profile.json → display name, colour, linestyle, marker)
#  The two Malaysian fine-tunes are the comparison being made, so they share the house
#  navy and differ by line style; the checkpoints they came from get lighter colours.
# ══════════════════════════════════════════════════════════════════════════════
MODELS = [
    ('whisper-large-v2',                    'whisper-large-v2',       '#95a5a6', ':',  'v'),
    ('whisper-large-v3',                    'whisper-large-v3',       '#1e8449', '-.', '^'),
    ('whisper-large-v3-turbo',              'whisper-large-v3-turbo', '#e67e22', '--', 'D'),
    ('malaysian-whisper-large-v2',          'malaysian-whisper-v2',   '#203882', '--', 's'),
    ('Malaysian-whisper-large-v3-turbo-v3', 'Malaysian-turbo-v3',     '#203882', '-',  'o'),
]
HIGHLIGHT = {'malaysian-whisper-v2', 'Malaysian-turbo-v3'}

# ══════════════════════════════════════════════════════════════════════════════
#  KNOBS  ── columns, left → right. (json key, axis label, tick order, tick labels)
#  `pattern` is categorical, so it is drawn as a dot plot rather than a line.
# ══════════════════════════════════════════════════════════════════════════════
KNOBS = [
    ('pattern',        'what is repeated',
        ['click', 'cv', 'vowel', 'laugh'], ['click', 'syllable', 'vowel', 'laugh']),
    ('n_repeats',      'how many repeats',
        ['3', '4', '5', '6', '8', '12'], ['3', '4', '5', '6', '8', '12']),
    ('rate_hz',        'repeat rate  (Hz)',
        ['3.0', '5.0', '7.0', '9.0'], ['3', '5', '7', '9']),
    ('tail_silence_s', 'trailing silence  (s)',
        ['0.0', '2.0', '8.0'], ['0', '2', '8']),
]

# ══════════════════════════════════════════════════════════════════════════════
#  ROWS  ── (json metric, row title, y-axis ceiling)
# ══════════════════════════════════════════════════════════════════════════════
ROWS = [
    ('runaway_rate', 'runaway\n(emits > 1.5× the spoken repeats)', 0.62),
    ('empty_rate',   'emits nothing\n(the repeated speech is deleted)', 1.02),
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
    line_width      = 1.9,
    ours_line_width = 2.4,
    marker_size     = 6.5,
    title_fontsize  = 10.5,
    axis_fontsize   = 9.5,
    tick_fontsize   = 8.5,
    anno_fontsize   = 7.6,
    dpi             = 200,
    output_file     = str(ROOT / 'bench' / 'reduplication_profile.png'),
)

# Annotations: (row metric, knob key, text, xy in axis fraction, ha)
ANNOTATIONS = [
    ('runaway_rate', 'pattern',
     'laughter runs away on\nhalf of the fine-tunes’ clips;\nclicks never do', (0.05, 0.93), 'left'),
    ('runaway_rate', 'n_repeats',
     'more repeats → more\nrunaway, every model', (0.05, 0.93), 'left'),
    ('runaway_rate', 'tail_silence_s',
     'silence after the speech\ndoes not trigger it — it\nlowers it for the base models', (0.05, 0.93), 'left'),
    ('empty_rate', 'pattern',
     'malaysian-v2 stays silent\non 96% of clicks; turbo-v3\non 60% of everything', (0.97, 0.93), 'right'),
    ('empty_rate', 'rate_hz',
     'faster repeats → turbo-v3\nmore often decides it\nwas not speech', (0.05, 0.93), 'left'),
]


def draw_panel(ax, prof, metric, knob, ticks, tick_labels, ymax, s, first_col, top_row):
    ax.set_facecolor(s['bg_color'])
    ax.grid(True, axis='y', color=s['grid_color'], linewidth=0.8, linestyle='--', alpha=0.6)
    ax.set_axisbelow(True)
    x = np.arange(len(ticks))
    categorical = knob == 'pattern'

    for key, name, color, ls, marker in MODELS:
        by = prof.get(key, {}).get('by', {}).get(knob, {})
        y = [by.get(t, {}).get(metric, np.nan) for t in ticks]
        ours = name in HIGHLIGHT
        lw = s['ours_line_width'] if ours else s['line_width']
        if categorical:
            ax.plot(x, y, linestyle='none', marker=marker, color=color,
                    markersize=s['marker_size'] + (1.0 if ours else 0),
                    markeredgecolor='white', markeredgewidth=0.7, zorder=4 if ours else 3)
        else:
            ax.plot(x, y, linestyle=ls, marker=marker, color=color, linewidth=lw,
                    markersize=s['marker_size'], markeredgecolor='white', markeredgewidth=0.7,
                    zorder=4 if ours else 3)

    ax.set_xlim(-0.45, len(ticks) - 0.55)
    ax.set_ylim(0, ymax)
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels, fontsize=s['tick_fontsize'], color=s['tick_color'])
    ax.set_yticks(np.arange(0, ymax, 0.2 if ymax > 0.8 else 0.1))
    ax.set_yticklabels([f'{v:.0%}' for v in ax.get_yticks()],
                       fontsize=s['tick_fontsize'], color=s['tick_color'])
    ax.tick_params(colors=s['tick_color'], length=0)
    if not first_col:
        ax.set_yticklabels([])
    if top_row:
        ax.set_title(dict(KNOB_LABELS)[knob], fontsize=s['title_fontsize'],
                     color=s['title_color'], fontweight='bold', pad=8)
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)
    for spine in ('left', 'bottom'):
        ax.spines[spine].set_color('#cccccc')

    for m, k, text, xy, ha in ANNOTATIONS:
        if m == metric and k == knob:
            ax.text(xy[0], xy[1], text, transform=ax.transAxes, ha=ha, va='top',
                    fontsize=s['anno_fontsize'], color=s['ours_color'], fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', fc=s['anno_bg'], ec=s['ours_color'],
                              alpha=0.95, linewidth=0.9), zorder=6)


KNOB_LABELS = [(k, label) for k, label, _, _ in KNOBS]


def main():
    s = STYLE
    prof = json.loads((ROOT / 'bench' / 'reduplication_profile.json').read_text())
    n = next(iter(prof.values()))['n']

    fig, axes = plt.subplots(len(ROWS), len(KNOBS), figsize=(16.5, 7.8), dpi=s['dpi'])
    fig.patch.set_facecolor(s['bg_color'])
    fig.subplots_adjust(wspace=0.14, hspace=0.42, left=0.065, right=0.985, top=0.80, bottom=0.08)

    for r, (metric, row_title, ymax) in enumerate(ROWS):
        for c, (knob, _, ticks, tick_labels) in enumerate(KNOBS):
            draw_panel(axes[r, c], prof, metric, knob, ticks, tick_labels, ymax, s,
                       first_col=(c == 0), top_row=(r == 0))
        axes[r, 0].set_ylabel(row_title, fontsize=s['axis_fontsize'], color=s['title_color'],
                              fontweight='bold', labelpad=10)

    handles = []
    for _, name, color, ls, marker in MODELS:
        ours = name in HIGHLIGHT
        handles.append(mlines.Line2D([], [], color=color, linestyle=ls, marker=marker,
                                     linewidth=s['ours_line_width'] if ours else s['line_width'],
                                     markersize=s['marker_size'], markeredgecolor='white',
                                     label=name))
    leg = fig.legend(handles=handles, fontsize=8.5, facecolor='#f5f5f5', edgecolor='#cccccc',
                     ncol=len(MODELS), framealpha=0.95, loc='upper center',
                     bbox_to_anchor=(0.5, 0.905))
    for txt in leg.get_texts():
        txt.set_color(s['ours_color'] if txt.get_text() in HIGHLIGHT else s['label_color'])
        if txt.get_text() in HIGHLIGHT:
            txt.set_fontweight('bold')

    fig.suptitle('What triggers the runaway — the reduplication arm sliced by its stimulus knobs',
                 fontsize=13.5, color=s['title_color'], fontweight='bold', y=0.975)
    fig.text(0.5, 0.925,
             f'{n:,} clips = 4 unit patterns × 6 repeat counts × 4 rates × 3 tail silences  •  '
             'greedy decoding  •  the runaway is driven by what is repeated and how often, '
             'not by the silence after it',
             ha='center', fontsize=8.5, color=s['caption_color'], style='italic')

    out = s['output_file']
    plt.savefig(out, dpi=s['dpi'], bbox_inches='tight',
                facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close(fig)
    print(f'Saved → {out}')


if __name__ == '__main__':
    main()
