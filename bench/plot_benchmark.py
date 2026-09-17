"""
plot_benchmark.py
─────────────────
Generates a PNG heatmap (models × arms) and a paired-bar chart showing how much of each
model's WER is caused by repetition runaway rather than by mis-transcription.

Numbers are read from bench/scores.json and bench/loop_decomposition.json rather than
hardcoded, so a re-run of the benchmark updates the figures with no edit here.

To add a new model:   benchmark it; it appears automatically (order below pins the rows).
To change style:      edit the STYLE dict.
"""

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parent.parent

# ══════════════════════════════════════════════════════════════════════════════
#  MODELS  (row order — display name → key in scores.json)
# ══════════════════════════════════════════════════════════════════════════════
MODELS = [
    ('whisper-large-v2',        'whisper-large-v2'),
    ('whisper-large-v3',        'whisper-large-v3'),
    ('whisper-large-v3-turbo',  'whisper-large-v3-turbo'),
    ('malaysian-whisper-v2',    'malaysian-whisper-large-v2'),
    ('Malaysian-turbo-v3',      'Malaysian-whisper-large-v3-turbo-v3'),
]

# ══════════════════════════════════════════════════════════════════════════════
#  TABLES  ── list of heatmaps to render top → bottom
#  Each column is (label, arm, metric-key). All columns in a table share a scale,
#  so only metrics measured in the same units belong together.
# ══════════════════════════════════════════════════════════════════════════════
TABLES = [
    dict(
        title    = 'Hallucination on non-speech  (reference is the empty string)',
        subtitle = '↓  lower is better',
        columns  = [('silence',        'silence',   'hallucination_rate'),
                    ('music',          'music',     'hallucination_rate'),
                    ('nonspeech',      'nonspeech', 'hallucination_rate'),
                    ('silence\nany',   'silence',   'hallucination_rate_any_output'),
                    ('music\nany',     'music',     'hallucination_rate_any_output'),
                    ('nonspeech\nany', 'nonspeech', 'hallucination_rate_any_output')],
        cmap_colors = ['#1a5276', '#1e8449', '#f9e79f', '#e67e22', '#922b21'],
        vmin = 0.0, vmax = 1.0, fmt = '{:.3f}',
    ),
    dict(
        title    = 'Word error rate  (what a mitigation must not break)',
        subtitle = '↓  lower is better',
        columns  = [('librispeech',      'librispeech_test_clean', 'wer'),
                    ('genuine',          'genuine',                'wer'),
                    ('genuine\nisolated','genuine_isolated',       'wer'),
                    ('speech\nin noise', 'speech_in_noise',        'wer')],
        cmap_colors = ['#1a5276', '#1e8449', '#f9e79f', '#e67e22', '#922b21'],
        vmin = 0.0, vmax = 0.7, fmt = '{:.3f}',
    ),
    dict(
        title    = 'Repetition behaviour on `reduplication`  (1,440 clips of known repeat counts)',
        subtitle = '↓  lower is better — except `emits nothing`, where both extremes are wrong',
        columns  = [('runaway\n>1.5×', 'reduplication', 'runaway_rate'),
                    ('run ≥ 6\ntokens', 'reduplication', 'loop_rate'),
                    ('emits\nnothing',  'reduplication', 'empty_rate')],
        cmap_colors = ['#1a5276', '#1e8449', '#f9e79f', '#e67e22', '#922b21'],
        vmin = 0.0, vmax = 0.6, fmt = '{:.3f}',
    ),
]

# ══════════════════════════════════════════════════════════════════════════════
#  STYLE  ── tweak font sizes, cell dimensions, colors here
# ══════════════════════════════════════════════════════════════════════════════
STYLE = dict(
    bg_color        = '#ffffff',   # figure / axes background
    cell_value_size = 8.5,         # font size inside each cell
    col_label_size  = 8.0,         # arm name on top
    model_label_size= 9.0,         # model name on the left
    title_size      = 12,          # per-table title
    main_title_size = 17,          # global title at the top
    caption_size    = 8.5,         # subtitle below global title
    col_color       = '#444444',
    model_color     = '#222222',
    ours_color      = '#203882',   # the two Malaysian fine-tunes
    title_color     = '#1a1a2e',
    caption_color   = '#666666',
    missing_bg      = '#eeeeee',   # cell bg when data is missing
    missing_text    = '#aaaaaa',   # cell text when data is missing
    cell_w          = 1.05,        # inches per column
    cell_h          = 0.50,        # inches per model row
    model_label_w   = 1.95,        # inches reserved for model names
    colorbar_w      = 0.22,        # inches for each colorbar
    gap_between     = 0.95,        # inches between tables
    margin_l        = 0.15,        # left margin
    margin_b        = 0.35,        # bottom margin
    dpi             = 200,
    output_file     = str(ROOT / 'bench' / 'benchmark_results.png'),
)

HIGHLIGHT = {'malaysian-whisper-v2', 'Malaysian-turbo-v3'}

# ══════════════════════════════════════════════════════════════════════════════
#  RENDERING  (no need to edit below unless changing layout logic)
# ══════════════════════════════════════════════════════════════════════════════

def make_cmap(colors, n=256):
    return LinearSegmentedColormap.from_list('custom', colors, N=n)


def build_matrix(scores, columns):
    """models × columns, missing metric → NaN."""
    mat = np.full((len(MODELS), len(columns)), np.nan)
    for i, (_, key) in enumerate(MODELS):
        arms = scores.get(key, {})
        for j, (_, arm, metric) in enumerate(columns):
            v = arms.get(arm, {}).get(metric)
            if isinstance(v, (int, float)):
                mat[i, j] = v
    return mat


def draw_table(fig, ax_heat, ax_cb, matrix, cmap, vmin, vmax, title, subtitle, columns, s, fmt):
    n_rows, n_cols = matrix.shape

    for r in range(n_rows):
        for c in range(n_cols):
            v = matrix[r, c]
            row_y = n_rows - r - 1                 # flip so row 0 is at top
            if np.isnan(v):
                fc, txt, tc = s['missing_bg'], '–', s['missing_text']
            else:
                norm = np.clip((v - vmin) / (vmax - vmin), 0, 1)
                fc = cmap(norm)
                txt = fmt.format(v)
                lum = 0.299*fc[0] + 0.587*fc[1] + 0.114*fc[2]
                tc = '#0a0f14' if lum > 0.45 else '#e8f4f8'
            ax_heat.add_patch(plt.Rectangle([c, row_y], 1, 1, facecolor=fc,
                                            edgecolor=s['bg_color'], linewidth=0.8))
            ax_heat.text(c + 0.5, row_y + 0.5, txt, ha='center', va='center',
                         fontsize=s['cell_value_size'], color=tc,
                         fontweight='bold', fontfamily='monospace')

    ax_heat.set_xlim(0, n_cols); ax_heat.set_ylim(0, n_rows)
    ax_heat.set_xticks(np.arange(n_cols) + 0.5)
    ax_heat.set_xticklabels([c[0] for c in columns], fontsize=s['col_label_size'],
                            color=s['col_color'], fontfamily='monospace')
    ax_heat.tick_params(axis='x', length=0, pad=3)
    ax_heat.xaxis.set_ticks_position('top')
    ax_heat.xaxis.set_label_position('top')

    ax_heat.set_yticks(np.arange(n_rows) + 0.5)
    labels = list(reversed([m[0] for m in MODELS]))
    ax_heat.set_yticklabels(labels, fontsize=s['model_label_size'],
                            color=s['model_color'], fontweight='bold')
    # the fine-tunes are the comparison being made — colour their names, not their cells
    for tick, name in zip(ax_heat.get_yticklabels(), labels):
        if name in HIGHLIGHT:
            tick.set_color(s['ours_color'])
    ax_heat.tick_params(axis='y', length=0, pad=6)
    for spine in ax_heat.spines.values():
        spine.set_visible(False)

    ax_heat.set_title(f'{title}   {subtitle}', fontsize=s['title_size'],
                      color=s['title_color'], fontweight='bold', pad=34, loc='left')

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cb = fig.colorbar(sm, cax=ax_cb, orientation='vertical')
    cb.ax.yaxis.set_tick_params(color=s['col_color'], labelsize=s['col_label_size'] - 1)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color=s['col_color'])
    cb.outline.set_edgecolor('#cccccc')
    cb.ax.set_facecolor(s['bg_color'])


def main():
    s = STYLE
    scores = json.loads((ROOT / 'bench' / 'scores.json').read_text())
    n_models = len(MODELS)

    widest = max(len(t['columns']) for t in TABLES)
    table_h = 0.62 + n_models * s['cell_h']
    total_w = s['margin_l'] + s['model_label_w'] + widest * s['cell_w'] + s['colorbar_w'] + 1.0
    total_h = (1.5 + table_h * len(TABLES) + s['gap_between'] * (len(TABLES) - 1) + 0.6)

    fig = plt.figure(figsize=(total_w, total_h), dpi=s['dpi'])
    fig.patch.set_facecolor(s['bg_color'])
    fw, fh = total_w, total_h
    to_frac = lambda x, y, w, h: (x/fw, y/fh, w/fw, h/fh)

    for idx, tbl in enumerate(TABLES):
        t_bottom = s['margin_b'] + (len(TABLES) - 1 - idx) * (table_h + s['gap_between'])
        cols = tbl['columns']
        heat_w_in = len(cols) * s['cell_w']
        heat_left = s['margin_l'] + s['model_label_w']
        cb_left = heat_left + heat_w_in + 0.16

        ax_h = fig.add_axes(to_frac(heat_left, t_bottom, heat_w_in, table_h))
        ax_c = fig.add_axes(to_frac(cb_left, t_bottom + 0.15, s['colorbar_w'], table_h - 0.75))
        for ax in (ax_h, ax_c):
            ax.set_facecolor(s['bg_color'])

        draw_table(fig, ax_h, ax_c, build_matrix(scores, cols), make_cmap(tbl['cmap_colors']),
                   tbl['vmin'], tbl['vmax'], tbl['title'], tbl['subtitle'], cols, s, tbl['fmt'])

    top_y = s['margin_b'] + table_h * len(TABLES) + s['gap_between'] * (len(TABLES) - 1)
    fig.text(0.5, (top_y + 0.92) / fh,
             'Whisper Hallucination & Repetition — 5 checkpoints × 11,852 clips',
             ha='center', va='bottom', fontsize=s['main_title_size'],
             color=s['title_color'], fontweight='bold')
    fig.text(0.5, (top_y + 0.62) / fh,
             'greedy decoding, no forced language, no temperature fallback  •  '
             'the Malaysian fine-tunes (navy) trade hallucination for repetition',
             ha='center', va='bottom', fontsize=s['caption_size'],
             color=s['caption_color'], style='italic')

    out = s['output_file']
    plt.savefig(out, dpi=s['dpi'], bbox_inches='tight',
                facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close(fig)
    print(f'Saved → {out}')




# ══════════════════════════════════════════════════════════════════════════════
#  WER DECOMPOSITION — how much of each model's WER is repetition runaway
#  Paired bars: total WER, and WER with looping clips removed. The gap is the
#  damage a ~1% tail of runaway clips does to a corpus mean.
# ══════════════════════════════════════════════════════════════════════════════

DECOMP_ARMS = [
    ('librispeech_test_clean', 'librispeech\ntest-clean'),
    ('genuine',                'genuine'),
    ('speech_in_noise',        'speech\nin noise'),
]

BAR_STYLE = dict(
    bg_color      = '#ffffff',
    grid_color    = '#e0e0e0',
    total_color   = '#922b21',   # WER as reported
    clean_color   = '#1e8449',   # WER once looping clips are removed
    total_edge    = '#6b1f18',
    clean_edge    = '#146638',
    ours_color    = '#203882',
    title_color   = '#1a1a2e',
    label_color   = '#333333',
    tick_color    = '#444444',
    caption_color = '#666666',
    anno_bg       = '#f5f5f5',
    title_fontsize= 12,
    axis_fontsize = 10,
    tick_fontsize = 9,
    value_fontsize= 8,
    dpi           = 200,
    output_file   = str(ROOT / 'bench' / 'wer_decomposition.png'),
)


def draw_decomp(ax, decomp, arm, arm_label, bs):
    names = [disp for disp, _ in MODELS]
    totals = [decomp.get(key, {}).get(arm, {}).get('wer') for _, key in MODELS]
    cleans = [decomp.get(key, {}).get(arm, {}).get('wer_excluding_looped') for _, key in MODELS]
    loops = [decomp.get(key, {}).get(arm, {}).get('loop_rate') for _, key in MODELS]

    y = np.arange(len(names))
    h = 0.38
    ax.set_facecolor(bs['bg_color'])
    ax.grid(True, axis='x', color=bs['grid_color'], linewidth=0.8, linestyle='--', alpha=0.6)
    ax.set_axisbelow(True)

    ax.barh(y + h/2, totals, height=h, color=bs['total_color'], edgecolor=bs['total_edge'],
            linewidth=0.8, label='WER as reported', zorder=3)
    ax.barh(y - h/2, cleans, height=h, color=bs['clean_color'], edgecolor=bs['clean_edge'],
            linewidth=0.8, label='WER excluding looped clips', zorder=3)

    span = max(v for v in totals if v is not None)
    for i, (t, c, lr) in enumerate(zip(totals, cleans, loops)):
        if t is None:
            continue
        ax.text(t + span*0.015, i + h/2, f'{t:.3f}', va='center', ha='left',
                fontsize=bs['value_fontsize'], color=bs['total_color'], fontweight='bold')
        ax.text(c + span*0.015, i - h/2, f'{c:.3f}', va='center', ha='left',
                fontsize=bs['value_fontsize'], color=bs['clean_color'], fontweight='bold')
        # Annotate only gaps big enough to change a conclusion, and place them to the RIGHT
        # of the value labels — centring them between the bars put them on top of the text.
        if t - c > 0.05:
            # In the gap BELOW the bar pair: to the right of them it collided with the
            # value labels, and past the axis it spilled into the next panel's labels.
            ax.annotate(f'{lr:.1%} of clips cost {t-c:.3f} WER',
                        xy=(span*0.015, i + 0.47), ha='left', va='center',
                        fontsize=7.6, color=bs['ours_color'], fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.25', fc=bs['anno_bg'],
                                  ec=bs['ours_color'], alpha=0.95, linewidth=0.9), zorder=5)

    ax.set_ylim(len(names) - 0.35, -0.6)   # room under the last row for its annotation
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=bs['tick_fontsize'], fontweight='bold')
    for tick, name in zip(ax.get_yticklabels(), names):
        tick.set_color(bs['ours_color'] if name in HIGHLIGHT else bs['label_color'])
    ax.set_xlim(0, span * 1.30)          # headroom for the value labels
    ax.set_xlabel('word error rate', fontsize=bs['axis_fontsize'], color=bs['tick_color'], labelpad=6)
    ax.tick_params(colors=bs['tick_color'], labelsize=bs['tick_fontsize'])
    ax.set_title(arm_label.replace('\n', ' '), fontsize=bs['title_fontsize'],
                 color=bs['title_color'], fontweight='bold', pad=10)
    for spine in ax.spines.values():
        spine.set_color('#cccccc')


def decomposition_main():
    bs = BAR_STYLE
    decomp = json.loads((ROOT / 'bench' / 'loop_decomposition.json').read_text())

    fig, axes = plt.subplots(1, len(DECOMP_ARMS), figsize=(17, 5.0), dpi=bs['dpi'])
    fig.patch.set_facecolor(bs['bg_color'])
    fig.subplots_adjust(wspace=0.60, left=0.10, right=0.985, top=0.70, bottom=0.14)

    for ax, (arm, label) in zip(axes, DECOMP_ARMS):
        draw_decomp(ax, decomp, arm, label, bs)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=8.5, facecolor='#f5f5f5', edgecolor='#cccccc',
               labelcolor=bs['label_color'], ncol=2, framealpha=0.95,
               loc='upper center', bbox_to_anchor=(0.5, 0.875))

    fig.suptitle('Most of the Malaysian turbo’s word error is a 1% tail of runaway clips',
                 fontsize=13.5, color=bs['title_color'], fontweight='bold', y=0.97)
    fig.text(0.5, 0.905,
             'A looping clip emits tokens until the cap and scores a WER in the tens, so a '
             'handful of them dominate the corpus mean',
             ha='center', fontsize=8.5, color=bs['caption_color'], style='italic')

    out = bs['output_file']
    plt.savefig(out, dpi=bs['dpi'], bbox_inches='tight',
                facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close(fig)
    print(f'Saved → {out}')


if __name__ == '__main__':
    main()
    decomposition_main()
