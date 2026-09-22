"""
plot_tts.py
───────────
Generates a PNG heatmap of TTS candidates × languages (ASR round-trip CER), with an AVG
column, in the same form as the Multilingual-TTS benchmark figure.

Numbers come from tts/tts_scores_summary.json (written by tts/aggregate_tts.py), which
excludes the degenerate lexicon entries — including them puts `bn` at CER 22.9 and makes the
colour scale useless.

To add a model:   score it with tts/score_tts.py, re-run aggregate_tts.py.
To change style:  edit the STYLE dict.
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
#  MODELS  (row order — key in the summary → display name, whether it is ours)
# ══════════════════════════════════════════════════════════════════════════════
MODELS = [
    ('scicom',             'Multilingual-Expressive-TTS-1.7B', True),
    ('omnivoice',          'OmniVoice (auto)',                 False),
    ('omnivoice_instruct', 'OmniVoice (voice design)',         False),
    ('xtts',               'XTTS-v2',                          False),
    ('chatterbox',         'Chatterbox Multilingual',          False),
    ('voxcpm',             'VoxCPM2',                          False),
    ('mms',                'MMS-TTS',                          False),
    ('toucan',             'ToucanTTS',                        False),
    ('higgs3',             'Higgs Audio v3',                   False),
    ('higgs',              'Higgs Audio v2',                   False),
]

# ══════════════════════════════════════════════════════════════════════════════
#  STYLE  ── tweak font sizes, cell dimensions, colors here
# ══════════════════════════════════════════════════════════════════════════════
STYLE = dict(
    bg_color        = '#ffffff',
    cell_value_size = 7.0,
    avg_value_size  = 7.6,
    lang_label_size = 8.0,
    model_label_size= 9.5,
    title_size      = 13,
    main_title_size = 16,
    caption_size    = 8.5,
    lang_color      = '#444444',
    model_color     = '#222222',
    ours_color      = '#203882',
    avg_label_color = '#203882',
    title_color     = '#1a1a2e',
    caption_color   = '#666666',
    missing_bg      = '#eeeeee',
    missing_text    = '#aaaaaa',
    cell_w          = 0.60,
    cell_h          = 0.55,
    avg_col_w       = 0.90,
    model_label_w   = 2.70,
    colorbar_w      = 0.26,
    margin_l        = 0.15,
    margin_b        = 0.40,
    cmap_colors     = ['#1a5276', '#1e8449', '#f9e79f', '#e67e22', '#922b21'],
    vmin            = 0.0,
    vmax            = 1.0,
    dpi             = 200,
    output_file     = str(ROOT / 'tts' / 'tts_results.png'),
)


def make_cmap(colors, n=256):
    return LinearSegmentedColormap.from_list('custom', colors, N=n)


def main():
    s = STYLE
    scores = json.loads((ROOT / 'tts' / 'tts_scores_summary.json').read_text())
    langs = sorted({l for r in scores.values() for l in r['per_lang_cer']})
    models = [(k, n, ours) for k, n, ours in MODELS if k in scores]

    mat = np.full((len(models), len(langs)), np.nan)
    for i, (key, _, _) in enumerate(models):
        for j, lang in enumerate(langs):
            v = scores[key]['per_lang_cer'].get(lang)
            if v is not None:
                mat[i, j] = v
    avgs = [scores[k]['mean_cer'] for k, _, _ in models]

    n_rows, n_cols = mat.shape
    heat_w = n_cols * s['cell_w']
    table_h = 0.55 + n_rows * s['cell_h']
    total_w = s['margin_l'] + s['model_label_w'] + heat_w + s['avg_col_w'] + s['colorbar_w'] + 0.9
    total_h = table_h + 1.75

    fig = plt.figure(figsize=(total_w, total_h), dpi=s['dpi'])
    fig.patch.set_facecolor(s['bg_color'])
    to_frac = lambda x, y, w, h: (x/total_w, y/total_h, w/total_w, h/total_h)

    heat_left = s['margin_l'] + s['model_label_w']
    ax_h = fig.add_axes(to_frac(heat_left, s['margin_b'], heat_w, table_h))
    ax_a = fig.add_axes(to_frac(heat_left + heat_w, s['margin_b'], s['avg_col_w'], table_h))
    ax_c = fig.add_axes(to_frac(heat_left + heat_w + s['avg_col_w'] + 0.18,
                                s['margin_b'] + 0.2, s['colorbar_w'], table_h - 0.75))
    for ax in (ax_h, ax_a, ax_c):
        ax.set_facecolor(s['bg_color'])

    cmap = make_cmap(s['cmap_colors'])

    def cell(ax, x, row_y, v, size, w=1.0):
        if np.isnan(v):
            fc, txt, tc = s['missing_bg'], '–', s['missing_text']
        else:
            norm = np.clip((v - s['vmin']) / (s['vmax'] - s['vmin']), 0, 1)
            fc = cmap(norm)
            txt = f'{v:.3f}'
            lum = 0.299*fc[0] + 0.587*fc[1] + 0.114*fc[2]
            tc = '#0a0f14' if lum > 0.45 else '#e8f4f8'
        ax.add_patch(plt.Rectangle([x, row_y], w, 1, facecolor=fc,
                                   edgecolor=s['bg_color'], linewidth=0.6))
        ax.text(x + w/2, row_y + 0.5, txt, ha='center', va='center', fontsize=size,
                color=tc, fontweight='bold', fontfamily='monospace')

    for r in range(n_rows):
        row_y = n_rows - r - 1
        for c in range(n_cols):
            cell(ax_h, c, row_y, mat[r, c], s['cell_value_size'])
        cell(ax_a, 0, row_y, avgs[r], s['avg_value_size'])

    ax_h.set_xlim(0, n_cols); ax_h.set_ylim(0, n_rows)
    ax_h.set_xticks(np.arange(n_cols) + 0.5)
    ax_h.set_xticklabels(langs, fontsize=s['lang_label_size'], color=s['lang_color'],
                         fontfamily='monospace')
    ax_h.tick_params(axis='x', length=0, pad=3)
    ax_h.xaxis.set_ticks_position('top'); ax_h.xaxis.set_label_position('top')
    ax_h.set_yticks(np.arange(n_rows) + 0.5)
    labels = [n for _, n, _ in reversed(models)]
    ours = [o for _, _, o in reversed(models)]
    ax_h.set_yticklabels(labels, fontsize=s['model_label_size'], color=s['model_color'],
                         fontweight='bold')
    for tick, is_ours in zip(ax_h.get_yticklabels(), ours):
        if is_ours:
            tick.set_color(s['ours_color'])
    ax_h.tick_params(axis='y', length=0, pad=6)
    for spine in ax_h.spines.values():
        spine.set_visible(False)

    ax_a.set_xlim(0, 1); ax_a.set_ylim(0, n_rows)
    ax_a.set_xticks([0.5]); ax_a.set_xticklabels(['MEAN'], fontsize=s['avg_value_size'],
                                                 color=s['avg_label_color'], fontweight='bold',
                                                 fontfamily='monospace')
    ax_a.xaxis.set_ticks_position('top'); ax_a.xaxis.set_label_position('top')
    ax_a.tick_params(axis='x', length=0, pad=3); ax_a.set_yticks([])
    for spine in ax_a.spines.values():
        spine.set_visible(False)
    ax_a.axvline(0, color=s['avg_label_color'], linewidth=1.5)

    ax_h.set_title('Character error rate, ASR round-trip (whisper-large-v3)   ↓  lower is better',
                   fontsize=s['title_size'], color=s['title_color'], fontweight='bold',
                   pad=30, loc='left')

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(vmin=s['vmin'], vmax=s['vmax']))
    sm.set_array([])
    cb = fig.colorbar(sm, cax=ax_c, orientation='vertical')
    cb.ax.yaxis.set_tick_params(color=s['lang_color'], labelsize=s['lang_label_size'] - 1)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color=s['lang_color'])
    cb.outline.set_edgecolor('#cccccc')

    top_y = s['margin_b'] + table_h
    fig.text(0.5, (top_y + 1.10) / total_h,
             'TTS candidate selection — 48 lexicon phrases, 22 languages',
             ha='center', va='bottom', fontsize=s['main_title_size'],
             color=s['title_color'], fontweight='bold')
    fig.text(0.5, (top_y + 0.84) / total_h,
             'can the ASR still recover the phrase from the synthesised audio?  '
             'a positive example nobody can transcribe is not a positive',
             ha='center', va='bottom', fontsize=s['caption_size'],
             color=s['caption_color'], style='italic')

    out = s['output_file']
    plt.savefig(out, dpi=s['dpi'], bbox_inches='tight',
                facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close(fig)
    print(f'Saved → {out}')


if __name__ == '__main__':
    main()
