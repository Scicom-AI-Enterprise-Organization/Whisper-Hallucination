"""
plot_voice_diversity.py
───────────────────────
Generates a PNG showing how many distinct voices the synthetic lexicon contains, per language,
before and after the speaker-name top-up.

The measurement is WavLM-base-plus-sv x-vectors on the calibrated scale the VC table uses
(`tts/vc_sim_calibration.json`): at 1.6 s the judge scores ≈0.605 between clips of different
speakers and ≈0.853 between two clips of the same one. A language whose clips sit near the
ceiling is one voice wearing many filenames — which is what OmniVoice's auto mode produces,
because it takes no speaker argument at all.

Numbers come from tts/voice_diversity.json (before) and tts/voice_diversity_after.json
(after), both written by tts/voice_diversity.py, so a re-measure updates the figure with no
edit here.

To change style:   edit STYLE.
"""

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

ROOT = Path(__file__).resolve().parent.parent

BEFORE = ROOT / 'tts' / 'voice_diversity.json'
AFTER = ROOT / 'tts' / 'voice_diversity_after.json'
CALIB = ROOT / 'tts' / 'vc_sim_calibration.json'
# Which languages the top-up actually touched; everything else is unchanged by construction
# and would pad the chart with 26 flat lines.
TOPUP = ROOT / 'tts' / 'lexicon_synth_topup_filter_report.json'

STYLE = dict(
    bg_color        = '#ffffff',
    grid_color      = '#e0e0e0',
    before_color    = '#922b21',   # one voice
    after_color     = '#203882',   # many voices — the house navy
    arrow_color     = '#b0b6c4',
    floor_color     = '#1e8449',
    ceiling_color   = '#922b21',
    band_color      = '#f5f5f5',
    title_color     = '#1a1a2e',
    label_color     = '#333333',
    tick_color      = '#444444',
    caption_color   = '#666666',
    anno_bg         = '#f5f5f5',
    marker_size     = 7.0,
    title_fontsize  = 12,
    axis_fontsize   = 10,
    tick_fontsize   = 8.5,
    dpi             = 200,
    output_file     = str(ROOT / 'tts' / 'voice_diversity.png'),
)


def load():
    before = json.loads(BEFORE.read_text())['engines']
    after = json.loads(AFTER.read_text())['engines']
    calib = json.loads(CALIB.read_text())
    floor = calib['floor_diff_speaker_matched']['mean']
    ceiling = calib['ceiling_same_speaker']['mean']

    # "Before" is OmniVoice auto mode: one clip per phrase, no speaker argument. "After" pools
    # every engine, because that is what the shipped corpus is — the auto-mode clips are kept
    # and the named-speaker renditions are added alongside them.
    b = before.get('omnivoice', {}).get('langs', {})
    a = {}
    for eng in after:
        for lang, v in after[eng]['langs'].items():
            if lang not in a or v['median_cosine'] < a[lang]['median_cosine']:
                a[lang] = v
    langs = set(b) & set(a)
    if TOPUP.exists():
        langs &= set(json.loads(TOPUP.read_text()))
    langs = sorted(langs, key=lambda l: -b[l]['median_cosine'])
    return b, a, langs, floor, ceiling


def main():
    s = STYLE
    b, a, langs, floor, ceiling = load()
    fig_h = max(4.0, 0.30 * len(langs) + 2.6)
    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(14.5, fig_h), dpi=s['dpi'],
        gridspec_kw=dict(width_ratios=[2.15, 1.0], wspace=0.24))
    fig.patch.set_facecolor(s['bg_color'])
    fig.subplots_adjust(left=0.075, right=0.985, top=0.80, bottom=0.11)

    # ── left: a dumbbell per language, before → after ────────────────────────────────
    y = np.arange(len(langs))
    ax.set_facecolor(s['bg_color'])
    ax.axvspan(0, floor, color=s['floor_color'], alpha=0.07, zorder=0)
    ax.axvspan(ceiling, 1.0, color=s['ceiling_color'], alpha=0.07, zorder=0)
    ax.axvline(floor, color=s['floor_color'], linewidth=1.3, linestyle='--', alpha=0.85, zorder=2)
    ax.axvline(ceiling, color=s['ceiling_color'], linewidth=1.3, linestyle='--', alpha=0.85, zorder=2)
    ax.grid(True, axis='x', color=s['grid_color'], linewidth=0.8, linestyle='--', alpha=0.6)
    ax.set_axisbelow(True)

    for i, l in enumerate(langs):
        x0, x1 = b[l]['median_cosine'], a[l]['median_cosine']
        ax.plot([x0, x1], [i, i], color=s['arrow_color'], linewidth=2.4, zorder=3,
                solid_capstyle='round')
        ax.plot(x0, i, 'o', color=s['before_color'], markersize=s['marker_size'],
                markeredgecolor='white', markeredgewidth=0.8, zorder=4)
        ax.plot(x1, i, 'o', color=s['after_color'], markersize=s['marker_size'] + 0.8,
                markeredgecolor='white', markeredgewidth=0.8, zorder=5)

    ax.set_yticks(y)
    ax.set_yticklabels(langs, fontsize=s['tick_fontsize'], color=s['label_color'],
                       fontweight='bold', fontfamily='monospace')
    ax.set_ylim(len(langs) - 0.4, -0.6)
    ax.set_xlim(0.45, 1.0)
    ax.tick_params(colors=s['tick_color'], labelsize=s['tick_fontsize'], length=0)
    ax.set_xlabel('median pairwise speaker cosine within a language  '
                  '←  more voices        fewer voices  →',
                  fontsize=s['axis_fontsize'], color=s['tick_color'], labelpad=7)
    ax.set_title('Every language the top-up touched', fontsize=s['title_fontsize'],
                 color=s['title_color'], fontweight='bold', pad=10, loc='left')
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)
    for spine in ('left', 'bottom'):
        ax.spines[spine].set_color('#cccccc')

    ax.text(floor, -0.52, f'  different speakers {floor:.2f}', ha='left', va='center',
            fontsize=7.6, color=s['floor_color'], fontweight='bold')
    ax.text(ceiling, -0.52, f'{ceiling:.2f} same speaker  ', ha='right', va='center',
            fontsize=7.6, color=s['ceiling_color'], fontweight='bold')

    # ── right: how far the whole distribution moved ──────────────────────────────────
    ax2.set_facecolor(s['bg_color'])
    ax2.grid(True, axis='y', color=s['grid_color'], linewidth=0.8, linestyle='--', alpha=0.6)
    ax2.set_axisbelow(True)
    vals_b = [b[l]['median_cosine'] for l in langs]
    vals_a = [a[l]['median_cosine'] for l in langs]
    parts = ax2.violinplot([vals_b, vals_a], positions=[0, 1], widths=0.7,
                           showmedians=True, showextrema=False)
    for pc, color in zip(parts['bodies'], (s['before_color'], s['after_color'])):
        pc.set_facecolor(color); pc.set_alpha(0.35); pc.set_edgecolor(color); pc.set_linewidth(1.4)
    parts['cmedians'].set_color([s['before_color'], s['after_color']])
    parts['cmedians'].set_linewidth(2.0)
    rng = np.random.default_rng(0)
    for pos, vals, color in ((0, vals_b, s['before_color']), (1, vals_a, s['after_color'])):
        ax2.plot(pos + rng.uniform(-0.12, 0.12, len(vals)), vals, 'o', color=color,
                 markersize=4.0, alpha=0.75, markeredgecolor='white', markeredgewidth=0.5)
    ax2.axhline(floor, color=s['floor_color'], linewidth=1.3, linestyle='--', alpha=0.85)
    ax2.axhline(ceiling, color=s['ceiling_color'], linewidth=1.3, linestyle='--', alpha=0.85)
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(['OmniVoice\nauto mode', 'with named\nspeakers added'],
                        fontsize=s['tick_fontsize'], color=s['label_color'], fontweight='bold')
    ax2.set_ylim(0.45, 1.0)
    ax2.tick_params(colors=s['tick_color'], labelsize=s['tick_fontsize'], length=0)
    ax2.set_title(f'{len(langs)} languages', fontsize=s['title_fontsize'],
                  color=s['title_color'], fontweight='bold', pad=10, loc='left')
    for spine in ('top', 'right'):
        ax2.spines[spine].set_visible(False)
    for spine in ('left', 'bottom'):
        ax2.spines[spine].set_color('#cccccc')
    ax2.annotate(f'median {np.median(vals_b):.2f} → {np.median(vals_a):.2f}',
                 xy=(0.5, 0.965), xycoords='axes fraction', ha='center', va='top',
                 fontsize=8.2, color=s['after_color'], fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.3', fc=s['anno_bg'], ec=s['after_color'],
                           alpha=0.95, linewidth=0.9))

    handles = [
        mlines.Line2D([], [], color=s['before_color'], marker='o', linestyle='none',
                      markersize=s['marker_size'], label='OmniVoice auto mode (no speaker argument)'),
        mlines.Line2D([], [], color=s['after_color'], marker='o', linestyle='none',
                      markersize=s['marker_size'], label='+ Multilingual-Expressive named speakers'),
    ]
    fig.legend(handles=handles, fontsize=8.5, facecolor='#f5f5f5', edgecolor='#cccccc',
               labelcolor=s['label_color'], ncol=2, framealpha=0.95,
               loc='upper center', bbox_to_anchor=(0.5, 0.895))

    fig.suptitle('A label is not a voice — measuring the speakers in the synthetic lexicon',
                 fontsize=13.5, color=s['title_color'], fontweight='bold', y=0.975)
    fig.text(0.5, 0.915,
             'WavLM-sv x-vectors at 1.6 s, the same calibrated scale as the voice-conversion '
             'table  •  OmniVoice renders one voice per language because it takes no speaker '
             'argument at all',
             ha='center', fontsize=8.5, color=s['caption_color'], style='italic')

    out = s['output_file']
    plt.savefig(out, dpi=s['dpi'], bbox_inches='tight',
                facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close(fig)
    print(f'Saved → {out}')


if __name__ == '__main__':
    main()
