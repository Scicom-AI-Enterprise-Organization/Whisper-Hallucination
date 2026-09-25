"""
plot_sweep.py
─────────────
Generates the training sweep's figure: what twelve fine-tunes bought, and what each one paid.

The benchmark's whole argument is that hallucination and accuracy pull against each other, so
every panel here puts a *cost* on the y-axis and a *gain* on the x-axis. A run that walks left
without walking up is a free win; a run that walks left and up bought its silence with
deletions.

  left    non-speech hallucination  vs  English WER (librispeech)
  middle  non-speech hallucination  vs  multilingual CER (FLEURS)
  right   words over real voice-free audio (wild)  vs  phrase recovery (lexicon_synth)

Colour is the method (LoRA rank, or full), marker is the learning rate, and the untuned
checkpoint is the black star every panel is read against. Numbers come from the run
directories via bench/sweep_table.py, so a re-evaluated run updates the figure with no edit
here.
"""

import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'bench'))
from sweep_table import row_for, load, fleurs_macro_cer  # noqa: E402

METHOD_COLOR = {
    'r32':  '#5b7fd4',
    'r64':  '#1e8449',
    'r128': '#e67e22',
    'full': '#922b21',
}
LR_MARKER = {'1e-4': 'o', '2e-4': 's', '5e-4': '^', '5e-6': 'o', '1e-5': 's', '2e-5': '^'}

STYLE = dict(
    bg_color       = '#ffffff',
    grid_color     = '#e0e0e0',
    title_color    = '#1a1a2e',
    label_color    = '#333333',
    tick_color     = '#444444',
    caption_color  = '#666666',
    base_color     = '#1a1a2e',
    good_color     = '#1e8449',
    marker_size    = 170,
    title_fontsize = 11.5,
    axis_fontsize  = 9.5,
    tick_fontsize  = 8.5,
    dpi            = 200,
    output_file    = str(ROOT / 'bench' / 'sweep_results.png'),
)

# Pooled by clip count, so a 42-clip silence arm does not weigh as much as 1,168 of FSD50K.
BLANK_N = {'silence': 42, 'music': 600, 'nonspeech': 1168}


def pooled_blank(vals):
    num = den = 0
    for arm, n in BLANK_N.items():
        v = vals.get(arm)
        if isinstance(v, (int, float)):
            num += v * n
            den += n
    return num / den if den else None


# The grid only. `runs/v3_lora_all_lr2e4` is the mix sweep's control and shares the prefix;
# it is a different experiment and does not belong on these axes.
GRID_GLOBS = ('runs/v3_lora_r*_all_lr*', 'runs/v3_full_all_lr*')


def collect(globs=GRID_GLOBS, from_json=None):
    if from_json:
        rows = {r['run']: r for r in json.loads(Path(from_json).read_text())}
        names = [n for n in rows
                 if (n.startswith('v3_lora_r') or n.startswith('v3_full_')) and '_all_lr' in n]
        vals_of = lambda n: rows[n]
    else:
        names = [os.path.basename(d) for d in sorted(set(sum((glob.glob(g) for g in globs), [])))
                 if os.path.isdir(f'{d}/bench')]
        vals_of = lambda n: row_for(f'runs/{n}')
    out = []
    for name in sorted(names):
        lr = name.rsplit('lr', 1)[-1]
        method = 'full' if '_full_' in name else name.split('_')[2]   # v3_lora_r32_all_lr...
        out.append((name, method, lr, vals_of(name)))
    return out


def base_row():
    b = json.loads((ROOT / 'bench' / 'scores.json').read_text())['whisper-large-v3']
    fl = b.get('fleurs', {}).get('cer_macro')
    return {
        'silence': b['silence']['hallucination_rate'],
        'music': b['music']['hallucination_rate'],
        'nonspeech': b['nonspeech']['hallucination_rate'],
        'lex_rec': 0.698, 'wild_words': 0.999, 'halas_cer': 0.549,
        'ls_wer': b['librispeech_test_clean']['wer'],
        'fl_cer': fl,
    }


def panel(ax, s, rows, base, xf, yf, xlabel, ylabel, title, subtitle, invert_y=False):
    ax.set_facecolor(s['bg_color'])
    ax.grid(True, color=s['grid_color'], linewidth=0.8, linestyle='--', alpha=0.6)
    ax.set_axisbelow(True)

    bx, by = xf(base), yf(base)
    if bx is not None and by is not None:
        ax.scatter([bx], [by], s=340, c=s['base_color'], marker='*', edgecolors='white',
                   linewidths=1.4, zorder=6)
        ax.annotate('base large-v3', (bx, by), textcoords='offset points', xytext=(-8, 12),
                    ha='right', fontsize=8.0, fontweight='bold', color=s['base_color'],
                    zorder=7)
        # Guide lines through the base point: the quadrant down-left of them is a free win.
        ax.axhline(by, color=s['base_color'], linewidth=0.8, linestyle=':', alpha=0.45, zorder=1)
        ax.axvline(bx, color=s['base_color'], linewidth=0.8, linestyle=':', alpha=0.45, zorder=1)

    for name, method, lr, vals in rows:
        x, y = xf(vals), yf(vals)
        if x is None or y is None:
            continue
        ax.scatter([x], [y], s=s['marker_size'], c=METHOD_COLOR.get(method, '#888888'),
                   marker=LR_MARKER.get(lr, 'D'), edgecolors='white', linewidths=1.3, zorder=5)

    ax.set_xlabel(xlabel, fontsize=s['axis_fontsize'], color=s['tick_color'], labelpad=7)
    ax.set_ylabel(ylabel, fontsize=s['axis_fontsize'], color=s['tick_color'], labelpad=7)
    ax.set_title(title, fontsize=s['title_fontsize'], color=s['title_color'],
                 fontweight='bold', pad=14, loc='left')
    ax.text(0, 1.015, subtitle, transform=ax.transAxes, fontsize=7.8,
            color=s['caption_color'], style='italic', va='bottom')
    ax.tick_params(colors=s['tick_color'], labelsize=s['tick_fontsize'], length=0)
    if invert_y:
        ax.invert_yaxis()
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)
    for spine in ('left', 'bottom'):
        ax.spines[spine].set_color('#cccccc')


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--from-json', default=None, help='a bench/sweep_table.py --json dump')
    a = ap.parse_args()

    s = STYLE
    rows = collect(from_json=a.from_json)
    base = base_row()
    if not rows:
        print('no evaluated runs under runs/v3_*_all_lr* — nothing to plot')
        return

    fig, axes = plt.subplots(1, 3, figsize=(17.5, 5.9), dpi=s['dpi'])
    fig.patch.set_facecolor(s['bg_color'])
    fig.subplots_adjust(wspace=0.26, left=0.055, right=0.985, top=0.76, bottom=0.13)

    panel(axes[0], s, rows, base,
          pooled_blank, lambda v: v.get('ls_wer'),
          'emits words over audio with NO speech  ←  better\n'
          '(silence + music + nonspeech, weighted)',
          'librispeech WER  ↓ better',
          'What it bought, and what English cost',
          'down-left of the dotted lines is a free win')
    panel(axes[1], s, rows, base,
          pooled_blank, lambda v: v.get('fl_cer'),
          'emits words over audio with NO speech  ←  better\n'
          '(silence + music + nonspeech, weighted)',
          'FLEURS macro CER  ↓ better',
          'The same trade, multilingual',
          'librispeech is English only; the mix is not')
    panel(axes[2], s, rows, base,
          lambda v: v.get('wild_words'), lambda v: v.get('lex_rec'),
          'words over REAL voice-free audio  ←  better\n(wild test split)',
          'phrase recovered when it IS spoken  ↑ better\n(lexicon_synth test)',
          'Real audio, both halves',
          'up-left is the corner the benchmark exists to find')

    handles = [mlines.Line2D([], [], color=c, marker='o', linestyle='none', markersize=8,
                             markeredgecolor='white', label=lab)
               for lab, c in [('LoRA r=32', METHOD_COLOR['r32']),
                              ('LoRA r=64', METHOD_COLOR['r64']),
                              ('LoRA r=128', METHOD_COLOR['r128']),
                              ('full fine-tune', METHOD_COLOR['full'])]]
    handles += [mlines.Line2D([], [], color='#888888', marker=m, linestyle='none', markersize=8,
                              markeredgecolor='white', label=lab)
                for lab, m in [('low lr', 'o'), ('mid lr', 's'), ('high lr', '^')]]
    handles += [mlines.Line2D([], [], color=s['base_color'], marker='*', linestyle='none',
                              markersize=13, markeredgecolor='white', label='untuned base')]
    fig.legend(handles=handles, fontsize=8.4, facecolor='#f5f5f5', edgecolor='#cccccc',
               labelcolor=s['label_color'], ncol=8, framealpha=0.95,
               loc='upper center', bbox_to_anchor=(0.5, 0.905))

    fig.suptitle('Twelve fine-tunes of whisper-large-v3 on the same mix — rank and learning '
                 'rate are the only variables',
                 fontsize=13.5, color=s['title_color'], fontweight='bold', y=0.985)
    fig.text(0.5, 0.94,
             f'mix `all`: 59,028 clips, 133 h, 47% blank  •  1,000 steps, 16 clips/step  •  '
             f'{len(rows)} runs evaluated  •  greedy decoding, auto language',
             ha='center', fontsize=8.5, color=s['caption_color'], style='italic')

    out = s['output_file']
    plt.savefig(out, dpi=s['dpi'], bbox_inches='tight',
                facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close(fig)
    print(f'Saved → {out}')


if __name__ == '__main__':
    main()
