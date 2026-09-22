"""
plot_tradeoff.py
────────────────
Generates the one diagram the whole benchmark exists to produce: a checkpoint's willingness to
invent text plotted against its ability to transcribe the same phrases when they are really
spoken.

Left panel, the trade-off itself. The x-axis is how often the model emits words over audio
with NO speech in it (`silence`, `music`, `nonspeech` pooled, weighted by clip count). The
y-axis is how often it recovers the phrase on `lexicon_synth`, where the phrase IS spoken.
The top-left corner is the model everyone wants — quiet on noise, accurate on speech — and the
measured point of this benchmark is that **nothing is there**: the checkpoints sit on a
diagonal, trading one for the other.

Right panel, the same five checkpoints per arm, normalised so "up" is always better. It shows
where the trade lands arm by arm rather than as a single summary.

Numbers come from bench/scores.json and bench/lexicon_synth_scores.json, so re-running either
scorer updates the figure with no edit here.
"""

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / 'bench' / 'scores.json'
LEX = ROOT / 'bench' / 'lexicon_synth_scores.json'

MODELS = [
    ('whisper-large-v2',                    'whisper-large-v2',       '#95a5a6', 'v'),
    ('whisper-large-v3',                    'whisper-large-v3',       '#1e8449', '^'),
    ('whisper-large-v3-turbo',              'whisper-large-v3-turbo', '#e67e22', 'D'),
    ('malaysian-whisper-large-v2',          'malaysian-whisper-v2',   '#5b7fd4', 's'),
    ('Malaysian-whisper-large-v3-turbo-v3', 'Malaysian-turbo-v3',     '#203882', 'o'),
]
HIGHLIGHT = {'malaysian-whisper-v2', 'Malaysian-turbo-v3'}

# Non-speech arms, pooled by clip count: a 42-clip silence arm should not weigh as much as
# 1,168 clips of FSD50K.
BLANK_ARMS = ['silence', 'music', 'nonspeech']

STYLE = dict(
    bg_color       = '#ffffff',
    grid_color     = '#e0e0e0',
    ours_color     = '#203882',
    title_color    = '#1a1a2e',
    label_color    = '#333333',
    tick_color     = '#444444',
    caption_color  = '#666666',
    anno_bg        = '#f5f5f5',
    good_color     = '#1e8449',
    bad_color      = '#922b21',
    marker_size    = 300,
    title_fontsize = 12,
    axis_fontsize  = 10,
    tick_fontsize  = 8.5,
    dpi            = 200,
    output_file    = str(ROOT / 'bench' / 'tradeoff.png'),
)


def pooled_blank_rate(scores, key, metric='hallucination_rate_any_output'):
    num = den = 0
    for arm in BLANK_ARMS:
        a = (scores.get(key) or {}).get(arm) or {}
        if a.get('n') and a.get(metric) is not None:
            num += a[metric] * a['n']
            den += a['n']
    return num / den if den else None


def main():
    s = STYLE
    scores = json.loads(BENCH.read_text())
    lex = json.loads(LEX.read_text())

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(16.0, 6.6), dpi=s['dpi'],
                                  gridspec_kw=dict(width_ratios=[1.25, 1.0], wspace=0.22))
    fig.patch.set_facecolor(s['bg_color'])
    fig.subplots_adjust(left=0.065, right=0.985, top=0.80, bottom=0.11)

    # ── left: invents-on-noise vs recovers-real-speech ───────────────────────────────
    ax.set_facecolor(s['bg_color'])
    ax.grid(True, color=s['grid_color'], linewidth=0.8, linestyle='--', alpha=0.6)
    ax.set_axisbelow(True)

    pts = []
    for key, name, color, marker in MODELS:
        x = pooled_blank_rate(scores, key)
        y = (lex.get(key) or {}).get('recovered_rate')
        if x is None or y is None:
            continue
        pts.append((x, y, name))
        ours = name in HIGHLIGHT
        ax.scatter([x], [y], s=s['marker_size'] + (60 if ours else 0), c=color, marker=marker,
                   edgecolors='white', linewidths=1.6, zorder=5)
        # The three OpenAI checkpoints all emit on 100% of blank clips, so they stack on the
        # right edge; fan the labels rather than letting them overprint.
        offsets = {'whisper-large-v2': (-16, -20), 'whisper-large-v3': (-14, 16),
                   'whisper-large-v3-turbo': (16, -4),
                   'malaysian-whisper-v2': (0, -24), 'Malaysian-turbo-v3': (0, 18)}
        dx, dy = offsets.get(name, (0, 16))
        ax.annotate(name, (x, y), textcoords='offset points', xytext=(dx, dy),
                    ha='right' if dx < 0 else ('left' if dx > 8 else 'center'),
                    fontsize=8.2, fontweight='bold',
                    color=s['ours_color'] if ours else s['label_color'], zorder=6)

    ax.set_xlim(-0.05, 1.30)
    ax.set_ylim(0.20, 0.82)
    ax.set_xticks(np.arange(0, 1.01, 0.2))
    ax.set_xticklabels([f'{v:.0%}' for v in np.arange(0, 1.01, 0.2)],
                       fontsize=s['tick_fontsize'], color=s['tick_color'])
    ax.set_yticks(np.arange(0.2, 0.81, 0.1))
    ax.set_yticklabels([f'{v:.0%}' for v in np.arange(0.2, 0.81, 0.1)],
                       fontsize=s['tick_fontsize'], color=s['tick_color'])
    ax.tick_params(colors=s['tick_color'], length=0)
    ax.set_xlabel('emits words over audio with NO speech  →  worse\n'
                  '(silence + music + nonspeech, 1,810 clips, weighted)',
                  fontsize=s['axis_fontsize'], color=s['tick_color'], labelpad=8)
    ax.set_ylabel('recovers the phrase when it IS spoken  →  better\n'
                  '(lexicon_synth, 6,267 clips, 83 languages)',
                  fontsize=s['axis_fontsize'], color=s['tick_color'], labelpad=8)
    ax.set_title('The trade every checkpoint makes', fontsize=s['title_fontsize'],
                 color=s['title_color'], fontweight='bold', pad=10, loc='left')

    # The empty corner is the finding, so label it rather than leaving it to inference.
    ax.annotate('nobody is here\nquiet on noise AND accurate on speech',
                xy=(0.02, 0.79), xytext=(0.02, 0.79), ha='left', va='top',
                fontsize=8.6, color=s['good_color'], fontweight='bold', style='italic',
                bbox=dict(boxstyle='round,pad=0.4', fc='#eafaf1', ec=s['good_color'],
                          alpha=0.9, linewidth=1.1), zorder=4)
    if len(pts) >= 2:
        xs = np.array([p[0] for p in pts]); ys = np.array([p[1] for p in pts])
        m, b = np.polyfit(xs, ys, 1)
        gx = np.linspace(0, 1.0, 10)
        ax.plot(gx, m * gx + b, color=s['bad_color'], linewidth=1.4, linestyle='--',
                alpha=0.55, zorder=2)
        r = float(np.corrcoef(xs, ys)[0, 1])
        ax.text(0.99, 0.02, f'the diagonal is the trade:  r = {r:+.2f}',
                transform=ax.transAxes, ha='right', va='bottom', fontsize=8.4,
                color=s['bad_color'], fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', fc=s['anno_bg'], ec=s['bad_color'],
                          alpha=0.95, linewidth=0.9), zorder=6)
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)
    for spine in ('left', 'bottom'):
        ax.spines[spine].set_color('#cccccc')

    # ── right: per-arm, oriented so up is always better ──────────────────────────────
    ROWS = [
        ('quiet on\nsilence',   lambda k: 1 - (scores[k]['silence']['hallucination_rate_any_output'])),
        ('quiet on\nmusic',     lambda k: 1 - (scores[k]['music']['hallucination_rate_any_output'])),
        ('quiet on\nnonspeech', lambda k: 1 - (scores[k]['nonspeech']['hallucination_rate_any_output'])),
        ('no runaway\non repeats', lambda k: 1 - scores[k]['reduplication']['runaway_rate']),
        ('recovers\nreal speech', lambda k: lex[k]['recovered_rate']),
        ('librispeech\n1 − WER', lambda k: 1 - scores[k]['librispeech_test_clean']['wer']),
    ]
    ax2.set_facecolor(s['bg_color'])
    ax2.grid(True, axis='y', color=s['grid_color'], linewidth=0.8, linestyle='--', alpha=0.6)
    ax2.set_axisbelow(True)
    x = np.arange(len(ROWS))
    for key, name, color, marker in MODELS:
        vals = []
        for _, fn in ROWS:
            try:
                vals.append(fn(key))
            except Exception:
                vals.append(np.nan)
        ours = name in HIGHLIGHT
        ax2.plot(x, vals, marker=marker, color=color, linewidth=2.4 if ours else 1.6,
                 markersize=7.5, markeredgecolor='white', markeredgewidth=0.8,
                 label=name, zorder=5 if ours else 3, alpha=1.0 if ours else 0.9)
    ax2.set_xticks(x)
    ax2.set_xticklabels([r[0] for r in ROWS], fontsize=8.0, color=s['label_color'])
    ax2.set_xlim(-0.35, len(ROWS) - 0.65)
    ax2.set_ylim(-0.02, 1.05)
    ax2.set_yticks(np.arange(0, 1.01, 0.2))
    ax2.set_yticklabels([f'{v:.0%}' for v in np.arange(0, 1.01, 0.2)],
                        fontsize=s['tick_fontsize'], color=s['tick_color'])
    ax2.tick_params(colors=s['tick_color'], length=0)
    ax2.set_title('Every axis oriented so higher is better',
                  fontsize=s['title_fontsize'], color=s['title_color'],
                  fontweight='bold', pad=10, loc='left')
    for spine in ('top', 'right'):
        ax2.spines[spine].set_visible(False)
    for spine in ('left', 'bottom'):
        ax2.spines[spine].set_color('#cccccc')
    leg = ax2.legend(fontsize=8.0, facecolor='#f5f5f5', edgecolor='#cccccc', framealpha=0.95,
                     loc='lower left', ncol=1)
    for txt in leg.get_texts():
        if txt.get_text() in HIGHLIGHT:
            txt.set_color(s['ours_color']); txt.set_fontweight('bold')

    fig.suptitle('Quiet on noise, or accurate on speech — the benchmark measures both halves',
                 fontsize=13.5, color=s['title_color'], fontweight='bold', y=0.965)
    fig.text(0.5, 0.895,
             'The Malaysian fine-tunes buy silence on noise with deletions on speech; the '
             'OpenAI checkpoints buy accuracy with text over every silent clip. '
             'greedy decoding, auto language.',
             ha='center', fontsize=8.6, color=s['caption_color'], style='italic')

    out = s['output_file']
    plt.savefig(out, dpi=s['dpi'], bbox_inches='tight',
                facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close(fig)
    print(f'Saved → {out}')


if __name__ == '__main__':
    main()
