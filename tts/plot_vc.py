"""
plot_vc.py
──────────
Generates a PNG scatter of the voice-conversion trade-off (intelligibility lost vs speaker
identity gained) plus a coverage chart showing which candidates could actually run.

Numbers come from tts/vc_scores.json and tts/vc_sim_calibration.json, so re-scoring updates
the figure with no edit here.

Speaker similarity is plotted on the CALIBRATED scale, not as a raw cosine: at the converted
clips' own duration, WavLM-sv scores ~0.60 between different speakers and ~0.85 between two
clips of the same one, so 0% reads as "a stranger" and 100% as "the target". Raw cosines
compress every system into a band that looks identical.

To add a system:   score it; add a row to SYSTEMS with its display name.
To change style:   edit SCATTER_STYLE.
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
#  SYSTEMS  (key in vc_scores.json → display name, family, whether it is ours)
#  family: 'conversion' converts existing audio · 'cloning' synthesises in a voice
#          'untargeted' cannot aim at a given speaker at all
# ══════════════════════════════════════════════════════════════════════════════
SYSTEMS = [
    ('knnvc',             'kNN-VC',                 'conversion', False),
    ('seedvc',            'seed-vc (6 s ref)',      'conversion', False),
    ('seedvc_longref',    'seed-vc (20 s ref)',     'conversion', False),
    ('openvoice',         'OpenVoice v2',           'conversion', False),
    ('openvoice_longref', 'OpenVoice (20 s ref)',   'conversion', False),
    ('openvoice_clone',   'OpenVoice + MeloTTS',    'cloning',    False),
    ('higgs3_clone',      'Higgs Audio v3',         'cloning',    False),
    ('scicom_untargeted', 'Multilingual-Expressive','untargeted', True),
]

# Systems that never produced a scorable grid — shown in the coverage panel only.
DID_NOT_RUN = [
    ('CosyVoice 2 (conversion)', 2, 184, 'flow encoder dies with SIGFPE'),
]

TOTAL_PAIRS = 184          # 46 scorable phrases × 4 targets (3 degenerate ones excluded)

SCATTER_STYLE = dict(
    bg_color        = '#ffffff',
    grid_color      = '#e0e0e0',
    ours_color      = '#203882',   # Scicom models
    ours_edge       = '#0f1f5c',
    conv_color      = '#27ae60',   # conversion candidates
    conv_edge       = '#1a8a48',
    clone_color     = '#e67e22',   # cloning candidates
    clone_edge      = '#b35f14',
    pick_color      = '#922b21',   # the recommended system
    title_color     = '#1a1a2e',
    label_color     = '#333333',
    tick_color      = '#444444',
    caption_color   = '#666666',
    anno_bg         = '#f5f5f5',
    band_color      = '#203882',
    marker_size     = 220,
    ours_size       = 280,
    label_fontsize  = 9.0,
    title_fontsize  = 13,
    axis_fontsize   = 10,
    tick_fontsize   = 9,
    dpi             = 200,
    output_file     = str(ROOT / 'tts' / 'vc_results.png'),
)

FAMILY_COLOR = {'conversion': ('conv_color', 'conv_edge'),
                'cloning':    ('clone_color', 'clone_edge'),
                'untargeted': ('ours_color', 'ours_edge')}

PICK = 'seedvc'            # the recommendation, ringed in the scatter


def load():
    # vc_scores.json carries every clip's detail and stays on the box (sync_excludes);
    # the summary is what travels with the repo.
    scores = json.loads((ROOT / 'tts' / 'vc_scores_summary.json').read_text())
    cal = json.loads((ROOT / 'tts' / 'vc_sim_calibration.json').read_text())
    floor = cal['floor_diff_speaker_matched']['mean']
    ceil = cal['ceiling_same_speaker']['mean']
    pts = []
    for key, name, family, ours in SYSTEMS:
        r = scores.get(key)
        if not r:
            continue
        to_pct = lambda v: None if v is None else (v - floor) / (ceil - floor) * 100
        pts.append(dict(key=key, label=name, family=family, ours=ours,
                        d_cer=r['mean_d_cer'], cer=r['mean_cer'],
                        tgt=to_pct(r['mean_sim_tgt']), src=to_pct(r['mean_sim_src']),
                        langs=r['langs'], n=r['n'], ok=r['synth_ok'],
                        targeted=r['targeted']))
    return pts, floor, ceil, cal


# ══════════════════════════════════════════════════════════════════════════════
#  RENDERING
# ══════════════════════════════════════════════════════════════════════════════

def draw_tradeoff(ax, pts, ss, offsets):
    """x = intelligibility given up, y = distance travelled toward the target speaker."""
    active = [p for p in pts if p['targeted'] and p['tgt'] is not None]

    ax.set_facecolor(ss['bg_color'])
    for spine in ax.spines.values():
        spine.set_color('#cccccc')
    ax.grid(True, color=ss['grid_color'], linewidth=0.8, linestyle='--', alpha=0.6, zorder=1)
    ax.set_axisbelow(True)

    # The calibrated band: below 0 is a stranger, 100 is the target speaker. Nothing gets
    # close to 100, which is the honest headline of this panel.
    ax.axhspan(100, 130, color=ss['band_color'], alpha=0.07, zorder=0)
    ax.axhline(100, color=ss['band_color'], linewidth=1.4, linestyle='--', alpha=0.55, zorder=2)
    ax.text(ax.get_xlim()[1], 101.5, ' same speaker (calibrated ceiling)', fontsize=8,
            color=ss['band_color'], alpha=0.75, style='italic', va='bottom', ha='right')
    ax.axhline(0, color='#999999', linewidth=1.2, linestyle=':', alpha=0.7, zorder=2)

    for p in active:
        ck, ek = FAMILY_COLOR[p['family']]
        is_pick = p['key'] == PICK
        ax.scatter(p['d_cer'], p['tgt'],
                   s=ss['ours_size'] if is_pick else ss['marker_size'],
                   color=ss[ck], edgecolors=ss['pick_color'] if is_pick else ss[ek],
                   linewidths=2.6 if is_pick else 1.2, zorder=6 if is_pick else 4)

    for p in active:
        dx, dy = offsets.get(p['key'], (0.012, 3.5))
        color = ss['pick_color'] if p['key'] == PICK else ss['label_color']
        weight = 'bold' if p['key'] == PICK else 'normal'
        ax.annotate(p['label'], xy=(p['d_cer'], p['tgt']),
                    xytext=(p['d_cer'] + dx, p['tgt'] + dy),
                    fontsize=ss['label_fontsize'], color=color, fontweight=weight,
                    arrowprops=dict(arrowstyle='-', color='#aaaaaa', linewidth=0.8,
                                    shrinkA=2, shrinkB=6),
                    bbox=dict(boxstyle='round,pad=0.25', fc=ss['anno_bg'], ec='none', alpha=0.85))
        ax.text(p['d_cer'] + dx, p['tgt'] + dy - 5.8,
                f"ΔCER {p['d_cer']:+.3f}  ·  {p['langs']} langs",
                fontsize=7.6, color=color, alpha=0.85,
                bbox=dict(boxstyle='round,pad=0.15', fc=ss['anno_bg'], ec='none', alpha=0.6))

    ax.text(0.02, 0.97, '↑ more of the target voice', transform=ax.transAxes,
            ha='left', va='top', fontsize=9, color='#00997a', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', fc=ss['anno_bg'], ec='#00997a',
                      alpha=0.85, linewidth=1.2))
    ax.text(0.98, 0.03, '← less intelligibility lost', transform=ax.transAxes,
            ha='right', va='bottom', fontsize=9, color='#f4845f', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', fc=ss['anno_bg'], ec='#f4845f',
                      alpha=0.85, linewidth=1.2))

    ax.set_xlabel('ΔCER added over the source clip  (lower is better)',
                  fontsize=ss['axis_fontsize'], color=ss['tick_color'], labelpad=6)
    ax.set_ylabel('distance travelled toward the target speaker (%)',
                  fontsize=ss['axis_fontsize'], color=ss['tick_color'], labelpad=6)
    ax.tick_params(colors=ss['tick_color'], labelsize=ss['tick_fontsize'])
    ax.set_title('Intelligibility given up  vs  speaker identity gained',
                 fontsize=ss['title_fontsize'], color=ss['title_color'],
                 fontweight='bold', pad=10)


def draw_conversion(ax, pts, ss):
    """Toward the target vs still-with-the-source, per system.

    The single most diagnostic view: a converter whose output is similar to the target AND
    to the source has not moved the voice, it has produced something in between. The gap
    between the two bars is the conversion that actually happened.
    """
    rows = [p for p in pts if p['src'] is not None]
    rows.sort(key=lambda p: -((p['tgt'] - p['src']) if p['tgt'] is not None else -1))

    y = np.arange(len(rows))
    h = 0.36
    ax.set_facecolor(ss['bg_color'])
    ax.grid(True, axis='x', color=ss['grid_color'], linewidth=0.8, linestyle='--', alpha=0.6)
    ax.set_axisbelow(True)

    for i, p in enumerate(rows):
        if p['tgt'] is not None:
            ax.barh(i - h/2, p['tgt'], height=h, color=ss['conv_color'],
                    edgecolor=ss['conv_edge'], linewidth=0.9, zorder=3)
            ax.text(p['tgt'] + 1.5, i - h/2, f"{p['tgt']:.0f}%", va='center', ha='left',
                    fontsize=7.8, color=ss['conv_edge'], fontweight='bold')
        else:
            ax.text(2, i - h/2, 'no target to aim at', va='center', ha='left',
                    fontsize=7.6, color='#8a8a8a', style='italic')
        ax.barh(i + h/2, p['src'], height=h, color=ss['clone_color'],
                edgecolor=ss['clone_edge'], linewidth=0.9, zorder=3)
        ax.text(p['src'] + 1.5, i + h/2, f"{p['src']:.0f}%", va='center', ha='left',
                fontsize=7.8, color=ss['clone_edge'], fontweight='bold')
        if p['tgt'] is not None:
            ax.text(96, i, f"gap {p['tgt'] - p['src']:+.0f}", va='center', ha='right',
                    fontsize=8, fontweight='bold',
                    color=ss['ours_color'] if p['tgt'] - p['src'] > 15 else '#8a8a8a')

    ax.set_yticks(y)
    ax.set_yticklabels([p['label'] for p in rows], fontsize=8.5, fontweight='bold',
                       color=ss['label_color'])
    ax.set_ylim(len(rows) - 0.4, -0.6)
    ax.set_xlim(0, 100)
    ax.set_xlabel('position on the calibrated speaker scale (%)',
                  fontsize=ss['axis_fontsize'], color=ss['tick_color'], labelpad=6)
    ax.tick_params(colors=ss['tick_color'], labelsize=ss['tick_fontsize'])
    ax.set_title('Did the voice actually move?',
                 fontsize=ss['title_fontsize'], color=ss['title_color'],
                 fontweight='bold', pad=10)
    for spine in ax.spines.values():
        spine.set_color('#cccccc')


def draw_coverage(ax, pts, ss):
    """How many of the grid's pairs each candidate actually produced, and in how many languages."""
    rows = [(p['label'], p['ok'], p['n'], p['langs'], p['family']) for p in pts]
    rows += [(lbl, ok, n, 0, 'failed') for lbl, ok, n, _ in DID_NOT_RUN]
    rows.sort(key=lambda r: -r[1] / max(r[2], 1))

    y = np.arange(len(rows))
    ax.set_facecolor(ss['bg_color'])
    ax.grid(True, axis='x', color=ss['grid_color'], linewidth=0.8, linestyle='--', alpha=0.6)
    ax.set_axisbelow(True)

    for i, (label, ok, n, langs, family) in enumerate(rows):
        if family == 'failed':
            color, edge = '#b3b3b3', '#8a8a8a'
        else:
            ck, ek = FAMILY_COLOR[family]
            color, edge = ss[ck], ss[ek]
        ax.barh(i, ok, height=0.62, color=color, edgecolor=edge, linewidth=0.9, zorder=3)
        note = f'{ok}/{n}' + (f'  ·  {langs} langs' if langs else '')
        ax.text(ok + TOTAL_PAIRS * 0.015, i, note, va='center', ha='left',
                fontsize=8, color=ss['label_color'], fontweight='bold')

    for lbl, _, _, why in DID_NOT_RUN:
        idx = [i for i, r in enumerate(rows) if r[0] == lbl][0]
        ax.text(TOTAL_PAIRS * 0.06, idx + 0.42, why, fontsize=7.6, color='#8a8a8a',
                style='italic', va='center')

    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=8.5, fontweight='bold',
                       color=ss['label_color'])
    ax.set_ylim(len(rows) - 0.4, -0.7)
    ax.set_xlim(0, TOTAL_PAIRS * 1.42)
    ax.set_xlabel('clips scored  (out of 184 source × target pairs)',
                  fontsize=ss['axis_fontsize'], color=ss['tick_color'], labelpad=6)
    ax.tick_params(colors=ss['tick_color'], labelsize=ss['tick_fontsize'])
    ax.set_title('Coverage — which candidates could run at all',
                 fontsize=ss['title_fontsize'], color=ss['title_color'],
                 fontweight='bold', pad=10)
    for spine in ax.spines.values():
        spine.set_color('#cccccc')


def main():
    ss = SCATTER_STYLE
    pts, floor, ceil, cal = load()

    fig, (ax_t, ax_g, ax_c) = plt.subplots(1, 3, figsize=(22.5, 6.8), dpi=ss['dpi'],
                                           gridspec_kw={'width_ratios': [1.25, 1.0, 1.0]})
    fig.patch.set_facecolor(ss['bg_color'])
    fig.subplots_adjust(wspace=0.52, left=0.052, right=0.99, top=0.79, bottom=0.11)

    # Hand-tuned to keep the four systems clustered near ΔCER 0.10-0.13 legible; leader
    # lines connect each label back to its dot.
    draw_tradeoff(ax_t, pts, ss, offsets={
        'openvoice_clone':   ( 0.030,  11.0),
        'knnvc':             (-0.150,  14.0),
        'seedvc':            ( 0.020,   9.0),
        'openvoice_longref': (-0.215, -10.0),
        'openvoice':         (-0.090, -19.0),
        'seedvc_longref':    (-0.105,  19.0),
        'higgs3_clone':      ( 0.006, -21.0),
    })
    draw_conversion(ax_g, pts, ss)
    draw_coverage(ax_c, pts, ss)

    handles = [
        mlines.Line2D([], [], marker='o', linestyle='None', color=ss['conv_color'],
                      markeredgecolor=ss['conv_edge'], markeredgewidth=1.2,
                      markersize=9, label='voice conversion'),
        mlines.Line2D([], [], marker='o', linestyle='None', color=ss['clone_color'],
                      markeredgecolor=ss['clone_edge'], markeredgewidth=1.2,
                      markersize=9, label='zero-shot cloning'),
        mlines.Line2D([], [], marker='o', linestyle='None', color=ss['ours_color'],
                      markeredgecolor=ss['ours_edge'], markeredgewidth=1.5,
                      markersize=9, label='Scicom (untargeted — no reference input)'),
        mlines.Line2D([], [], marker='o', linestyle='None', color='#ffffff',
                      markeredgecolor=ss['pick_color'], markeredgewidth=2.4,
                      markersize=10, label='best targeted pick with a usable licence'),
        plt.Rectangle((0, 0), 1, 1, fc=ss['conv_color'], ec=ss['conv_edge'],
                      label='→ toward the target speaker'),
        plt.Rectangle((0, 0), 1, 1, fc=ss['clone_color'], ec=ss['clone_edge'],
                      label='← still with the source speaker'),
    ]
    fig.legend(handles=handles, fontsize=8.5, facecolor='#f5f5f5', edgecolor='#cccccc',
               labelcolor=ss['label_color'], ncol=6, framealpha=0.95,
               loc='upper center', bbox_to_anchor=(0.5, 0.895))

    fig.suptitle('Multilingual voice conversion — 49 lexicon phrases × 4 target speakers',
                 fontsize=13.5, color=ss['title_color'], fontweight='bold', y=0.975)
    fig.text(0.5, 0.925,
             f'speaker similarity calibrated at the clips’ own duration: '
             f'{floor:.3f} cosine between different speakers = 0%, '
             f'{ceil:.3f} between clips of the same one = 100%',
             ha='center', fontsize=8.5, color=ss['caption_color'], style='italic')

    out = ss['output_file']
    plt.savefig(out, dpi=ss['dpi'], bbox_inches='tight',
                facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close(fig)
    print(f'Saved → {out}')


if __name__ == '__main__':
    main()
