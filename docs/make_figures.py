# docs/make_figures.py -- Abbildungen fuer die README (aus artifacts/results/*)
# Aufruf: python docs/make_figures.py
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
OUT = os.path.join(ROOT, 'docs')
RES = os.path.join(ROOT, 'artifacts', 'results')

SURFACE, INK, INK2, MUTED, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#8a8984', '#e4e3df'
BLUE, ORANGE, AQUA, RED, GREEN = '#2a78d6', '#eb6834', '#1baf7a', '#e34948', '#008300'
NAMES = {'preis': 'Preis', 'makro': 'Makro', 'stablecoins': 'Stablecoins', 'coinbase': 'Coinbase',
         'korea': 'Korea', 'aufmerksamkeit': 'Wikipedia', 'derivate': 'Funding'}

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'text.color': INK, 'axes.labelcolor': INK2,
                     'xtick.color': INK2, 'ytick.color': INK2, 'axes.edgecolor': GRID, 'figure.facecolor': SURFACE,
                     'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE})


def _clean(ax):
    for s in ('top', 'right', 'left'):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)


def _box(ax, x, y, w, h, text, edge, fill='#ffffff', size=11, weight='normal'):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.12',
                                linewidth=2, edgecolor=edge, facecolor=fill))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=size, color=INK, weight=weight)


def _arrow(ax, x0, y0, x1, y1):
    ax.annotate('', (x1, y1), (x0, y0), arrowprops=dict(arrowstyle='-|>', color=MUTED, lw=1.6))


def fig_pipeline():
    fig, ax = plt.subplots(figsize=(12, 6.2))
    ax.set_xlim(0, 12); ax.set_ylim(0, 7.2); ax.axis('off')
    stations = [('Preis', 'Binance-Kerzen'), ('Makro', 'Nasdaq · VIX · Zins · Gold · $'),
                ('Stablecoins', 'DefiLlama'), ('Coinbase', 'US-Nachfrage'), ('Korea', 'Upbit-Prämie'),
                ('Wikipedia', 'Aufmerksamkeit'), ('Funding', 'Hebel am Derivatemarkt')]
    ax.text(1.6, 6.85, 'Wetterstationen', ha='center', weight='bold', color=INK2)
    ax.text(5.0, 6.85, 'je Station ein Modell', ha='center', weight='bold', color=INK2)
    ax.text(8.0, 6.85, 'Ensemble', ha='center', weight='bold', color=INK2)
    ax.text(10.7, 6.85, 'Entscheidung', ha='center', weight='bold', color=INK2)
    for i, (n, sub) in enumerate(stations):
        y = 5.95 - i * 0.88
        _box(ax, 0.1, y, 3.0, 0.68, f'{n}\n', BLUE, size=11, weight='bold')
        ax.text(1.6, y + 0.2, sub, ha='center', fontsize=8.5, color=INK2)
        _box(ax, 3.9, y + 0.04, 2.2, 0.6, 'P(Long) · P(Short)', GRID, size=9.5)
        _arrow(ax, 3.1, y + 0.34, 3.88, y + 0.34)
        _arrow(ax, 6.1, y + 0.34, 7.0, 3.6)
    _box(ax, 7.0, 2.7, 2.0, 1.8, 'Klima\n+ Σ Abweichungen\n(Log-Odds)', ORANGE, size=10.5)
    _box(ax, 9.6, 4.0, 2.25, 0.9, 'P ≥ 45 %\n→ Long / Short', GREEN, size=10.5)
    _box(ax, 9.6, 2.3, 2.25, 0.9, 'sonst\n→ warten', GRID, size=10.5)
    _arrow(ax, 9.0, 3.6, 9.58, 4.45)
    _arrow(ax, 9.0, 3.6, 9.58, 2.75)
    ax.text(5.0, 0.05, 'monatlich neu trainiert – nur mit Vergangenheitsdaten, deren Ausgang schon feststand',
            ha='center', fontsize=9.5, color=INK2, style='italic')
    fig.savefig(os.path.join(OUT, 'pipeline.png'), dpi=130, bbox_inches='tight')
    plt.close(fig)


def fig_timing():
    rows = [('Binance-Kerzen, Funding, Coinbase, Upbit, Gold, EUR', 1),
            ('Stablecoins (DefiLlama), Wikipedia', 2), ('FRED: Nasdaq, VIX, US-10J-Zins', 3)]
    fig, ax = plt.subplots(figsize=(11, 3.3))
    days = ['t−4', 't−3', 't−2', 't−1', 't (heute)']
    for i, (name, lag) in enumerate(rows):
        y = len(rows) - 1 - i
        for d in range(5):
            known = d <= 4 - lag
            today = d == 4
            ax.add_patch(Rectangle((d + 0.04, y + 0.12), 0.92, 0.76, linewidth=0,
                                   facecolor=(BLUE if known else ('#f6e3da' if today else '#eeedea'))))
            if known:
                ax.text(d + 0.5, y + 0.5, '✓', ha='center', va='center', color='white', fontsize=13, weight='bold')
        ax.text(-0.15, y + 0.5, name, ha='right', va='center', fontsize=10)
    ax.axvline(4.0, color=ORANGE, lw=2)
    ax.text(4.02, len(rows) + 0.05, 'Entscheidung 00:05 UTC', color=ORANGE, fontsize=10, va='bottom')
    ax.set_xlim(-0.05, 5); ax.set_ylim(0, len(rows) + 0.4)
    ax.set_xticks(np.arange(5) + 0.5); ax.set_xticklabels(days)
    ax.set_yticks([]); _clean(ax); ax.spines['bottom'].set_visible(False)
    ax.set_title('Welche Datentage der Bot zum Entscheidungszeitpunkt verwenden darf (blau)', loc='left',
                 fontsize=12, color=INK, pad=22)
    fig.savefig(os.path.join(OUT, 'zeitachse.png'), dpi=130, bbox_inches='tight')
    plt.close(fig)


def fig_bss():
    r = json.load(open(os.path.join(RES, 'research.json'), encoding='utf-8'))
    st = list(NAMES)
    val = [np.nanmean([r['validierung'][s]['bss_long'], r['validierung'][s]['bss_short']]) for s in st]
    tst = [r['test_info_je_station'][s]['bss'] for s in st]
    order = np.argsort(val)
    y = np.arange(len(st))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(y + 0.2, np.array(val)[order], height=0.36, color=BLUE, label='Validierung 01/2024–03/2025',
            edgecolor=SURFACE, linewidth=2)
    ax.barh(y - 0.2, np.array(tst)[order], height=0.36, color=ORANGE, label='Endtest 04/2025–10/2026',
            edgecolor=SURFACE, linewidth=2)
    ax.axvline(0, color=INK2, lw=1)
    ax.set_yticks(y); ax.set_yticklabels([NAMES[st[i]] for i in order])
    ax.xaxis.grid(True, color=GRID); ax.set_axisbelow(True); _clean(ax)
    lim = max(abs(np.nanmin(val + tst)), abs(np.nanmax(val + tst))) * 1.25
    ax.set_xlim(-lim, lim)
    ax.set_xlabel('← schlechter als Klima      Brier Skill Score (Mittel Long/Short)      besser als Klima →')
    ax.legend(loc='upper left', frameon=False, ncol=2, bbox_to_anchor=(0, 1.1))
    ax.set_title('Treffsicherheit je Station: kaum Abstand zum Klima', loc='left', fontsize=13, pad=34)
    fig.savefig(os.path.join(OUT, 'bss_stationen.png'), dpi=130, bbox_inches='tight')
    plt.close(fig)


def fig_probs():
    w = pd.read_csv(os.path.join(RES, 'walkforward.csv'), index_col=0, parse_dates=True)
    st = list(NAMES)
    p = np.concatenate([w[f'{s}_pL'].dropna().values for s in st])
    clim = w['climL'].mean()
    fig, ax = plt.subplots(figsize=(10, 4.2))
    bins = np.arange(0.15, 0.6, 0.005)
    ax.hist(p, bins=bins, color=BLUE, edgecolor=SURFACE, linewidth=0.6)
    ax.axvline(clim, color=INK2, lw=2, ls='--')
    ax.axvline(0.45, color=GREEN, lw=2)
    top = ax.get_ylim()[1]
    ax.text(clim - 0.006, top * 0.97, f'Klima {clim * 100:.0f} %', color=INK2, fontsize=10, ha='right',
            bbox=dict(facecolor=SURFACE, edgecolor='none', pad=2))
    ax.text(0.454, top * 0.92, 'Handelsschwelle 45 %', color=GREEN, fontsize=10)
    share = (p >= 0.45).mean() * 100
    note = f'nur {share:.1f} % der\nVorhersagen liegen hier'.replace('.', ',')
    ax.text(0.454, top * 0.80, note, color=INK2, fontsize=9.5, va='top')
    ax.yaxis.set_visible(False); _clean(ax)
    ax.set_xlim(0.15, 0.6)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f'{v * 100:.0f} %'))
    ax.set_xlabel('vorhergesagte Wahrscheinlichkeit, dass ein Long sein Ziel erreicht (alle Stationen, 2023–2026)')
    ax.set_title('Die Stationen bleiben fast immer nahe am Klima', loc='left', fontsize=13)
    fig.savefig(os.path.join(OUT, 'wahrscheinlichkeiten.png'), dpi=130, bbox_inches='tight')
    plt.close(fig)


def fig_trade():
    rng = np.random.default_rng(7)
    entry, atr = 100.0, 2.0
    steps = np.r_[0, rng.normal(0.35, 0.9, 7)]
    path = entry + np.cumsum(steps)
    path[-1] = entry + 1.5 * atr + 0.2
    fig, ax = plt.subplots(figsize=(10, 4.4))
    ax.axhspan(entry - atr, entry + 1.5 * atr, xmin=0, xmax=1, color='#eef4fc', zorder=0)
    ax.axhline(entry + 1.5 * atr, color=GREEN, lw=2)
    ax.axhline(entry - atr, color=RED, lw=2)
    ax.axhline(entry, color=MUTED, lw=1, ls='--')
    ax.plot(range(len(path)), path, color=BLUE, lw=2, marker='o', ms=6, markeredgecolor=SURFACE, markeredgewidth=2)
    ax.axvline(9.5, color=MUTED, lw=1, ls=':')
    ax.text(10.4, entry + 1.5 * atr, 'Ziel = +1,5 × ATR', va='center', color=GREEN, fontsize=10.5)
    ax.text(10.4, entry, 'Einstieg (Tages-Open)', va='center', color=INK2, fontsize=10.5)
    ax.text(10.4, entry - atr, 'Stop = −1 × ATR', va='center', color=RED, fontsize=10.5)
    ax.text(9.4, entry - atr - 0.75, 'spätestens nach 10 Tagen raus', ha='right', color=INK2, fontsize=9.5)
    ax.set_xlim(-0.5, 10.2); ax.set_ylim(entry - atr - 1.2, entry + 1.5 * atr + 1.0)
    ax.set_xticks(range(0, 10)); ax.set_xticklabels([f'Tag {i + 1}' for i in range(10)], fontsize=9)
    ax.set_yticks([]); _clean(ax)
    ax.set_title('Trade-Regel: wer kommt zuerst – Ziel oder Stop?', loc='left', fontsize=13)
    fig.savefig(os.path.join(OUT, 'trade_regel.png'), dpi=130, bbox_inches='tight')
    plt.close(fig)


SCH = os.path.join(ROOT, 'forschung', 'schaerfe')


def fig_schaerfe_uebersicht():
    rows = [('1 / 1b', 'Coinbase-Prämie stündlich', 'trennt gute von schlechten Stunden,\nschlägt die Kosten nicht', 'no'),
            ('2', 'Korea-Prämie stündlich', 'schwächer, nutzt sich ab', 'no'),
            ('3', 'Wal-Einzahlungen schärfen', 'Lernphase gut, Test −0,09 %', 'no'),
            ('4', 'Listing-Short, großer Pump', '+14,5 % je Ereignis, t 2,56\n(nur ~41 Fälle)', 'yes'),
            ('5', 'Funding-Querschnitt (40 Coins)', 'in allen 3 Zeiträumen positiv,\nt knapp unter 2', 'mid')]
    col = {'yes': GREEN, 'mid': '#eda100', 'no': RED}
    lab = {'yes': 'bestanden', 'mid': 'knapp', 'no': 'durch'}
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.set_xlim(0, 11); ax.set_ylim(0, len(rows) + 0.9); ax.axis('off')
    for x, h in ((0.15, 'Schritt'), (1.25, 'Station'), (5.0, 'Ergebnis'), (9.2, 'Urteil')):
        ax.text(x, len(rows) + 0.45, h, weight='bold', color=INK2, va='center')
    for i, (n, st, res, v) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.add_patch(Rectangle((0, y + 0.06), 11, 0.88, facecolor='#ffffff' if i % 2 else '#f4f3f0', linewidth=0))
        ax.text(0.15, y + 0.5, n, va='center', fontsize=11)
        ax.text(1.25, y + 0.5, st, va='center', fontsize=11, weight='bold')
        ax.text(5.0, y + 0.5, res, va='center', fontsize=10, color=INK2)
        ax.add_patch(FancyBboxPatch((9.2, y + 0.27), 1.55, 0.46, boxstyle='round,pad=0.02,rounding_size=0.15',
                                    facecolor=col[v], linewidth=0))
        ax.text(9.975, y + 0.5, lab[v], va='center', ha='center', color='white', weight='bold', fontsize=10)
    ax.set_title('Schärfe-Arbeit: fünf Stationen einzeln geprüft (jede vorab festgelegt)', loc='left', fontsize=13)
    fig.savefig(os.path.join(OUT, 'schaerfe_uebersicht.png'), dpi=130, bbox_inches='tight')
    plt.close(fig)


def fig_quantile():
    q = pd.read_csv(os.path.join(SCH, 's01b_s02_quantile.csv'))
    q = q.groupby(['station', 'zeitraum', 'gruppe'], sort=False)['R_maker'].mean().reset_index()
    groups = ['untere 50 %', '50–90 %', '90–99 %', 'oberste 1 %']
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
    for ax, st, title in zip(axes, ('coinbase', 'korea'), ('Coinbase-Prämie', 'Korea-Prämie')):
        x = np.arange(len(groups))
        for k, (per, c) in enumerate((('Validierung', BLUE), ('Endtest', ORANGE))):
            v = [q[(q.station == st) & (q.zeitraum == per) & (q.gruppe == g)]['R_maker'].mean() for g in groups]
            ax.bar(x + (k - 0.5) * 0.38, v, width=0.36, color=c, label=per, edgecolor=SURFACE, linewidth=2)
        ax.axhline(0, color=INK2, lw=1)
        ax.set_xticks(x); ax.set_xticklabels(groups, fontsize=9.5)
        ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True); _clean(ax)
        ax.set_title(title, loc='left', fontsize=11.5)
    axes[0].set_ylabel('Ø Ergebnis je Trade (R, Limit-Einstieg)')
    axes[0].legend(frameon=False, loc='lower right')
    fig.suptitle('Je höher die Vorhersage, desto besser – aber nirgends stabil über null',
                 x=0.06, ha='left', fontsize=13)
    fig.text(0.06, -0.02, 'Stunden nach Höhe der vorhergesagten Wahrscheinlichkeit gruppiert; Mittel aus Long und Short.',
             fontsize=9.5, color=INK2)
    fig.savefig(os.path.join(OUT, 'schaerfe_quantile.png'), dpi=130, bbox_inches='tight')
    plt.close(fig)


def fig_listing():
    d = pd.read_csv(os.path.join(SCH, 's04_ereignisse.csv'))
    thr = d[~d.oos].pump.median()
    fig, ax = plt.subplots(figsize=(11, 5))
    lim = 1.0
    for oos, c, lab in ((False, MUTED, 'Lernphase bis 06/2025'), (True, BLUE, 'Test ab 07/2025')):
        x = d[d.oos == oos]
        y = x.erg.clip(-lim, lim)
        ax.scatter(x.pump.clip(-0.3, 1.5) * 100, y * 100, s=46, color=c if oos else 'none', edgecolors=c, linewidths=1.6,
                   label=lab, zorder=3)
    ax.axvline(thr * 100, color=GREEN, lw=2)
    ax.axhline(0, color=INK2, lw=1)
    o = d[d.oos]
    for grp, (a, b) in (('klein', (o.pump.min(), thr)), ('gross', (thr, o.pump.max()))):
        m = o[(o.pump >= a) & (o.pump < b + 1e-9)].erg.mean() if grp == 'gross' else o[o.pump < thr].erg.mean()
        a, b = max(a, -0.3), min(b, 1.5)
        ax.plot([a * 100, b * 100], [m * 100, m * 100], color=ORANGE, lw=3, zorder=4)
        ax.text((a + b) / 2 * 100, -88, f'Ø Test {m * 100:+.1f} %'.replace('.', ','), color=ORANGE, fontsize=11,
                ha='center', weight='bold', bbox=dict(facecolor=SURFACE, edgecolor=ORANGE, pad=4))
    ax.text(thr * 100 + 2, lim * 100 - 8, f'Schwelle {thr * 100:.0f} % Pump\n(Median der Lernphase)', color=GREEN,
            fontsize=10, va='top', bbox=dict(facecolor=SURFACE, edgecolor='none', pad=2))
    ax.set_xlim(-30, 150)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f'{v:.0f} %'))
    ax.set_xlabel('Pump in den ersten 30 min nach der Upbit-Ankündigung (%)')
    ax.set_ylabel('Ergebnis Short 72 h inkl. Funding (%)')
    ax.set_ylim(-lim * 100 - 5, lim * 100 + 5)
    ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True); _clean(ax)
    ax.legend(frameon=False, loc='lower right')
    ax.set_title('Listing-Short: großer Pump → großer Rückfall', loc='left', fontsize=13)
    fig.text(0.06, -0.02, 'Werte außerhalb des Bereichs am Rand abgeschnitten. In der Lernphase war kein Unterschied zu sehen – '
             'Vorwärtsdaten nötig.', fontsize=9.5, color=INK2)
    fig.savefig(os.path.join(OUT, 'schaerfe_listing.png'), dpi=130, bbox_inches='tight')
    plt.close(fig)


def fig_funding():
    d = pd.read_csv(os.path.join(SCH, 's05_tage.csv'), index_col=0, parse_dates=True)
    d = d[d.index >= '2021-01-01']
    eq = (1 + d.r).cumprod()
    fig, ax = plt.subplots(figsize=(11, 4.6))
    spans = (('Lernen', '2021-01-01', '2023-12-31', '#f4f3f0'), ('Validierung', '2024-01-01', '2025-03-31', '#eef4fc'),
             ('Endtest', '2025-04-01', str(d.index.max().date()), '#fdf0ea'))
    info = json.load(open(os.path.join(SCH, 's05_ergebnis.json'), encoding='utf-8'))
    keys = {'Lernen': 'lernen (Info)', 'Validierung': 'validierung', 'Endtest': 'endtest'}
    for k, (name, a, b, c) in enumerate(spans):
        ax.axvspan(pd.Timestamp(a), pd.Timestamp(b), color=c, zorder=0)
        v = info[keys[name]]
        txt = f'{name}\n{v["ertrag_pa_pct"]:+.0f} %/J\nSharpe {v["sharpe"]:.2f}'.replace('.', ',')
        ax.text(pd.Timestamp(a) + pd.Timedelta(days=15), 0.97 if k != 2 else 0.30, txt,
                transform=ax.get_xaxis_transform(), va='top', fontsize=9.5, color=INK2)
    ax.plot(eq.index, eq.values, color=BLUE, lw=2)
    ax.set_yscale('log')
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f'{v:g}×'))
    ax.yaxis.grid(True, color=GRID, which='both'); ax.set_axisbelow(True); _clean(ax)
    ax.set_ylabel('Kapital (Start = 1×, log)')
    ax.set_title('Funding-Querschnitt: Short hohes / Long niedriges Funding, marktneutral', loc='left', fontsize=13)
    fig.text(0.06, -0.02, 'Nach Kosten und echtem Funding. Effekt lässt nach (88 → 33 → 20 %/J). '
             'Universum = heute existierende Coins (leicht geschönt).', fontsize=9.5, color=INK2)
    fig.savefig(os.path.join(OUT, 'schaerfe_funding.png'), dpi=130, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    fig_pipeline(); fig_timing(); fig_bss(); fig_probs(); fig_trade()
    fig_schaerfe_uebersicht(); fig_quantile(); fig_listing(); fig_funding()
    print('Abbildungen in docs/ geschrieben.')
