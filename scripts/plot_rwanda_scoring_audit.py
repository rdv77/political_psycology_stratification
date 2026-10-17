"""Plot the scoring audit and a separately labelled exploratory text map."""
from __future__ import annotations
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'research/rwanda_politicians_2021_2026/source_scales_v2/bottleneck_2026_10_18'


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    read = lambda path: json.loads((ROOT / path).read_text(encoding='utf-8'))
    audit = read('reports/rwanda_scoring_bottleneck_2026_10_18.json')
    model = read('reports/rwanda_text_structure_2026_10_18.json')['baseline']
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11})

    actors = sorted(audit['actors'], key=lambda a: -a['eligible_candidates'])
    y = np.arange(len(actors))
    fig, ax = plt.subplots(figsize=(14, 18))
    # Two distinct series use the same unit: source fact candidates.
    for n, actor in enumerate(actors):
        ax.hlines(n, actor['native_reviewed_eligible_candidates'], actor['eligible_candidates'], color='#b8c9cf', lw=2)
    ax.scatter([a['eligible_candidates'] for a in actors], y, color='#8c9da3', s=38, label='Все кандидаты', zorder=3)
    ax.scatter([a['native_reviewed_eligible_candidates'] for a in actors], y, color='#176d77', s=22, label='Из них содержательно проверены', zorder=4)
    ax.set_yticks(y, [a['name'] for a in actors])
    ax.invert_yaxis()
    ax.set_xscale('symlog', linthresh=1)
    ax.set_xlim(0, 6200)
    ax.set_xticks([0, 1, 10, 100, 1000, 5000], ['0', '1', '10', '100', '1 000', '5 000'])
    for n, actor in enumerate(actors):
        value = actor['eligible_candidates']
        ax.text(max(value * 1.08, .12), n,
                f"{actor['native_reviewed_eligible_candidates']} / {value}", va='center', fontsize=9)
    ax.set_title('Где остановилось оценивание: проверенные фрагменты / доступные кандидаты', pad=18)
    ax.set_xlabel('Количество фрагментов; шкала с логарифмическими интервалами')
    ax.legend(loc='lower right')
    ax.grid(axis='x', alpha=.15)
    fig.text(.5, .012, 'До дополнительной шестиисточниковой пробы. Кандидат ещё не является доказательством свойства.\n'
             'Это охват содержательной проверки, а не доля ошибок извлечения.', ha='center', fontsize=10)
    fig.tight_layout(rect=(0, .045, 1, 1))
    fig.savefig(OUT / 'review_coverage.png', dpi=150)
    plt.close(fig)

    # C1 is a shared positive vocabulary direction. C2/C3 show differences.
    people = [a for a in model['actors'] if a['text_coordinates'] is not None]
    rich = [a for a in people if a['literal_group_count'] >= 8]
    sparse = [a for a in people if a['literal_group_count'] < 8]
    fig, ax = plt.subplots(figsize=(15, 11))
    ax.scatter([a['text_coordinates'][1] for a in rich], [a['text_coordinates'][2] for a in rich],
               s=60, color='#176d77', label='Не менее 8 групп буквального материала')
    ax.scatter([a['text_coordinates'][1] for a in sparse], [a['text_coordinates'][2] for a in sparse],
               s=60, facecolors='none', edgecolors='#bc7134', label='Менее 8 групп: особенно ограниченные сведения')
    # Numbered labels keep every name readable in a separate editable legend.
    ordered = sorted(people, key=lambda a: a['name'])
    offsets = {'albert-murasira': (-20, -12), 'juvenal-marizamunda': (3, 14),
               'christopher-kayumba': (-18, 7), 'paul-kagame': (7, -12),
               'jean-claude-musabyimana': (8, -7), 'valentine-rugwabiza': (-15, -3),
               'edouard-ngirente': (6, 14)}
    for i, actor in enumerate(ordered, 1):
        x, y = actor['text_coordinates'][1:3]
        ax.annotate(str(i), (x, y), xytext=offsets.get(actor['actor_id'], (5, 5)), textcoords='offset points', fontsize=9)
    components = model['components']
    pct = lambda index: 100 * components[index]['centered_text_variance_share']
    ax.set_xlabel(f'Текстовая компонента 2 ({pct(1):.2f}% дисперсии текстовых профилей)')
    ax.set_ylabel(f'Текстовая компонента 3 ({pct(2):.2f}% дисперсии текстовых профилей)')
    ax.set_title('Диагностическая карта тем в сохранённых текстовых извлечениях', pad=16)
    ax.axhline(0, color='#9ba8ae', lw=.6)
    ax.axvline(0, color='#9ba8ae', lw=.6)
    ax.grid(alpha=.12)
    ax.legend(loc='upper right', fontsize=8)
    fig.subplots_adjust(left=.08, bottom=.13, right=.67, top=.92)
    for i, actor in enumerate(ordered, 1):
        fig.text(.69, .92 - (i - 1) * .022,
                 f"{i}. {actor['name']} ({actor['literal_group_count']})", fontsize=8.5, va='top')
    fig.text(.08, .035, '35 текстовых профилей. В скобках — число групп материала, а не независимых событий.\n'
             'Сходство лексики не означает общность убеждений, союз или статистический кластер.\n'
             'Это не факторный анализ 38 утверждённых шкал. Семь участников без подходящих текстов не имеют координат.',
             fontsize=10)
    fig.savefig(OUT / 'text_map_c2_c3.png', dpi=160)
    plt.close(fig)
    print(json.dumps({'charts': 2, 'output': str(OUT), 'api_calls': 0}))


if __name__ == '__main__':
    main()
