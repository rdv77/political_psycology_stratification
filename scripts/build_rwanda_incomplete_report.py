"""Render public no-API incomplete-matrix benchmark into charts and tables."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
import textwrap

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scripts.compare_rwanda_incomplete_latent import DESTINATION, input_data

COLORS = {'bayesian_fa': '#126d91', 'ridge_factorization': '#bc6645'}


def aggregate_validation(benchmark):
    rows = []
    for method, ranks in [('constant_3', [0]), ('trait_mean', [0]),
                          ('bayesian_fa', [1, 2, 3]), ('ridge_factorization', [1, 2, 3])]:
        for q in ranks:
            folds = [r for r in benchmark['validation'] if r['method'] == method and r['q'] == q]
            n = sum(r['n'] for r in folds)
            rows.append({'method': method, 'q': q, 'n': n,
                'pooled_rmse': float(np.sqrt(sum(r['n'] * r['rmse'] ** 2 for r in folds) / n)),
                'pooled_mae': float(sum(r['n'] * r['mae'] for r in folds) / n),
                'fold_rmse_sd': float(np.std([r['rmse'] for r in folds], ddof=1))})
    return rows


def write_csv(name, rows):
    with (DESTINATION / name).open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run():
    benchmark = json.loads((DESTINATION / 'benchmark.json').read_text(encoding='utf-8'))
    data = input_data()
    rows = aggregate_validation(benchmark)
    write_csv('validation_summary.csv', rows)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'figure.facecolor': '#ffffff', 'savefig.facecolor': '#ffffff'})
    fig, ax = plt.subplots(figsize=(9.5, 5.6), layout='constrained')
    for method, offset, label in [('bayesian_fa', -.07, 'Байесовский факторный анализ'),
                                   ('ridge_factorization', .07, 'Регуляризованная факторизация')]:
        local = [r for r in rows if r['method'] == method]
        ax.plot(np.arange(1, 4) + offset, [r['pooled_rmse'] for r in local], 'o-',
                color=COLORS[method], label=label)
    for method, label, color in [('constant_3', 'Постоянный прогноз «3»', '#555555'),
                                ('trait_mean', 'Среднее по признаку', '#8d8d8d')]:
        row = next(r for r in rows if r['method'] == method)
        ax.axhline(row['pooled_rmse'], color=color, linestyle='--', label=label)
    ax.set(xticks=[1, 2, 3], xlabel='Допустимое число скрытых измерений', ylabel='Ошибка RMSE, баллы 0–5',
           title=f"Предсказание {benchmark['validation_coverage']['tested_once']} скрытых известных оценок\nМеньшая ошибка означает лучший прогноз")
    ax.legend(loc='best', fontsize=9)
    ax.grid(axis='y', alpha=.15)
    fig.savefig(DESTINATION / 'validation.png', dpi=170)
    plt.close(fig)

    columns = []
    headers = []
    for q in (1, 2, 3):
        fit = benchmark['models'][f'bayesian_fa_q{q}']
        columns.append(np.array(fit['loadings']))
        headers += [f'{q}D: F{k + 1}' for k in range(q)]
    load = np.column_stack(columns)
    fig, ax = plt.subplots(figsize=(12, 11.5), layout='constrained')
    limit = max(.3, float(np.max(abs(load))))
    image = ax.imshow(load, cmap='RdBu_r', vmin=-limit, vmax=limit, aspect='auto')
    labels = [f"{sid}  " + '\n'.join(textwrap.wrap(data['scale_names'][sid], 43)) for sid in data['scale_ids']]
    ax.set(yticks=np.arange(len(labels)), yticklabels=labels, xticks=np.arange(6), xticklabels=headers,
           title='Условная интерпретация байесовских осей\nСредние нагрузки после согласования поворотов; это не коэффициенты корреляции')
    ax.tick_params(axis='y', labelsize=8)
    for i in range(load.shape[0]):
        for j in range(load.shape[1]):
            ax.text(j, i, f'{load[i, j]:.2f}', ha='center', va='center', fontsize=8,
                    color='white' if abs(load[i, j]) > .48 else '#222222')
    ax.axvline(.5, color='white', lw=3)
    ax.axvline(2.5, color='white', lw=3)
    fig.colorbar(image, ax=ax, shrink=.65, label='Нагрузка в исходных единицах модели')
    fig.savefig(DESTINATION / 'bayesian_loadings.png', dpi=160)
    plt.close(fig)

    q1 = benchmark['models']['bayesian_fa_q1']
    score, lower, upper = [np.array(q1[key])[:, 0] for key in ('scores', 'score_lower', 'score_upper')]
    order = np.argsort(score)
    fig, ax = plt.subplots(figsize=(11, 11), layout='constrained')
    y = np.arange(len(order))
    ax.hlines(y, lower[order], upper[order], color='#8eb8c9', linewidth=2)
    ax.scatter(score[order], y, color=COLORS['bayesian_fa'], zorder=3, s=24)
    labels = [f"{data['actor_names'][data['actor_ids'][i]]}  [{np.isfinite(data['midpoint'][i]).sum()}]" for i in order]
    ax.set(yticks=y, yticklabels=labels,
           xlabel='Условная координата F1; [число информативных признаков] в подписи',
           title='Одна ось: положение 31 участника и неопределённость\n95% байесовские интервалы при выбранной ориентации; все содержат ноль')
    ax.axvline(0, color='#777777', linestyle='--', lw=1)
    ax.tick_params(axis='y', labelsize=9)
    fig.savefig(DESTINATION / 'bayesian_q1_uncertainty.png', dpi=170)
    plt.close(fig)

    # All 31 names on both projections. The scales deliberately remain equal:
    # compact posterior means must not be mistaken for well-separated groups.
    q3 = benchmark['models']['bayesian_fa_q3']
    scores = np.asarray(q3['scores'])
    fig, axes = plt.subplots(2, 1, figsize=(16, 18))
    fig.subplots_adjust(left=.29, right=.71, top=.93, bottom=.06, hspace=.30)
    for ax, b in zip(axes, (1, 2)):
        ax.scatter(scores[:, 0], scores[:, b], color=COLORS['bayesian_fa'], s=24)
        # Label ALL 31 actors in two ordered external columns. Dense posterior
        # means make in-place labels unreadable; leader lengths have no meaning.
        order = np.argsort(scores[:, 0])
        for side, indices in [('left', order[:15]), ('right', order[15:])]:
            indices = sorted(indices, key=lambda i: -scores[i, b])
            for i, y_label in zip(indices, np.linspace(.96, .04, len(indices))):
                ax.annotate(data['actor_names'][data['actor_ids'][i]],
                    (scores[i, 0], scores[i, b]), xytext=(-.13 if side == 'left' else 1.04, y_label),
                    textcoords='axes fraction', ha='right' if side == 'left' else 'left',
                    va='center', fontsize=10,
                    arrowprops={'arrowstyle': '-', 'lw': .5, 'color': '#aaaaaa'})
        ax.set(xlim=(-1.03, .70), ylim=(-.58, .58), xlabel='F1', ylabel=f'F{b + 1}',
               title=f'F1 × F{b + 1}: средние координаты')
        ax.axvline(0, color='#dddddd', lw=.7)
        ax.axhline(0, color='#dddddd', lw=.7)
    fig.suptitle('Три измерения: вычисленные положения, без выделения кластеров\nРасстояния и названия осей остаются чувствительными к исходным допущениям', fontsize=13)
    fig.savefig(DESTINATION / 'bayesian_q3_objects.png', dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), layout='constrained')
    for kind, label, color in [('unchanged_data_refit', 'Повтор на тех же данных', '#6e9c79'),
                                ('interval_perturbation', 'Изменение внутри диапазонов', '#ba784e')]:
        for ax, key, title in zip(axes, ('worst_axis_congruence', 'distance_spearman'),
                                  ('Сходство наиболее изменившейся оси', 'Сходство расстояний между участниками')):
            values = [[r[key] for r in benchmark['stress'] if r['method'] == 'bayesian_fa'
                       and r['q'] == q and r['kind'] == kind and r[key] is not None] for q in (1, 2, 3)]
            med = [np.median(v) for v in values]
            ax.plot([1, 2, 3], med, 'o-', color=color, label=label)
            ax.fill_between([1, 2, 3], [min(v) for v in values], [max(v) for v in values], color=color, alpha=.1)
            ax.set(xticks=[1, 2, 3], ylim=(-.05, 1.05), xlabel='Число измерений', title=title)
            ax.grid(axis='y', alpha=.15)
    axes[0].legend(fontsize=8, loc='lower left')
    fig.suptitle('Устойчивость байесовских решений\nЛиния — медиана, полоса — минимум и максимум пересчётов', fontsize=12)
    fig.savefig(DESTINATION / 'stability.png', dpi=170)
    plt.close(fig)

    model_rows, score_rows = [], []
    for method in ('bayesian_fa', 'ridge_factorization', 'weak_ridge'):
        for q in (1, 2, 3):
            fit = benchmark['models'][f'{method}_q{q}']
            active = fit.get('effective_rank_threshold_0_01') != 0
            for j, sid in enumerate(data['scale_ids']):
                for k in range(q):
                    model_rows.append({'method': method, 'dimensions': q, 'axis': k + 1,
                        'scale_id': sid, 'scale_name': data['scale_names'][sid],
                        'loading': fit['loadings'][j][k] if active else '',
                        'status': 'orientation_conditional' if active else 'collapsed_no_latent_signal'})
            for i, actor in enumerate(data['actor_ids']):
                for k in range(q):
                    score_rows.append({'method': method, 'dimensions': q, 'axis': k + 1,
                        'actor_id': actor, 'actor_name': data['actor_names'][actor],
                        'informative_traits': int(np.isfinite(data['midpoint'][i]).sum()),
                        'coordinate': fit['scores'][i][k] if active else '',
                        'conditional_95_lower': fit['score_lower'][i][k] if 'score_lower' in fit else '',
                        'conditional_95_upper': fit['score_upper'][i][k] if 'score_upper' in fit else '',
                        'status': 'orientation_conditional' if active else 'collapsed_no_latent_signal'})
    write_csv('loadings.csv', model_rows)
    write_csv('actor_coordinates.csv', score_rows)
    appendices = ['# Участники и признаки расчётного сравнения', '',
                  'Положения не являются новыми оценками по шкалам. Справочные должности взяты из утверждённого списка; их актуальность заново не проверялась.', '',
                  '| Участник | Роль в утверждённом списке | Измерено из 24 | Статус |', '|---|---|---:|---|']
    profiles = json.loads((ROOT / 'reports/rwanda_native_mass_profiles_2026_10_18.json').read_text(encoding='utf-8'))
    for actor in profiles['actors']:
        if actor['id'] in data['actor_ids']:
            i = data['actor_ids'].index(actor['id'])
            count = int(np.isfinite(data['midpoint'][i]).sum())
            status = 'Условное оценивание; есть данные' if count >= 4 else 'Очень слабое индивидуальное покрытие'
        else:
            count, status = 0, 'Координаты не определены из данных'
        appendices.append(f"| {actor['name']} | {actor['role']} | {count} | {status} |")
    appendices += ['', '## 24 включённых признака', '', '| ID | Признак | Точки | Диапазоны |', '|---|---|---:|---:|']
    for j, sid in enumerate(data['scale_ids']):
        points = int(data['exact_mask'][:, j].sum())
        ranges = int(np.isfinite(data['midpoint'][:, j]).sum()) - points
        appendices.append(f"| {sid} | {data['scale_names'][sid]} | {points} | {ranges} |")
    appendices += ['', '## Не включены в общую модель', '', '| Признак | Информативных участников |', '|---|---:|']
    for scale in data['excluded_scales']:
        appendices.append(f"| {data['scale_names'][scale['id']]} | {scale['observations']} |")
    (DESTINATION / 'PARTICIPANTS_AND_SCALES.md').write_text('\n'.join(appendices) + '\n', encoding='utf-8')
    print(json.dumps(rows, ensure_ascii=False, indent=2))
    print('Charts, public tables and 42-person appendix saved; no API calls.')


if __name__ == '__main__':
    run()
