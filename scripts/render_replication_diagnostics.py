"""Render all retained learning curves only after complete independent-replication analysis.

This is editorial postprocessing: no fitting, hypothesis tests or checkpoint selection.
"""
from __future__ import annotations

try:
    import _path  # noqa: F401
except ModuleNotFoundError:
    pass

import hashlib
import json
from pathlib import Path
import subprocess

from scripts.run_replication import BASE, ROOT, BUDGETS, jobs, namespace, verify_frozen


def render() -> None:
    report_path = ROOT / 'results/replication_20261015_analysis.json'
    if not report_path.is_file():
        raise RuntimeError('Complete replication analysis is absent; diagnostic rendering withheld')
    report = json.loads(report_path.read_text())
    if not report.get('complete') or report.get('policies') != 110 or report.get('evaluation_cells') != 13_680:
        raise RuntimeError('Incomplete replication; diagnostic rendering withheld')
    verify_frozen()
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import numpy as np
    import pandas as pd

    figure, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    axes = axes.ravel()
    palette = {'ppo_point': '#2563eb', 'ppo_uq': '#dc2626', 'ppo_placebo': '#b45309'}
    audit = []
    for method, seed, budget in jobs():
        directory = BASE / 'models/policies' / namespace(budget) / method / str(seed)
        metadata = json.loads((directory / 'metadata.json').read_text())
        curve_path = directory / metadata['learning_curve']
        curve = pd.read_csv(curve_path, comment='#')
        steps = curve.l.cumsum()
        if not len(curve) or not np.isfinite(curve.r).all() or int(steps.iloc[-1]) != metadata['actual_timesteps']:
            raise ValueError(f'Invalid/incomplete retained learning curve: {curve_path}')
        if metadata['method'] != method or metadata['train_seed'] != seed or metadata['config']['total_timesteps'] != budget:
            raise ValueError(f'Curve/checkpoint identity mismatch: {directory}')
        ax = axes[BUDGETS.index(budget)]
        ax.plot(steps, curve.r.rolling(20, min_periods=1).mean(), color=palette[method], alpha=0.45, linewidth=0.7)
        audit.append({'method': method, 'seed': seed, 'requested_timesteps': budget,
                      'actual_timesteps': metadata['actual_timesteps'], 'episodes': len(curve),
                      'curve_sha256': hashlib.sha256(curve_path.read_bytes()).hexdigest()})
    for budget, ax in zip(BUDGETS, axes):
        ax.set_title(f'{budget:,} requested steps')
        ax.set_xlabel('Actual environment steps')
        ax.set_ylabel('Episode return (past 20 episodes)')
        ax.grid(alpha=0.2)
    axes[-1].axis('off')
    axes[-1].legend([Line2D([], [], color=color) for color in palette.values()],
                    list(palette), loc='center', frameon=False)
    axes[-1].text(0.05, 0.2, 'Each line is one training seed.\nAll terminal policies retained.\nNo curve-based checkpoint selection.\nInitial averages use fewer than 20 episodes.',
                  transform=axes[-1].transAxes, fontsize=9)
    figure.suptitle('Independent replication: every retained training curve (descriptive diagnostics)')
    target = ROOT / 'results/paper_figures/replication_budget_learning_curves'
    for suffix in ('png', 'pdf'):
        figure.savefig(target.with_suffix('.' + suffix), dpi=180)
    plt.close(figure)
    audit_path = ROOT / 'results/replication_20261015_learning_curve_audit.json'
    audit_path.write_text(json.dumps({'complete': True, 'policies': len(audit), 'smoothing':
        'Trailing 20-episode arithmetic mean; initial windows use all available past episodes.',
        'interpretation': 'Descriptive diagnostics; no inference of convergence, optimality or equivalence.',
        'source_archive_sha256': json.loads((BASE / 'frozen_inputs.json').read_text())['source_archive_sha256'],
        'curves': audit}, indent=2) + '\n')
    manuscript = ROOT / 'docs/replication_20261015_results.md'
    text = manuscript.read_text() + '\n## All retained learning curves\n\n'
    text += 'Each line represents one of the fixed training seeds. Returns are trailing means of '
    text += '20 episodes (shorter initial windows). No curve selected a checkpoint or changed a budget. '
    text += 'These diagnostics do not establish convergence or optimality. Actual step counts and '
    text += 'curve hashes are retained in `results/replication_20261015_learning_curve_audit.json`.\n\n'
    text += '![Every retained training curve](../results/paper_figures/replication_budget_learning_curves.png)\n'
    text += '\nThe frozen forecaster metadata retains several historical-reference fields. '
    text += 'They do not describe the new refit; see `results/replication_20261015_provenance_note.json` '
    text += 'for the explicit field-level clarification and measured validation disagreement.\n'
    output = manuscript.with_name('replication_20261015_results_with_diagnostics.md')
    output.write_text(text)
    for suffix in ('docx', 'pdf'):
        command = ['pandoc', str(output), '--from=gfm', '--standalone', '-V', 'geometry:margin=1in',
                   '-V', 'mainfont=DejaVu Serif', '--output=' + str(output.with_suffix('.' + suffix))]
        if suffix == 'pdf':
            command.append('--pdf-engine=xelatex')
        subprocess.run(command, cwd=ROOT / 'docs', check=True)
    print('Rendered all 110 learning curves and the complete independent-replication report.')


if __name__ == '__main__':
    render()
