"""Validate and summarize fixed-entry runtime results for G0 through G6."""
from __future__ import annotations
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import statistics

BASE = Path('/scratch/yh6415/project/agentopt-research/analysis/usd_cost_checkpoints_latest_under_20seed/new_qa_figures/gittins_g2_main_figures')
SOURCE = BASE / 'source_data/ablation/three_objective_direction_policy_runtime_fixed_30pct_entries_20seed'
OUT = BASE / 'corrected_gd_igd_complete_returned_set_20260929/three_objective/direction_ablation'
CONFIGURATIONS = [
    ('four_direction_real_cost', 'G0 Four directions', 4),
    ('q_only_real_cost', 'G1 One direction (Q)', 1),
    ('l_only_real_cost', 'G2 One direction (L)', 1),
    ('d_only_real_cost', 'G3 One direction (D)', 1),
    ('pure_axes_real_cost', 'G4 Three directions', 3),
    ('six_ql_dense_real_cost', 'G5 Six directions (Q-L dense)', 6),
    ('six_three_edges_real_cost', 'G6 Six directions (three edges)', 6),
]
DATASETS = ('hotpotqa', 'mathqa')
SEEDS = tuple(range(42, 62))

def csv_text(rows: list[dict]) -> str:
    out = io.StringIO(newline='')
    writer = csv.DictWriter(out, fieldnames=list(rows[0]), lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    return out.getvalue()

def main() -> None:
    expected_paths = [SOURCE / c / f'seed-{s}' / d / 'runtime.json'
                      for c, _, _ in CONFIGURATIONS for d in DATASETS for s in SEEDS]
    missing = [str(p) for p in expected_paths if not p.is_file()]
    if missing:
        raise RuntimeError(f'{len(missing)} missing runtime files; first: {missing[:3]}')
    baselines = {d: json.loads((SOURCE / f'four_direction_real_cost/seed-42/{d}/runtime.json').read_text()) for d in DATASETS}
    records, rows, summary = [], [], []
    for configuration, label, count in CONFIGURATIONS:
        for dataset in DATASETS:
            group = []
            for seed in SEEDS:
                path = SOURCE / configuration / f'seed-{seed}' / dataset / 'runtime.json'
                raw = path.read_bytes()
                r = json.loads(raw)
                expected = {
                    'configuration': configuration, 'benchmark': dataset, 'seed': seed,
                    'available_entries': 20000, 'total_evaluations': 6000, 'target_entry_cap': 6000,
                    'entry_budget_fraction': 0.3, 'realized_entry_fraction': 0.3,
                    'acquisition_cost_mode': 'real', 'continuation_mode': 'eta_decay',
                    'direction_scheduler': 'round_robin', 'exact_observation_budget': True,
                    'force_to_observation_budget': True, 'stop_reason': 'observation_budget',
                }
                for key, value in expected.items():
                    if r.get(key) != value:
                        raise ValueError(f'{path}: {key}={r.get(key)!r}, expected {value!r}')
                if len(r['directions']) != count:
                    raise ValueError(f'{path}: direction count differs')
                for key in ['input_sha256', 'source_sha256']:
                    if r[key] != baselines[dataset][key]:
                        raise ValueError(f'{path}: {key} differs from original runtime study')
                seconds = float(r['policy_wall_time_seconds'])
                if not math.isfinite(seconds) or seconds <= 0:
                    raise ValueError(f'{path}: invalid timer')
                if count == 1 and not all(r['followup_provenance']['validation'].values()):
                    raise ValueError(f'{path}: follow-up validation failed')
                row = {
                    'configuration': configuration, 'dataset': dataset, 'seed': seed,
                    'policy_wall_time_seconds': seconds, 'available_entries': r['available_entries'],
                    'total_evaluations': r['total_evaluations'],
                    'realized_entry_fraction': r['realized_entry_fraction'],
                    'forced_continuation_after_eta_floor': r['forced_continuation_after_eta_floor'],
                    'source': str(path),
                }
                rows.append(row)
                group.append(row)
                records.append({'source': str(path), 'sha256': hashlib.sha256(raw).hexdigest(), 'result': r})
            times = [r['policy_wall_time_seconds'] for r in group]
            summary.append({
                'configuration': configuration, 'label': label, 'dataset': dataset, 'n_runs': len(times),
                'mean_seconds': statistics.fmean(times),
                'error_2se_seconds': 2 * statistics.stdev(times) / math.sqrt(len(times)),
                'min_seconds': min(times), 'max_seconds': max(times),
                'ratio_to_pure_axes': 0.0, 'mean_total_evaluations': 6000.0,
                'mean_realized_entry_fraction': 0.3,
                'forced_continuation_runs': sum(r['forced_continuation_after_eta_floor'] for r in group),
            })
    axes = {r['dataset']: r['mean_seconds'] for r in summary if r['configuration'] == 'pure_axes_real_cost'}
    for r in summary:
        r['ratio_to_pure_axes'] = r['mean_seconds'] / axes[r['dataset']]
    manifest = {
        'source_root': str(SOURCE),
        'configurations': {c: label for c, label, _ in CONFIGURATIONS},
        'direction_counts': {c: n for c, _, n in CONFIGURATIONS},
        'datasets': DATASETS, 'seeds': SEEDS, 'n_runs': len(rows), 'new_single_direction_runs': 120,
        'entry_budget_fraction': 0.3, 'entries_per_run': 6000,
        'entry_budget_rule': 'exactly ceil(0.30 * available entries), allowing a shorter final batch',
        'runtime_field': 'runtime.policy_wall_time_seconds',
        'timer': 'Original simulate_three_objective_gittins timer; includes online policy computation and final online bookkeeping; excludes data loading, offline oracle diagnostics, result serialization, and Slurm queue time.',
        'uncertainty': 'mean +/- 2 sample standard errors over 20 matched seeds',
        'all_code_hashes_match_original': True, 'all_input_hashes_match_original': True,
        'all_runs_reach_exact_cap': True,
        'slurm_jobs': sorted({r['result']['slurm'].get('SLURM_ARRAY_JOB_ID', '') for r in records}),
        'caveat': 'Runs were independently scheduled on HPC compute nodes; node conditions may differ. Single-direction timings were collected in a later batch using identical recorded code and data hashes.',
        'aggregate_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    files = {
        'direction_runtime_by_seed_seven_settings.csv': csv_text(rows),
        'direction_runtime_summary_seven_settings.csv': csv_text(summary),
        'direction_runtime_manifest_seven_settings.json': json.dumps(manifest, indent=2) + '\n',
        'direction_runtime_records_seven_settings.json': json.dumps(records, indent=2) + '\n',
    }
    OUT.mkdir(parents=True, exist_ok=True)
    for name, text in files.items():
        (OUT / name).write_text(text, encoding='utf-8')
    print(json.dumps({'manifest': manifest, 'summary': summary, 'files': {str(OUT / name): hashlib.sha256(text.encode()).hexdigest() for name, text in files.items()}}))

if __name__ == '__main__':
    main()
