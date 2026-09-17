"""Recompute historical pilot aggregates; never runs a model or validates a benchmark.

Usage: python code/scripts/audit_pilot_artifact.py --out results/DOC-AUDIT-20260917/result.json
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess


def audit(path):
    raw = path.read_bytes()
    data = json.loads(raw)
    rows = data['per_frame_records']
    assert rows and len(rows) == data['num_frames']
    names = {k.split('_')[1]: k for k in data if k.startswith('condition_')}
    summaries = {}
    for letter, key in names.items():
        recomputed = {}
        for metric in ('mpjpe', 'pa_mpjpe', 'pve'):
            values = [r[f'cond_{letter}'][metric] for r in rows]
            assert all(math.isfinite(v) for v in values)
            value = sum(values) / len(values)
            assert abs(value - data[key][f'mean_{metric}_mm']) < 1e-6
            recomputed[f'mean_{metric}_mm'] = value
            recomputed[f'delta_{metric}_vs_a_mm'] = sum(
                r[f'cond_{letter}'][metric] - r['cond_a'][metric] for r in rows
            ) / len(rows)
        worse = sum(r[f'cond_{letter}']['pa_mpjpe'] > r['cond_a']['pa_mpjpe'] + 0.1 for r in rows)
        recomputed['worse_count_pa_threshold_0_1_mm'] = worse
        recomputed['worsening_rate_pct'] = 100 * worse / len(rows)
        if letter != 'a':
            assert abs(recomputed['worsening_rate_pct'] - data[key]['worsening_rate_pct']) < 1e-6
        summaries[letter] = recomputed
    return {
        'artifact': str(path), 'sha256': hashlib.sha256(raw).hexdigest(),
        'records': len(rows), 'distinct_sequences': len({r['seq_name'] for r in rows}),
        'unique_sequence_frame_pairs': len({(r['seq_name'], r['frame_idx']) for r in rows}),
        'actor_identity_recorded': all('actor_idx' in r for r in rows),
        'summaries': summaries, 'aggregate_consistency': 'PASS',
        'scientific_validity': 'INCONCLUSIVE: provenance and protocol failures; arithmetic only',
        'original_run_git_commit': data.get('git_commit'),
        'original_run_command': data.get('run_command'),
        'original_run_log': data.get('log_path'),
        'original_config_snapshot': data.get('config'),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    if Path.cwd().resolve() != root:
        parser.error('Run from the repository root')
    paths = sorted(Path('results').glob('pilot_*way_experiment_results.json'))
    report = {
        'experiment_id': 'DOC-AUDIT-20260917', 'kind': 'CPU artifact arithmetic audit, not a benchmark',
        'audit_base_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'audit_worktree_dirty': bool(subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip()),
        'audits': [audit(p) for p in paths],
        'current_source_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in sorted(Path('code').rglob('*.py'))},
        'source_hash_scope': 'current inspection snapshot, NOT reconstructed historical run code',
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ('experiment_id', 'audit_base_commit', 'audit_worktree_dirty')}))
    for row in report['audits']:
        print(row['artifact'], row['records'], row['distinct_sequences'], row['aggregate_consistency'])


if __name__ == '__main__':
    main()
