"""报告数组序列化和已完成反向计时的恢复保护。"""
import json
from pathlib import Path
import runpy

import numpy as np
import pytest


SCRIPT = Path(__file__).resolve().parents[1]/'scripts/benchmark_reverse_cuda.py'


def test_report_serializes_numpy_positions_and_nested_stage_marks(tmp_path):
    module = runpy.run_path(str(SCRIPT))
    target = tmp_path/'report.json'
    module['write'](target, {'position_m': np.array([1., 2.]),
        'timing': {'mark_data': {'position': np.array([3., 4.])}}, 'count': np.int64(5)})
    assert json.loads(target.read_text()) == {
        'position_m': [1., 2.], 'timing': {'mark_data': {'position': [3., 4.]}}, 'count': 5}


def make_saved_reverse(tmp_path):
    module = runpy.run_path(str(SCRIPT))
    source = tmp_path/'old'; source.mkdir()
    cases = [{'case_id': 'c1', 'config': {}, 'observation': {}}]
    name = 'src/time_bias_localization/reverse_cuda.py'
    protocol = {'cases': cases, 'timing_repeats': 2,
        'source_hashes': {name: module['digest'](SCRIPT.parents[1]/name)}}
    module['write'](source/'protocol.json', protocol)
    module['write'](source/'setup.json', {'cpu_public_prefix_seconds': 1.})
    module['write'](source/'cases/c1/inputs.json', {'samples': []})
    for repeat in range(2):
        module['append'](source/'timings.jsonl', {'case_id': 'c1', 'repeat': repeat,
            'comparison': {'discrete_fields_equal': True, 'numeric_fields_allclose': True}})
    return module, source, cases


def test_reuse_preserves_source_and_records_actual_copy_hashes(tmp_path):
    module, source, cases = make_saved_reverse(tmp_path)
    before = {str(p): p.read_bytes() for p in source.rglob('*') if p.is_file()}
    destination = tmp_path/'new'
    rows, prepared = module['reuse_reverse_measurements'](source, cases, 2, destination)
    assert len(rows) == 2 and prepared == [(cases[0], [])]
    assert before == {str(p): p.read_bytes() for p in source.rglob('*') if p.is_file()}
    record = json.loads((destination/'reverse_reuse.json').read_text())
    for relative, sha in record['copied_sha256'].items():
        assert module['digest'](destination/relative) == sha


@pytest.mark.parametrize('change', ['incomplete', 'failed_comparison', 'algorithm_changed', 'case_changed'])
def test_reuse_rejects_incomplete_or_incompatible_measurements(tmp_path, change):
    module, source, cases = make_saved_reverse(tmp_path)
    if change in ('incomplete', 'failed_comparison'):
        rows = [json.loads(line) for line in (source/'timings.jsonl').read_text().splitlines()]
        if change == 'incomplete':
            rows.pop()
        else:
            rows[0]['comparison']['discrete_fields_equal'] = False
        (source/'timings.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
    elif change == 'algorithm_changed':
        protocol = json.loads((source/'protocol.json').read_text())
        protocol['source_hashes']['src/time_bias_localization/reverse_cuda.py'] = 'wrong'
        module['write'](source/'protocol.json', protocol)
    else:
        cases = [{'case_id': 'different', 'config': {}, 'observation': {}}]
    with pytest.raises(ValueError):
        module['reuse_reverse_measurements'](source, cases, 2, tmp_path/'new')
    assert not (tmp_path/'new/reverse_reuse.json').exists()
