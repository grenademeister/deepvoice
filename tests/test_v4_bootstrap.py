from pathlib import Path


def test_bootstrap_limits_each_pool_to_requested_count(tmp_path):
    from tools.bootstrap_v4_online_sources import select
    rows = []
    for label in (0, 1):
        for i in range(5):
            rows.append({'path': str(tmp_path / f'{label}_{i}.wav'), 'label': label, 'group': f'{label}_{i}'})
    chosen = select(rows, per_class=2, seed=7)
    assert len(chosen) == 4
    assert {row['label'] for row in chosen} == {0, 1}


def test_bootstrap_rejects_insufficient_pool():
    from tools.bootstrap_v4_online_sources import select
    rows = [{'path': 'a.wav', 'label': 0, 'group': 'a'}, {'path': 'b.wav', 'label': 1, 'group': 'b'}]
    try:
        select(rows, per_class=2, seed=7)
    except ValueError as error:
        assert 'insufficient' in str(error)
    else:
        raise AssertionError('expected insufficient pool failure')
