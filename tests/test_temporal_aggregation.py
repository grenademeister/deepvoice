from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model"))

from temporal_aggregation import aggregate_temporal_scores


def test_single_window_is_unchanged():
    assert aggregate_temporal_scores([0.37]) == pytest.approx(0.37)


def test_isolated_maximum_does_not_control_result():
    scores = [0.1, 0.2, 0.3, 0.4, 1.0]
    assert aggregate_temporal_scores(scores) == pytest.approx(0.76)
    assert aggregate_temporal_scores(scores) < max(scores)


def test_consistently_high_windows_remain_high():
    assert aggregate_temporal_scores([0.91, 0.92, 0.93, 0.94]) == pytest.approx(0.937)


def test_empty_input_uses_zero_default():
    assert aggregate_temporal_scores([]) == 0.0


def test_df_arena_uses_temporal_quantile():
    source = (ROOT / "script.py").read_text()
    assert "return aggregate_temporal_scores(segment_scores)" in source


def test_sonics_uses_temporal_quantile():
    source = (ROOT / "model" / "sonics_infer.py").read_text()
    assert "return aggregate_temporal_scores(scores, default=0.0)" in source
