"""Temporal aggregation utilities for detector window scores."""

import numpy as np


def aggregate_temporal_scores(scores, quantile=0.90, default=0.0):
    """Aggregate window probabilities with a robust upper quantile."""
    values = np.asarray(scores, dtype=np.float64)
    if values.size == 0:
        return float(default)
    return float(np.quantile(values, quantile))
