"""Score calibration with isotonic regression or Platt scaling."""

from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


class ScoreCalibrator:
    def __init__(self, method: str = "isotonic") -> None:
        self.method = method
        self.model = None

    def fit(self, scores: np.ndarray, labels: np.ndarray) -> None:
        if self.method == "platt":
            self.model = LogisticRegression()
            self.model.fit(scores.reshape(-1, 1), labels)
        else:
            self.model = IsotonicRegression(out_of_bounds="clip")
            self.model.fit(scores, labels)

    def transform(self, scores: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Calibrator not fit")
        if self.method == "platt":
            return self.model.predict_proba(scores.reshape(-1, 1))[:, 1]
        return self.model.predict(scores)

    def reliability_bins(self, scores: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> dict:
        bins = np.linspace(0, 1, n_bins + 1)
        bin_ids = np.digitize(scores, bins) - 1
        result = {"bin_centers": [], "predicted": [], "observed": []}
        for b in range(n_bins):
            mask = bin_ids == b
            if mask.sum() == 0:
                continue
            result["bin_centers"].append((bins[b] + bins[b + 1]) / 2)
            result["predicted"].append(float(scores[mask].mean()))
            result["observed"].append(float(labels[mask].mean()))
        return result
