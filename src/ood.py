"""
Legacy OOD detector.

SteelGuard's production pipeline now uses the
supervised SteelDomainGate in domain_gate.py.

This module is retained temporarily for compatibility
and for comparison experiments.
"""

import numpy as np

from sklearn.neighbors import NearestNeighbors


class SteelDomainDetector:

    def __init__(
        self,
        reference_features,
        threshold,
    ):

        self.threshold = float(
            threshold
        )

        self.nearest_neighbor_model = (
            NearestNeighbors(
                n_neighbors=1,
                metric="euclidean",
            )
        )

        self.nearest_neighbor_model.fit(
            reference_features
        )

    def check(
        self,
        features,
    ):

        distances, _ = (
            self.nearest_neighbor_model.kneighbors(
                features
            )
        )

        distance = float(
            distances[0][0]
        )

        return {
            "is_in_domain":
                distance <= self.threshold,

            "distance":
                distance,

            "threshold":
                self.threshold,
        }