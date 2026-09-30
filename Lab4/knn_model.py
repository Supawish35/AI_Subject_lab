import numpy as np
from collections import Counter

class KNNClassifier:
    """
    K-Nearest Neighbors (K-NN) Classifier implemented with NumPy.
    Supports both odd and even K values. In case of an even-K vote tie,
    ties are broken deterministically by nearest neighbor distance / proximity.
    """
    def __init__(self, k: int = 5):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit training data into memory."""
        self.X_train = np.array(X, dtype=float)
        self.y_train = np.array(y)

    def _predict_single(self, test_vector: np.ndarray):
        """
        Calculate Euclidean distance across all samples in the training set,
        then identify the K closest neighbors and perform majority voting.
        If K is even and a tie occurs, tie-breaker prioritizes the class
        with the closer nearest neighbor (distance-weighted / nearest neighbor priority).
        """
        # 1. Euclidean Distance: sqrt(sum((x - y)^2))
        distances = np.sqrt(np.sum((self.X_train - test_vector) ** 2, axis=1))

        # 2. Sort indices by distance ascending
        sorted_indices = np.argsort(distances)

        # 3. Pick top-K nearest neighbors
        nearest_k_indices = sorted_indices[:self.k]
        nearest_k_distances = distances[nearest_k_indices]
        nearest_k_classes = self.y_train[nearest_k_indices]

        # 4. Count votes
        vote_counts = Counter(nearest_k_classes)
        max_votes = max(vote_counts.values())
        top_candidates = [c for c, count in vote_counts.items() if count == max_votes]

        # 5. Determine winning class
        if len(top_candidates) == 1:
            predicted_class = top_candidates[0]
        else:
            # Tie-break (especially when K is even):
            # Pick the candidate whose closest neighbor has the smallest distance
            best_cand = None
            best_dist = float('inf')
            for c in top_candidates:
                # Find min distance for this class among top-K
                c_dists = [d for d, cls_name in zip(nearest_k_distances, nearest_k_classes) if cls_name == c]
                if c_dists and min(c_dists) < best_dist:
                    best_dist = min(c_dists)
                    best_cand = c
            predicted_class = best_cand if best_cand is not None else top_candidates[0]

        return predicted_class, nearest_k_indices, nearest_k_distances, nearest_k_classes, vote_counts

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Predict target labels for an entire matrix of test samples."""
        X_test = np.array(X_test, dtype=float)
        predictions = [self._predict_single(row)[0] for row in X_test]
        return np.array(predictions)

    def predict_detailed(self, test_vector: np.ndarray):
        """Predict single sample and return detailed neighbor and voting metrics."""
        return self._predict_single(np.array(test_vector, dtype=float))
