import numpy as np


class KNeighborsClassifier:
    def __init__(
        self,
        n_neighbors=5,
        *,
        weights='uniform',
        algorithm='auto',
        leaf_size=30,
        p=2,
        metric='minkowski',
        metric_params=None,
        n_jobs=None,
    ):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.p = p
        self.metric = metric
        self.metric_params = metric_params
        self.n_jobs = n_jobs

        self.classes_ = None
        self.n_features_in_ = None
        self.n_samples_fit_ = None
        self.effective_metric_ = None
        self.effective_metric_params_ = None

        if not isinstance(self.n_neighbors, int) or self.n_neighbors <= 0:
            raise ValueError(f"n_neighbors must be a positive integer, got {self.n_neighbors}")
        if self.weights not in ('uniform', 'distance') and not callable(self.weights):
            raise ValueError("weights must be 'uniform', 'distance', or a callable")
        if self.metric not in ('minkowski', 'euclidean', 'manhattan') and not callable(self.metric):
            raise ValueError("metric must be 'minkowski', 'euclidean', 'manhattan', or a callable")
        if self.p <= 0:
            raise ValueError(f"p must be positive, got {self.p}")

    # --- distance utilities ---
    def _compute_distances(self, X):
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array")
        if self._X_fit is None:
            raise ValueError("This KNeighborsClassifier instance is not fitted yet.")
        if X.shape[1] != self._X_fit.shape[1]:
            raise ValueError("X has different number of features than during fit")

        if callable(self.metric):
            # Custom metric: compute pairwise distances row-wise (slow but explicit)
            n_queries = X.shape[0]
            n_train = self._X_fit.shape[0]
            distances = np.empty((n_queries, n_train), dtype=float)
            for i in range(n_queries):
                for j in range(n_train):
                    distances[i, j] = self.metric(X[i], self._X_fit[j], **(self.metric_params or {}))
            return distances

        metric = self.metric
        p = self.p
        if metric == 'euclidean' or (metric == 'minkowski' and p == 2):
            # Efficient Euclidean computation: ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a.b
            X_sq = np.sum(X ** 2, axis=1, keepdims=True)
            fit_sq = np.sum(self._X_fit ** 2, axis=1)
            # numerical stability: clip negatives to zero before sqrt
            dist_sq = np.maximum(X_sq + fit_sq - 2 * (X @ self._X_fit.T), 0.0)
            return np.sqrt(dist_sq)
        elif metric == 'manhattan' or (metric == 'minkowski' and p == 1):
            # L1 distance via broadcasting in chunks if needed
            # For simplicity and clarity, do full broadcast; datasets here are moderate
            distances = np.sum(np.abs(X[:, None, :] - self._X_fit[None, :, :]), axis=2)
            return distances
        elif metric == 'minkowski':
            # general p-norm
            distances = np.sum(np.abs(X[:, None, :] - self._X_fit[None, :, :]) ** p, axis=2) ** (1.0 / p)
            return distances
        else:
            # should not happen due to validation
            raise ValueError(f"Unsupported metric: {self.metric}")

    def _get_weights_from_distances(self, distances):
        if self.weights == 'uniform':
            return None  # uniform voting
        if self.weights == 'distance':
            # inverse distance weights; handle zero distances by assigning large weight
            with np.errstate(divide='ignore'):
                w = 1.0 / np.maximum(distances, 0.0)
            # For exact matches (distance 0), set weight to a very large number
            w[~np.isfinite(w)] = 1e12
            return w
        if callable(self.weights):
            w = self.weights(distances)
            if w.shape != distances.shape:
                raise ValueError("weights callable must return array of same shape as distances")
            return w
        # already validated
        return None

    # --- core estimator API ---
    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array")
        if y.ndim != 1:
            raise ValueError("y must be a 1D array for classification")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples")

        self._X_fit = X.astype(float, copy=False)
        self._y_fit = y
        self.n_features_in_ = X.shape[1]
        self.n_samples_fit_ = X.shape[0]
        self.classes_ = np.unique(y)

        # resolve effective metric info
        if self.metric == 'minkowski' and self.p == 2:
            self.effective_metric_ = 'euclidean'
        elif self.metric == 'minkowski' and self.p == 1:
            self.effective_metric_ = 'manhattan'
        else:
            self.effective_metric_ = self.metric
        self.effective_metric_params_ = dict(self.metric_params or {})
        if self.metric == 'minkowski':
            self.effective_metric_params_['p'] = self.p

        # Note: algorithm, leaf_size, n_jobs unused; using brute-force as baseline
        return self

    def kneighbors(self, X=None, n_neighbors=None, return_distance=True):
        if self._X_fit is None:
            raise ValueError("This KNeighborsClassifier instance is not fitted yet.")

        if X is None:
            X_query = self._X_fit
        else:
            X_query = np.asarray(X)

        k = n_neighbors if n_neighbors is not None else self.n_neighbors
        if not isinstance(k, int) or k <= 0:
            raise ValueError("n_neighbors must be a positive integer")
        if k > self.n_samples_fit_:
            raise ValueError("n_neighbors cannot be larger than number of fitted samples")

        distances = self._compute_distances(X_query)
        # argsort to get indices of k smallest distances
        neighbor_idx = np.argpartition(distances, kth=k - 1, axis=1)[:, :k]
        # sort the first k per row by actual distances to ensure order
        row_indices = np.arange(X_query.shape[0])[:, None]
        sorted_order = np.argsort(distances[row_indices, neighbor_idx], axis=1)
        neighbor_idx = neighbor_idx[row_indices, sorted_order]

        if return_distance:
            neighbor_dist = distances[row_indices, neighbor_idx]
            return neighbor_dist, neighbor_idx
        return neighbor_idx

    def predict(self, X):
        X = np.asarray(X)
        if self._X_fit is None:
            raise ValueError("This KNeighborsClassifier instance is not fitted yet.")
        if X.ndim != 2:
            raise ValueError("X must be a 2D array")
        if X.shape[1] != self.n_features_in_:
            raise ValueError("X has different number of features than during fit")

        distances, indices = self.kneighbors(X, n_neighbors=self.n_neighbors, return_distance=True)
        weights = self._get_weights_from_distances(distances)

        y_pred = []
        for i in range(X.shape[0]):
            neighbor_labels = self._y_fit[indices[i]]
            if weights is None:
                # uniform vote
                votes, counts = np.unique(neighbor_labels, return_counts=True)
                y_pred.append(votes[np.argmax(counts)])
            else:
                # weighted vote by summing weights per class
                class_weights = {c: 0.0 for c in self.classes_}
                for lbl, w in zip(neighbor_labels, weights[i]):
                    class_weights[lbl] += float(w)
                # pick class with max accumulated weight
                best_class = max(class_weights.items(), key=lambda kv: kv[1])[0]
                y_pred.append(best_class)
        return np.asarray(y_pred)

    def predict_proba(self, X):
        X = np.asarray(X)
        if self._X_fit is None:
            raise ValueError("This KNeighborsClassifier instance is not fitted yet.")
        if X.ndim != 2:
            raise ValueError("X must be a 2D array")
        if X.shape[1] != self.n_features_in_:
            raise ValueError("X has different number of features than during fit")

        distances, indices = self.kneighbors(X, n_neighbors=self.n_neighbors, return_distance=True)
        weights = self._get_weights_from_distances(distances)

        n_classes = len(self.classes_)
        proba = np.zeros((X.shape[0], n_classes), dtype=float)

        for i in range(X.shape[0]):
            neighbor_labels = self._y_fit[indices[i]]
            if weights is None:
                # uniform: each neighbor contributes 1/k to its class
                for lbl in neighbor_labels:
                    class_idx = np.where(self.classes_ == lbl)[0][0]
                    proba[i, class_idx] += 1.0
            else:
                for lbl, w in zip(neighbor_labels, weights[i]):
                    class_idx = np.where(self.classes_ == lbl)[0][0]
                    proba[i, class_idx] += float(w)
            # normalize to sum to 1; if all zeros (shouldn't happen), raise
            total = np.sum(proba[i])
            if total <= 0:
                raise ValueError("Encountered zero total weight when computing probabilities")
            proba[i] /= total

        return proba

    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
