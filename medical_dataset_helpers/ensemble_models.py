import numpy as np
from multiprocessing import Pool, cpu_count
import os


def _predict_tree(args):
    tree, X = args
    return tree.predict(X)


def _predict_proba_tree(args):
    tree, X = args
    return tree.predict_proba(X)


class DecisionTreeClassifier:
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1,
                 max_features=None, random_state=None, criterion='gini'):
        """
        Parameters:
        -----------
        max_depth : int or None, default=None
        min_samples_split : int, default=2
        min_samples_leaf : int, default=1
        max_features : int, float, str or None, default=None
        random_state : int or None, default=None
        criterion : str, default='gini'
            Supported: 'gini' or 'entropy'
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.criterion = criterion
        self.tree_ = None
        self.classes_ = None
        self.n_features_in_ = None

        if criterion not in ('gini', 'entropy'):
            raise ValueError(f"Only 'gini' and 'entropy' are supported, got {criterion}")

    def _gini(self, y):
        probs = np.bincount(y) / len(y)
        return 1.0 - np.sum(probs ** 2)

    def _entropy(self, y):
        probs = np.bincount(y) / len(y)
        probs = probs[probs > 0]
        return -np.sum(probs * np.log2(probs))

    def _impurity(self, y):
        if self.criterion == 'gini':
            return self._gini(y)
        else:
            return self._entropy(y)

    def _find_best_split(self, X, y, feature_indices):
        best_feature, best_threshold = None, None
        best_score = float('inf')
        current_impurity = self._impurity(y)

        n_total = y.shape[0]
        classes = np.unique(y)
        is_binary = (classes.size == 2) and (classes[0] == 0) and (classes[1] == 1)

        for feature_idx in feature_indices:
            x = X[:, feature_idx]
            if np.all(x == x[0]):
                continue

            order = np.argsort(x, kind='mergesort')
            xs = x[order]
            ys = y[order]

            diff = np.diff(xs)
            if not np.any(diff != 0):
                continue
            candidates = np.where(diff != 0)[0]  
            if self.min_samples_leaf > 1:
                min_left = self.min_samples_leaf - 1
                max_left = n_total - self.min_samples_leaf - 1
                mask_valid = (candidates >= min_left) & (candidates <= max_left)
                if not np.any(mask_valid):
                    continue
                candidates = candidates[mask_valid]

            if is_binary:
                cum_pos = np.cumsum(ys == 1)
                total_pos = cum_pos[-1]

                left_size = candidates + 1
                right_size = n_total - left_size
                left_pos = cum_pos[candidates]
                right_pos = total_pos - left_pos

                if self.criterion == 'gini':
                    lp = left_pos / left_size
                    rp = right_pos / right_size
                    left_gini = 1.0 - (lp * lp + (1.0 - lp) * (1.0 - lp))
                    right_gini = 1.0 - (rp * rp + (1.0 - rp) * (1.0 - rp))
                    score = (left_size / n_total) * left_gini + (right_size / n_total) * right_gini
                else:  
                    lp = left_pos / left_size
                    rp = right_pos / right_size
                    lp = np.clip(lp, 1e-12, 1 - 1e-12)
                    rp = np.clip(rp, 1e-12, 1 - 1e-12)
                    left_ent = -(lp * np.log2(lp) + (1.0 - lp) * np.log2(1.0 - lp))
                    right_ent = -(rp * np.log2(rp) + (1.0 - rp) * np.log2(1.0 - rp))
                    score = (left_size / n_total) * left_ent + (right_size / n_total) * right_ent

                idx = int(np.argmin(score))
                if score[idx] < best_score:
                    best_score = float(score[idx])
                    pos = candidates[idx]
                    best_feature = feature_idx
                    best_threshold = (xs[pos] + xs[pos + 1]) * 0.5
            else:
                k = classes.size
                y_remap = np.searchsorted(classes, ys)
                cum_counts = np.zeros((n_total, k), dtype=np.int64)
                cum_counts[0, y_remap[0]] = 1
                for i in range(1, n_total):
                    cum_counts[i] = cum_counts[i - 1]
                    cum_counts[i, y_remap[i]] += 1

                for pos in candidates:
                    left_size = pos + 1
                    right_size = n_total - left_size
                    left_counts = cum_counts[pos]
                    right_counts = cum_counts[-1] - left_counts

                    if left_size < self.min_samples_leaf or right_size < self.min_samples_leaf:
                        continue

                    if self.criterion == 'gini':
                        lp = left_counts / left_size
                        rp = right_counts / right_size
                        left_imp = 1.0 - np.sum(lp * lp)
                        right_imp = 1.0 - np.sum(rp * rp)
                    else:
                        lp = left_counts / left_size
                        rp = right_counts / right_size
                        lp = lp.clip(1e-12, 1.0)
                        rp = rp.clip(1e-12, 1.0)
                        left_imp = -np.sum(lp * np.log2(lp))
                        right_imp = -np.sum(rp * np.log2(rp))

                    score = (left_size / n_total) * left_imp + (right_size / n_total) * right_imp
                    if score < best_score:
                        best_score = float(score)
                        best_feature = feature_idx
                        best_threshold = (xs[pos] + xs[pos + 1]) * 0.5

        return best_feature, best_threshold, best_score, current_impurity

    def _build_tree(self, X, y, depth=0):
        if (self.max_depth is not None and depth >= self.max_depth) or \
           (len(y) < self.min_samples_split) or \
           (np.all(y == y[0])):
            values, counts = np.unique(y, return_counts=True)
            return {'is_leaf': True, 'class': values[np.argmax(counts)]}

        n_features = X.shape[1]
        if self.max_features is None:
            feature_indices = np.arange(n_features)
        elif isinstance(self.max_features, int):
            np.random.seed(self.random_state + depth if self.random_state is not None else None)
            feature_indices = np.random.choice(n_features, self.max_features, replace=False)
        elif isinstance(self.max_features, float):
            n_select = max(1, int(self.max_features * n_features))
            np.random.seed(self.random_state + depth if self.random_state is not None else None)
            feature_indices = np.random.choice(n_features, n_select, replace=False)
        else:
            feature_indices = np.arange(n_features)

        best_feature, best_threshold, best_score, current_score = self._find_best_split(X, y, feature_indices)

        if best_feature is None or best_score >= current_score:
            values, counts = np.unique(y, return_counts=True)
            return {'is_leaf': True, 'class': values[np.argmax(counts)]}

        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask

        left_tree = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_tree = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return {
            'is_leaf': False,
            'feature': best_feature,
            'threshold': best_threshold,
            'left': left_tree,
            'right': right_tree
        }

    def _predict_sample(self, sample, node):
        if node['is_leaf']:
            return node['class']
        if sample[node['feature']] <= node['threshold']:
            return self._predict_sample(sample, node['left'])
        else:
            return self._predict_sample(sample, node['right'])

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y).astype(int)
        self.n_features_in_ = X.shape[1]
        self.classes_ = np.unique(y)

        if self.random_state is not None:
            np.random.seed(self.random_state)

        self.tree_ = self._build_tree(X, y)
        return self

    def predict(self, X):
        X = np.asarray(X)
        return np.array([self._predict_sample(sample, self.tree_) for sample in X])

    def predict_proba(self, X):
        preds = self.predict(X)
        probs = np.zeros((len(preds), len(self.classes_)))
        for i, c in enumerate(self.classes_):
            probs[:, i] = (preds == c).astype(float)
        return probs

    def score(self, X, y):
        preds = self.predict(X)
        return np.mean(preds == y)


class RandomForestClassifier:
    def __init__(self, n_estimators=100, criterion='gini', max_depth=None,
                 min_samples_split=2, min_samples_leaf=1, max_features='sqrt',
                 bootstrap=True, oob_score=False, n_jobs=None,
                 random_state=None, verbose=0, warm_start=False, max_samples=None):
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self.warm_start = warm_start
        self.max_samples = max_samples

        self.estimators_ = []
        self.classes_ = None
        self.n_features_in_ = None
        self.oob_score_ = None

    def _resolve_n_jobs(self):
        if self.n_jobs in (None, 1):
            return 1
        if self.n_jobs == -1:
            return max(1, cpu_count())
        return max(1, int(self.n_jobs))

    @staticmethod
    def _bootstrap_sample_static(X, y, max_samples, seed=None):
        n = len(X)
        if max_samples is None:
            m = n
        elif isinstance(max_samples, int):
            m = min(max_samples, n)
        elif isinstance(max_samples, float):
            m = int(max_samples * n)
        else:
            m = n
        if seed is not None:
            np.random.seed(seed)
        indices = np.random.choice(n, size=m, replace=True)
        return X[indices], y[indices]

    @staticmethod
    def _fit_single_tree(args):
        (
            X,
            y,
            criterion,
            max_depth,
            min_samples_split,
            min_samples_leaf,
            max_features_tree,
            bootstrap,
            max_samples,
            seed,
        ) = args
        if bootstrap:
            X_boot, y_boot = RandomForestClassifier._bootstrap_sample_static(
                X, y, max_samples, seed
            )
        else:
            X_boot, y_boot = X, y

        tree = DecisionTreeClassifier(
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features_tree,
            random_state=seed,
        )
        tree.fit(X_boot, y_boot)
        return tree

    def _bootstrap_sample(self, X, y, seed=None):
        n = len(X)
        if self.max_samples is None:
            m = n
        elif isinstance(self.max_samples, int):
            m = min(self.max_samples, n)
        elif isinstance(self.max_samples, float):
            m = int(self.max_samples * n)
        else:
            m = n
        np.random.seed(seed)
        indices = np.random.choice(n, size=m, replace=True)
        return X[indices], y[indices], indices

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y).astype(int)
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]
        self.estimators_ = []

        if self.random_state is not None:
            np.random.seed(self.random_state)

        if self.max_features == 'sqrt':
            max_features_tree = max(1, int(np.sqrt(self.n_features_in_)))
        elif self.max_features == 'log2':
            max_features_tree = max(1, int(np.log2(self.n_features_in_)))
        elif isinstance(self.max_features, float):
            max_features_tree = max(1, int(self.max_features * self.n_features_in_))
        elif isinstance(self.max_features, int):
            max_features_tree = min(self.max_features, self.n_features_in_)
        else:
            max_features_tree = None

        n_jobs = self._resolve_n_jobs()

        seeds = [None if self.random_state is None else self.random_state + i for i in range(self.n_estimators)]
        args_list = [
            (
                X,
                y,
                self.criterion,
                self.max_depth,
                self.min_samples_split,
                self.min_samples_leaf,
                max_features_tree,
                self.bootstrap,
                self.max_samples,
                seeds[i],
            )
            for i in range(self.n_estimators)
        ]

        if n_jobs == 1:
            for i, args in enumerate(args_list):
                if self.verbose > 0 and i % 10 == 0:
                    print(f"Building tree {i+1}/{self.n_estimators}")
                tree = self._fit_single_tree(args)
                self.estimators_.append(tree)
        else:
            if self.verbose > 0:
                print(f"Building {self.n_estimators} trees in parallel with {n_jobs} jobs")
            with Pool(processes=n_jobs) as pool:
                for i, tree in enumerate(pool.imap_unordered(self._fit_single_tree, args_list), start=1):
                    if self.verbose > 0 and i % 10 == 0:
                        print(f"Built {i}/{self.n_estimators} trees")
                    self.estimators_.append(tree)

        return self

    def predict(self, X):
        X = np.asarray(X)
        n_jobs = self._resolve_n_jobs()
        if n_jobs == 1:
            all_preds = np.array([tree.predict(X) for tree in self.estimators_])
        else:
            with Pool(processes=n_jobs) as pool:
                all_preds = np.array(list(pool.imap_unordered(_predict_tree, [(tree, X) for tree in self.estimators_])))

        classes = self.classes_
        if classes is not None and classes.size == 2 and classes[0] == 0 and classes[1] == 1:
            mean_votes = all_preds.mean(axis=0)
            return (mean_votes >= 0.5).astype(int)

        preds = []
        for i in range(X.shape[0]):
            votes, counts = np.unique(all_preds[:, i], return_counts=True)
            preds.append(votes[np.argmax(counts)])
        return np.array(preds)

    def predict_proba(self, X):
        X = np.asarray(X)
        n_jobs = self._resolve_n_jobs()
        if n_jobs == 1:
            all_probas = np.array([tree.predict_proba(X) for tree in self.estimators_])
        else:
            with Pool(processes=n_jobs) as pool:
                all_probas = np.array(list(pool.imap_unordered(_predict_proba_tree, [(tree, X) for tree in self.estimators_])))
        return np.mean(all_probas, axis=0)

    def score(self, X, y):
        preds = self.predict(X)
        return np.mean(preds == y)
