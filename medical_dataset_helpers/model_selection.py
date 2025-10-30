import numpy as np


def train_test_split(X, y, test_size=0.25, train_size=None, random_state=None, shuffle=True):
    X = np.asarray(X)
    y = np.asarray(y)
    n_samples = X.shape[0]
    if n_samples != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")
    if isinstance(test_size, float):
        if not 0.0 < test_size < 1.0:
            raise ValueError("test_size must be between 0 and 1 when float")
        n_test = int(n_samples * test_size)
    elif isinstance(test_size, int):
        n_test = test_size
    else:
        raise ValueError("test_size must be float or int")
    
    if train_size is None:
        n_train = n_samples - n_test
    elif isinstance(train_size, float):
        if not 0.0 < train_size < 1.0:
            raise ValueError("train_size must be between 0 and 1 when float")
        n_train = int(n_samples * train_size)
        n_test = n_samples - n_train
    elif isinstance(train_size, int):
        n_train = train_size
        n_test = n_samples - n_train
    else:
        raise ValueError("train_size must be float, int, or None")
    
    if n_train + n_test > n_samples:
        raise ValueError("train_size + test_size cannot exceed total samples")
    
    if shuffle:
        if random_state is not None:
            np.random.seed(random_state)
        indices = np.random.permutation(n_samples)
    else:
        indices = np.arange(n_samples)
    
    train_indices = indices[:n_train]
    test_indices = indices[n_train:n_train + n_test]
    
    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]
    
    return X_train, X_test, y_train, y_test


class KFold:
    
    def __init__(self, n_splits=5, shuffle=False, random_state=None):
        if n_splits < 2:
            raise ValueError("k-fold cross-validation requires at least one train/test split")
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
    
    def split(self, X, y=None, groups=None):
        X = np.asarray(X)
        n_samples = X.shape[0]
        indices = np.arange(n_samples)
        
        if self.shuffle:
            if self.random_state is not None:
                np.random.seed(self.random_state)
            indices = np.random.permutation(indices)
        
        fold_sizes = np.full(self.n_splits, n_samples // self.n_splits, dtype=int)
        fold_sizes[:n_samples % self.n_splits] += 1
        
        current = 0
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            test_indices = indices[start:stop]
            train_indices = np.concatenate([indices[:start], indices[stop:]])
            yield train_indices, test_indices
            current = stop


class GridSearchCV:
    def __init__(self, estimator, param_grid, cv=5, scoring=None, 
                 n_jobs=1, verbose=0, refit=True):
        self.estimator = estimator
        self.param_grid = param_grid
        self.cv = cv
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.refit = refit
        
        self.best_estimator_ = None
        self.best_params_ = None
        self.best_score_ = None
        self.cv_results_ = None
    
    def _generate_param_combinations(self, param_grid):
        if isinstance(param_grid, dict):
            param_grid = [param_grid]
        
        combinations = [{}]
        for grid in param_grid:
            new_combinations = []
            for params in combinations:
                for key, values in grid.items():
                    if not isinstance(values, (list, tuple, np.ndarray)):
                        values = [values]
                    for value in values:
                        new_params = params.copy()
                        new_params[key] = value
                        new_combinations.append(new_params)
            combinations = new_combinations
        
        return combinations
    
    def _set_params(self, estimator, params):
        for key, value in params.items():
            if hasattr(estimator, key):
                setattr(estimator, key, value)
            elif hasattr(estimator, 'set_params'):
                estimator.set_params(**{key: value})
            else:
                parts = key.split('__')
                obj = estimator
                for part in parts[:-1]:
                    if hasattr(obj, 'named_steps'):
                        obj = obj.named_steps.get(part, obj)
                    elif hasattr(obj, part):
                        obj = getattr(obj, part)
                    else:
                        break
                if hasattr(obj, parts[-1]):
                    setattr(obj, parts[-1], value)
    
    def _score(self, estimator, X, y):
        if self.scoring is None:
            if hasattr(estimator, 'score'):
                return estimator.score(X, y)
            else:
                raise ValueError("No scoring method available")
        elif callable(self.scoring):
            return self.scoring(estimator, X, y)
        elif self.scoring == 'accuracy':
            try:
                from .metrics import accuracy_score
            except ImportError:
                from metrics import accuracy_score
            y_pred = estimator.predict(X)
            return accuracy_score(y, y_pred)
        else:
            raise ValueError(f"Unknown scoring: {self.scoring}")
    
    def fit(self, X, y=None):
        X = np.asarray(X)
        if y is not None:
            y = np.asarray(y)
        
        if isinstance(self.cv, int):
            cv = KFold(n_splits=self.cv, shuffle=True, random_state=42)
        else:
            cv = self.cv
        
        param_combinations = self._generate_param_combinations(self.param_grid)
        
        results = {
            'params': [],
            'mean_test_score': [],
            'std_test_score': [],
            'rank_test_score': []
        }
        
        best_score = -np.inf
        best_params = None
        best_estimator = None
        
        for i, params in enumerate(param_combinations):
            if self.verbose > 0:
                print(f"Fitting {i+1}/{len(param_combinations)}: {params}")
            
            estimator = self.estimator
            
            self._set_params(estimator, params)
            
            cv_scores = []
            for train_idx, test_idx in cv.split(X, y):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train = y[train_idx] if y is not None else None
                y_test = y[test_idx] if y is not None else None
                
                if y_train is not None:
                    estimator.fit(X_train, y_train)
                else:
                    estimator.fit(X_train)
                
                score = self._score(estimator, X_test, y_test)
                cv_scores.append(score)
            
            mean_score = np.mean(cv_scores)
            std_score = np.std(cv_scores)
            
            results['params'].append(params)
            results['mean_test_score'].append(mean_score)
            results['std_test_score'].append(std_score)
            
            if mean_score > best_score:
                best_score = mean_score
                best_params = params
                best_estimator = estimator
        
        ranks = np.argsort(np.argsort(-np.array(results['mean_test_score']))) + 1
        results['rank_test_score'] = ranks.tolist()
        
        self.cv_results_ = results
        self.best_score_ = best_score
        self.best_params_ = best_params
        
        if self.refit:
            if self.verbose > 0:
                print(f"Refitting best estimator with params: {best_params}")
            self._set_params(self.estimator, best_params)
            if y is not None:
                self.estimator.fit(X, y)
            else:
                self.estimator.fit(X)
            self.best_estimator_ = self.estimator
        else:
            self.best_estimator_ = best_estimator
        
        return self
    
    def predict(self, X):
        if self.best_estimator_ is None:
            raise ValueError("This GridSearchCV instance is not fitted yet.")
        return self.best_estimator_.predict(X)
    
    def score(self, X, y):
        if self.best_estimator_ is None:
            raise ValueError("This GridSearchCV instance is not fitted yet.")
        return self._score(self.best_estimator_, X, y)

