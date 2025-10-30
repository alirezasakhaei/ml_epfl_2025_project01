
import numpy as np

class MinMaxScaler:
    """
    Min-Max Scaler that scales features to a range [0, 1].
    """
    
    def __init__(self, feature_range=(0, 1)):
        self.feature_range = feature_range
        self.data_min_ = None
        self.data_max_ = None
        self.scale_ = None
        self.min_ = None
    
    def fit(self, X, y=None):
        X = np.asarray(X)
        self.data_min_ = np.min(X, axis=0)
        self.data_max_ = np.max(X, axis=0)
        
        data_range = self.data_max_ - self.data_min_
        data_range[data_range == 0] = 1.0
        
        feature_range_min, feature_range_max = self.feature_range
        scale = (feature_range_max - feature_range_min) / data_range
        self.scale_ = scale
        self.min_ = feature_range_min - self.data_min_ * scale
        
        return self
    
    def transform(self, X):
        if self.scale_ is None:
            raise ValueError("This MinMaxScaler instance is not fitted yet.")
        
        X = np.asarray(X)
        X_scaled = X * self.scale_ + self.min_
        return X_scaled
    
    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


class StandardScaler:
    """
    Standard Scaler that standardizes features by removing the mean and 
    scaling to unit variance (z-score normalization).
    """
    
    def __init__(self):
        self.mean_ = None
        self.scale_ = None
        self.var_ = None
    
    def fit(self, X, y=None):
        X = np.asarray(X)
        
        self.mean_ = np.mean(X, axis=0)
        self.var_ = np.var(X, axis=0, ddof=0)  
        
        self.scale_ = np.sqrt(self.var_)
        self.scale_[self.scale_ == 0] = 1.0
        
        return self
    
    def transform(self, X):
        if self.scale_ is None:
            raise ValueError("This StandardScaler instance is not fitted yet.")
        
        X = np.asarray(X)
        X_scaled = (X - self.mean_) / self.scale_
        
        return X_scaled
    
    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)
    
    def inverse_transform(self, X):
        if self.scale_ is None:
            raise ValueError("This StandardScaler instance is not fitted yet.")
        
        X = np.asarray(X)
        X_original = X * self.scale_ + self.mean_
        
        return X_original
