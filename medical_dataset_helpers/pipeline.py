import numpy as np

class Pipeline:
    """
    Pipeline of transforms with a final estimator.
    """
    
    def __init__(self, steps):
        self.steps = steps
        self.named_steps = {name: step for name, step in steps}
    
    def fit(self, X, y=None):
        X_transformed = X
        
        for _, transformer in self.steps[:-1]:
            transformer.fit(X_transformed)
            X_transformed = transformer.transform(X_transformed)
        
        _, final_estimator = self.steps[-1]
        if y is not None:
            final_estimator.fit(X_transformed, y)
        else:
            final_estimator.fit(X_transformed)
        
        return self
    
    def predict(self, X):
        X_transformed = X
        
        for ـ, transformer in self.steps[:-1]:
            X_transformed = transformer.transform(X_transformed)
        _, final_estimator = self.steps[-1]
        return final_estimator.predict(X_transformed)
    
    def predict_proba(self, X):
        X_transformed = X
        
        for _, transformer in self.steps[:-1]:
            X_transformed = transformer.transform(X_transformed)
        
        final_name, final_estimator = self.steps[-1]
        if hasattr(final_estimator, 'predict_proba'):
            return final_estimator.predict_proba(X_transformed)
        else:
            raise AttributeError(f"{final_name} does not have predict_proba method")
    
    def transform(self, X):
        X_transformed = X
        
        for _, transformer in self.steps:
            X_transformed = transformer.transform(X_transformed)
        
        return X_transformed
    
    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)
    
    def score(self, X, y):
        X_transformed = X
        
        for _, transformer in self.steps[:-1]:
            X_transformed = transformer.transform(X_transformed)
        
        _, final_estimator = self.steps[-1]
        if hasattr(final_estimator, 'score'):
            return final_estimator.score(X_transformed, y)
        else:
            y_pred = final_estimator.predict(X_transformed)
            return np.mean(y_pred == y)

