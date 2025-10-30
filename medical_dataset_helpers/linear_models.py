import numpy as np

class LogisticRegression:
    
    def __init__(self, max_iter=1000, learning_rate=0.1, tol=1e-4, 
                 random_state=None, verbose=0, penalty='l2', C=1.0):
        """
        Parameters:
        -----------
        max_iter : int, default=1000
            Maximum number of iterations for gradient descent.
        learning_rate : float, default=0.1
            Learning rate (step size) for gradient descent.
        tol : float, default=1e-4
            Tolerance for stopping criterion.
        random_state : int or None, default=None
            Random seed for initialization.
        verbose : int, default=0
            Verbosity level.
        penalty : {'l2', None}, default='l2'
            Specify the norm of the penalty. 'l2' for L2 regularization (Ridge).
            None for no regularization.
        C : float, default=1.0
            Inverse of regularization strength. Must be a positive float.
            Like in support vector machines, smaller values specify stronger regularization.
        """
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.tol = tol
        self.random_state = random_state
        self.verbose = verbose
        self.penalty = penalty
        self.C = C
        self.coef_ = None
        self.intercept_ = None
        self.classes_ = None
        self.n_iter_ = None
        
        if penalty not in [None, 'l2']:
            raise ValueError(f"penalty must be 'l2' or None, got {penalty}")
        if C <= 0:
            raise ValueError(f"C must be positive, got {C}")
    
    def _sigmoid(self, z):
        z = np.clip(z, -500, 500)  
        return 1 / (1 + np.exp(-z))
    
    def _compute_loss(self, predictions, y, coef):
        predictions = np.clip(predictions, 1e-15, 1 - 1e-15)
        loss = -np.mean(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))
        
        if self.penalty == 'l2':
            regularization = np.sum(coef ** 2) / (2.0 * self.C)
            loss += regularization
        
        return loss
    
    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        
        if y.ndim == 1:
            self.classes_ = np.unique(y)
            n_classes = len(self.classes_)
        else:
            raise ValueError("y must be 1D array for binary classification")
        
        if n_classes != 2:
            raise ValueError("This implementation only supports binary classification")
        
        y_binary = (y == self.classes_[1]).astype(float)
        
        n_samples, n_features = X.shape
        
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        coef = np.random.randn(n_features) * 0.01
        intercept_val = 0.0
        
        prev_loss = float('inf')
        compute_loss_every = max(1, self.max_iter // 20) 
        loss = None  
        for iteration in range(self.max_iter):
            z = np.dot(X, coef) + intercept_val
            predictions = self._sigmoid(z)
            
            if iteration % compute_loss_every == 0 or iteration == self.max_iter - 1:
                loss = self._compute_loss(predictions, y_binary, coef)
                
                if abs(prev_loss - loss) < self.tol:
                    if self.verbose > 0:
                        print(f"Converged at iteration {iteration}, loss={loss:.6f}")
                    break
                
                prev_loss = loss
            
            error = predictions - y_binary
            grad_coef = np.dot(error, X) / n_samples
            
            if self.penalty == 'l2':
                grad_coef += coef / self.C
            
            grad_intercept = np.mean(error)
            
            coef = coef - self.learning_rate * grad_coef
            intercept_val = intercept_val - self.learning_rate * grad_intercept
            
            if self.verbose > 0 and iteration % 100 == 0:
                if loss is not None:
                    current_loss = loss if iteration % compute_loss_every == 0 else prev_loss
                else:
                    current_loss = self._compute_loss(predictions, y_binary, coef)
                print(f"Iteration {iteration}, loss={current_loss:.6f}")
        
        self.coef_ = coef.reshape(1, -1)
        self.intercept_ = np.array([intercept_val])
        self.n_iter_ = iteration + 1
        
        return self
    
    def predict_proba(self, X):
        if self.coef_ is None:
            raise ValueError("This LogisticRegression instance is not fitted yet.")
        
        X = np.asarray(X)
        coef_flat = self.coef_.flatten()
        intercept_val = self.intercept_[0]
        z = np.dot(X, coef_flat) + intercept_val
        proba_positive = self._sigmoid(z)
        proba_negative = 1 - proba_positive
        
        return np.column_stack([proba_negative, proba_positive])
    
    def predict(self, X):
        if self.coef_ is None:
            raise ValueError("This LogisticRegression instance is not fitted yet.")
        
        X = np.asarray(X)
        coef_flat = self.coef_.flatten()
        intercept_val = self.intercept_[0]
        z = np.dot(X, coef_flat) + intercept_val
        predicted_indices = (z > 0).astype(int)
        return self.classes_[predicted_indices]
    
    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == y)


