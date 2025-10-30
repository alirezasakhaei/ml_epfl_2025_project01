import numpy as np


def _activation(name):
    if name == 'identity':
        return (
            lambda z: z,
            lambda z: np.ones_like(z),
        )
    if name == 'logistic':
        return (
            lambda z: 1.0 / (1.0 + np.exp(-z)),
            lambda z: (
                lambda a: a * (1.0 - a)
            )((1.0 / (1.0 + np.exp(-z))))
        )
    if name == 'tanh':
        return (
            lambda z: np.tanh(z),
            lambda z: 1.0 - np.tanh(z) ** 2,
        )
    if name == 'relu':
        return (
            lambda z: np.maximum(0.0, z),
            lambda z: (z > 0.0).astype(z.dtype),
        )
    raise ValueError(f"Unsupported activation: {name}")


def _softmax(z):
    z_shift = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z_shift)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


class MLPClassifier:
    def __init__(
        self,
        hidden_layer_sizes=(100,),
        activation='relu',
        solver='adam',
        alpha=1e-4,
        batch_size='auto',
        learning_rate='constant',
        learning_rate_init=1e-3,
        power_t=0.5,
        max_iter=200,
        shuffle=True,
        random_state=None,
        tol=1e-4,
        verbose=False,
        warm_start=False,
        momentum=0.9,
        nesterovs_momentum=True,
        early_stopping=False,
        validation_fraction=0.1,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-8,
        n_iter_no_change=10,
        max_fun=15000,
    ):
        self.hidden_layer_sizes = tuple(hidden_layer_sizes) if isinstance(hidden_layer_sizes, (list, tuple)) else (hidden_layer_sizes,)
        self.activation = activation
        self.solver = solver
        self.alpha = float(alpha)
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.learning_rate_init = float(learning_rate_init)
        self.power_t = float(power_t)
        self.max_iter = int(max_iter)
        self.shuffle = bool(shuffle)
        self.random_state = None if random_state is None else int(random_state)
        self.tol = float(tol)
        self.verbose = bool(verbose)
        self.warm_start = bool(warm_start)
        self.momentum = float(momentum)
        self.nesterovs_momentum = bool(nesterovs_momentum)
        self.early_stopping = bool(early_stopping)
        self.validation_fraction = float(validation_fraction)
        self.beta_1 = float(beta_1)
        self.beta_2 = float(beta_2)
        self.epsilon = float(epsilon)
        self.n_iter_no_change = int(n_iter_no_change)
        self.max_fun = int(max_fun)

        self.coefs_ = None
        self.intercepts_ = None
        self.n_layers_ = None
        self.n_outputs_ = None
        self.n_features_in_ = None
        self.loss_curve_ = []
        self.classes_ = None

        self._opt_m = None
        self._opt_v = None
        self._opt_t = 0

    def _init_params(self, n_features, n_classes):
        layer_sizes = (n_features,) + self.hidden_layer_sizes + (n_classes,)
        rng = np.random.RandomState(self.random_state)
        self.coefs_ = []
        self.intercepts_ = []
        for fan_in, fan_out in zip(layer_sizes[:-1], layer_sizes[1:]):
            if self.activation == 'relu':
                scale = np.sqrt(2.0 / fan_in)
            else:
                scale = np.sqrt(1.0 / fan_in)
            W = rng.normal(0.0, scale, size=(fan_in, fan_out))
            b = np.zeros((1, fan_out))
            self.coefs_.append(W)
            self.intercepts_.append(b)

        self.n_layers_ = len(layer_sizes)
        self.n_outputs_ = n_classes
        self.n_features_in_ = n_features
        self._opt_m = [np.zeros_like(W) for W in self.coefs_] + [np.zeros_like(b) for b in self.intercepts_]
        self._opt_v = [np.zeros_like(W) for W in self.coefs_] + [np.zeros_like(b) for b in self.intercepts_]
        self._opt_t = 0

    def _forward(self, X):
        act_fn, _ = _activation(self.activation)
        a = X
        activations = [a]
        zs = []
        for i in range(len(self.coefs_) - 1):
            z = a @ self.coefs_[i] + self.intercepts_[i]
            a = act_fn(z)
            zs.append(z)
            activations.append(a)
        zL = activations[-1] @ self.coefs_[-1] + self.intercepts_[-1]
        zs.append(zL)
        if self.n_outputs_ == 1:
            aL = 1.0 / (1.0 + np.exp(-zL))
        else:
            aL = _softmax(zL)
        activations.append(aL)
        return activations, zs

    def _loss(self, y_true_onehot, activations):
        aL = activations[-1]
        if self.n_outputs_ == 1:
            y = y_true_onehot
            eps = 1e-12
            loss = -np.mean(y * np.log(aL + eps) + (1.0 - y) * np.log(1.0 - aL + eps))
        else:
            eps = 1e-12
            loss = -np.mean(np.sum(y_true_onehot * np.log(aL + eps), axis=1))
        reg = 0.5 * self.alpha * sum(np.sum(W * W) for W in self.coefs_)
        return loss + reg

    def _backprop(self, X, y_onehot, activations, zs):
        _, act_prime = _activation(self.activation)
        grads_W = [None] * len(self.coefs_)
        grads_b = [None] * len(self.intercepts_)

        aL = activations[-1]
        m = X.shape[0]
        if self.n_outputs_ == 1:
            dZ = (aL - y_onehot) 
        else:
            dZ = (aL - y_onehot)  

        grads_W[-1] = (activations[-2].T @ dZ) / m + self.alpha * self.coefs_[-1]
        grads_b[-1] = np.sum(dZ, axis=0, keepdims=True) / m

        for l in range(len(self.coefs_) - 2, -1, -1):
            dA = dZ @ self.coefs_[l + 1].T
            dZ = dA * act_prime(zs[l])
            grads_W[l] = (activations[l].T @ dZ) / m + self.alpha * self.coefs_[l]
            grads_b[l] = np.sum(dZ, axis=0, keepdims=True) / m

        return grads_W, grads_b

    def _update_adam(self, grads_W, grads_b, lr):
        self._opt_t += 1
        t = self._opt_t
        beta1, beta2, eps = self.beta_1, self.beta_2, self.epsilon

        params = self.coefs_ + self.intercepts_
        grads = grads_W + grads_b
        for i, (p, g) in enumerate(zip(params, grads)):
            self._opt_m[i] = beta1 * self._opt_m[i] + (1.0 - beta1) * g
            self._opt_v[i] = beta2 * self._opt_v[i] + (1.0 - beta2) * (g * g)
            m_hat = self._opt_m[i] / (1.0 - beta1 ** t)
            v_hat = self._opt_v[i] / (1.0 - beta2 ** t)
            p -= lr * m_hat / (np.sqrt(v_hat) + eps)

    def _update_sgd(self, grads_W, grads_b, lr, velocity):
        params = self.coefs_ + self.intercepts_
        grads = grads_W + grads_b
        for i, (p, g) in enumerate(zip(params, grads)):
            velocity[i] = self.momentum * velocity[i] - lr * g
            if self.nesterovs_momentum:
                p += self.momentum * velocity[i] - lr * g
            else:
                p += velocity[i]

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        if y.ndim != 1:
            raise ValueError("y must be 1D labels")

        self.classes_ = np.unique(y)
        class_to_index = {c: i for i, c in enumerate(self.classes_)}
        y_idx = np.vectorize(class_to_index.get)(y)
        n_classes = len(self.classes_)
        y_onehot = None
        if n_classes == 2:
            y_onehot = y_idx.reshape(-1, 1)
        else:
            y_onehot = np.eye(n_classes, dtype=float)[y_idx]

        n_samples, n_features = X.shape
        if not self.warm_start or self.coefs_ is None:
            self._init_params(n_features, 1 if n_classes == 2 else n_classes)

        rng = np.random.RandomState(self.random_state)
        batch_size = n_samples if self.solver == 'lbfgs' else (min(200, n_samples) if self.batch_size == 'auto' else int(self.batch_size))

        if self.early_stopping:
            split = int((1.0 - self.validation_fraction) * n_samples)
            indices = np.arange(n_samples)
            if self.shuffle:
                rng.shuffle(indices)
            train_idx, val_idx = indices[:split], indices[split:]
            X_train, y_train_oh = X[train_idx], y_onehot[train_idx]
            X_val, y_val_oh = X[val_idx], y_onehot[val_idx]
        else:
            X_train, y_train_oh = X, y_onehot
            X_val, y_val_oh = None, None

        best_val_loss = np.inf
        no_improve = 0
        velocity = [np.zeros_like(W) for W in (self.coefs_ + self.intercepts_)] if self.solver == 'sgd' else None

        for epoch in range(self.max_iter):
            if self.solver == 'sgd':
                if self.learning_rate == 'constant':
                    lr = self.learning_rate_init
                elif self.learning_rate == 'invscaling':
                    lr = self.learning_rate_init / ((epoch + 1) ** self.power_t)
                elif self.learning_rate == 'adaptive':
                    lr = self.learning_rate_init  # will decay on plateau
                else:
                    raise ValueError(f"Unsupported learning_rate: {self.learning_rate}")
            elif self.solver == 'adam':
                lr = self.learning_rate_init
            else:
                raise ValueError(f"Unsupported solver: {self.solver}")

            indices = np.arange(X_train.shape[0])
            if self.shuffle:
                rng.shuffle(indices)
            for start in range(0, X_train.shape[0], batch_size):
                end = start + batch_size
                batch_idx = indices[start:end]
                Xb = X_train[batch_idx]
                yb = y_train_oh[batch_idx]
                acts, zs = self._forward(Xb)
                gW, gB = self._backprop(Xb, yb, acts, zs)
                if self.solver == 'adam':
                    self._update_adam(gW, gB, lr)
                else:
                    self._update_sgd(gW, gB, lr, velocity)

            train_loss = self._loss(y_train_oh, self._forward(X_train)[0])
            self.loss_curve_.append(train_loss)

            if self.early_stopping and X_val is not None:
                val_loss = self._loss(y_val_oh, self._forward(X_val)[0])
                improved = (best_val_loss - val_loss) > self.tol
                if improved:
                    best_val_loss = val_loss
                    no_improve = 0
                else:
                    no_improve += 1
                    if self.learning_rate == 'adaptive' and self.solver == 'sgd' and no_improve >= 2:
                        self.learning_rate_init /= 5.0
                        no_improve = 0
                if no_improve >= self.n_iter_no_change:
                    break
            else:
                if len(self.loss_curve_) > 1 and abs(self.loss_curve_[-2] - self.loss_curve_[-1]) < self.tol:
                    break

        return self

    def predict_proba(self, X):
        X = np.asarray(X)
        a, _ = self._forward(X)
        out = a[-1]
        if self.n_outputs_ == 1:
            proba_pos = out.reshape(-1)
            proba = np.vstack([1.0 - proba_pos, proba_pos]).T
            return proba
        return out

    def predict(self, X):
        proba = self.predict_proba(X)
        if self.n_outputs_ == 1:
            y_idx = (proba[:, 1] >= 0.5).astype(int)
        else:
            y_idx = np.argmax(proba, axis=1)
        return self.classes_[y_idx]


