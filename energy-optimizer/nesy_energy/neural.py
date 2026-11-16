"""Neural half: an MLP load forecaster trained with a knowledge-informed loss.

Besides fitting the data, the loss penalises predictions that break two symbolic
facts about the plant: load is never negative, and load never falls when the
production rate rises (all else equal). NumPy only, so it deploys anywhere.
"""
from __future__ import annotations

import numpy as np


class MLP:
    def __init__(self, n_in: int, hidden=(32, 32), seed: int = 0):
        rng = np.random.default_rng(seed)
        sizes = [n_in, *hidden, 1]
        self.W = [rng.normal(0, np.sqrt(1 / a), (a, b)) for a, b in zip(sizes, sizes[1:])]
        self.b = [np.zeros(b) for b in sizes[1:]]

    def forward(self, X):
        acts = [X]
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = acts[-1] @ W + b
            acts.append(z if i == len(self.W) - 1 else np.tanh(z))
        return acts

    def backward(self, acts, dy):
        gW, gb, delta = [None] * len(self.W), [None] * len(self.W), dy
        for i in reversed(range(len(self.W))):
            gW[i], gb[i] = acts[i].T @ delta, delta.sum(0)
            if i:
                delta = (delta @ self.W[i].T) * (1 - acts[i] ** 2)
        return gW + gb


class Forecaster:
    def __init__(self, seed: int = 0):
        self.seed = seed
        self.net: MLP | None = None

    def fit(self, X, y, monotone=(), knowledge_weight: float = 5.0, epochs: int = 200,
            lr: float = 3e-3, batch: int = 64) -> "Forecaster":
        """`monotone`: feature indices the output must be non-decreasing in."""
        rng = np.random.default_rng(self.seed)
        self.xm, self.xs = X.mean(0), X.std(0) + 1e-9
        self.ym, self.ys = y.mean(), y.std() + 1e-9
        Xn, yn = (X - self.xm) / self.xs, ((y - self.ym) / self.ys)[:, None]
        self.net = net = MLP(X.shape[1], seed=self.seed)
        params = net.W + net.b
        m, v, step = [0 * p for p in params], [0 * p for p in params], 0
        for _ in range(epochs):
            order = rng.permutation(len(Xn))
            for k in range(0, len(Xn), batch):
                xb, yb = Xn[order[k:k + batch]], yn[order[k:k + batch]]
                n = len(xb)
                acts = net.forward(xb)
                out = acts[-1]
                dy = 2 * (out - yb) / n                                   # data fit
                if knowledge_weight:
                    neg = np.maximum(0, -(out + self.ym / self.ys))        # load >= 0
                    dy = dy - knowledge_weight * 2 * neg / n
                grads = None
                for j in (monotone if knowledge_weight else ()):
                    x2 = xb.copy()
                    x2[:, j] += rng.uniform(0.05, 0.4, n) / self.xs[j]
                    acts2 = net.forward(x2)
                    viol = np.maximum(0, out - acts2[-1])                  # f(x) <= f(x + d)
                    g = knowledge_weight * 2 * viol / n
                    dy = dy + g
                    g2 = net.backward(acts2, -g)
                    grads = g2 if grads is None else [a + b for a, b in zip(grads, g2)]
                g1 = net.backward(acts, dy)
                grads = g1 if grads is None else [a + b for a, b in zip(grads, g1)]
                step += 1
                for i, (p, g) in enumerate(zip(params, grads)):           # Adam
                    m[i] = 0.9 * m[i] + 0.1 * g
                    v[i] = 0.999 * v[i] + 0.001 * g * g
                    p -= lr * (m[i] / (1 - 0.9 ** step)) / (np.sqrt(v[i] / (1 - 0.999 ** step)) + 1e-8)
        return self

    def predict(self, X) -> np.ndarray:
        out = self.net.forward((X - self.xm) / self.xs)[-1][:, 0]
        return out * self.ys + self.ym

    def mae(self, X, y) -> float:
        return float(np.abs(self.predict(X) - y).mean())

    def monotone_violation_rate(self, X, j: int, delta: float = 0.2, tol: float = 0.01) -> float:
        """Share of probe points where raising feature j by `delta` lowers the forecast."""
        X2 = X.copy()
        X2[:, j] += delta
        return float((self.predict(X2) < self.predict(X) - tol).mean())
