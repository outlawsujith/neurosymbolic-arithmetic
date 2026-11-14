"""The neural half: a small multilayer perceptron with manual backprop and Adam."""

from __future__ import annotations

import numpy as np


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class MLP:
    """Fully connected ReLU network that outputs logits."""

    def __init__(self, sizes: list[int], seed: int = 0):
        rng = np.random.default_rng(seed)
        self.sizes = list(sizes)
        self.W = [rng.normal(0.0, np.sqrt(2.0 / m), (m, n))
                  for m, n in zip(sizes[:-1], sizes[1:])]
        self.b = [np.zeros(n) for n in sizes[1:]]
        self._m = [np.zeros_like(p) for p in self.params]
        self._v = [np.zeros_like(p) for p in self.params]
        self._t = 0
        self._acts: list[np.ndarray] = []

    @property
    def params(self) -> list[np.ndarray]:
        return [p for pair in zip(self.W, self.b) for p in pair]

    def forward(self, X: np.ndarray) -> np.ndarray:
        self._acts = [X]
        h = X
        last = len(self.W) - 1
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            h = h @ W + b
            if i < last:
                h = np.maximum(h, 0.0)
            self._acts.append(h)
        return h

    def backward(self, dlogits: np.ndarray) -> list[np.ndarray]:
        """Gradients of the loss w.r.t. every parameter, given dLoss/dlogits
        for the batch most recently passed through `forward`."""
        grads: list[np.ndarray] = []
        dz = dlogits
        for i in reversed(range(len(self.W))):
            a = self._acts[i]
            grads = [a.T @ dz, dz.sum(axis=0)] + grads
            if i > 0:
                dz = (dz @ self.W[i].T) * (a > 0)
        return grads

    def step(self, grads: list[np.ndarray], lr: float = 1e-3,
             beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8) -> None:
        """One Adam update."""
        self._t += 1
        for p, g, m, v in zip(self.params, grads, self._m, self._v):
            m *= beta1
            m += (1 - beta1) * g
            v *= beta2
            v += (1 - beta2) * g * g
            m_hat = m / (1 - beta1 ** self._t)
            v_hat = v / (1 - beta2 ** self._t)
            p -= lr * m_hat / (np.sqrt(v_hat) + eps)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return softmax(self.forward(X))

    def save(self, path: str) -> None:
        arrays = {f"W{i}": W for i, W in enumerate(self.W)}
        arrays.update({f"b{i}": b for i, b in enumerate(self.b)})
        np.savez(path, sizes=np.array(self.sizes), **arrays)

    @classmethod
    def load(cls, path: str) -> "MLP":
        data = np.load(path)
        net = cls([int(s) for s in data["sizes"]])
        net.W = [data[f"W{i}"] for i in range(len(net.W))]
        net.b = [data[f"b{i}"] for i in range(len(net.b))]
        return net
