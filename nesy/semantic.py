"""The bridge between the two halves: differentiable probabilistic reasoning.

A `SymbolicTask` wraps an ordinary symbolic function f(a, b) over digits. The
network gives a distribution over each digit; the task computes the exact
probability of every possible output by summing over all "worlds" (a, b) that
the symbolic program maps to that output (weighted model counting):

    P(y) = sum over a, b with f(a, b) = y of  p1[a] * p2[b]

The loss -log P(y_observed) is differentiable in p1 and p2, so the label on
the *output* of the program trains the perception network on its *inputs*,
which are never labelled. This is the idea behind DeepProbLog and the
semantic loss, written out for the two-digit case.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

EPS = 1e-12


class SymbolicTask:
    def __init__(self, name: str, fn: Callable[[int, int], int], domain: int = 10):
        self.name = name
        self.fn = fn
        self.domain = domain
        table = np.array([[fn(a, b) for b in range(domain)] for a in range(domain)])
        self.outputs = sorted(set(table.ravel().tolist()))
        self._index = {y: i for i, y in enumerate(self.outputs)}
        # mask[k, a, b] is 1 when the program maps world (a, b) to output k.
        self.mask = np.stack([(table == y) for y in self.outputs]).astype(float)

    def encode(self, d1: np.ndarray, d2: np.ndarray) -> np.ndarray:
        """Output index for each pair of true digits (the training label)."""
        return np.array([self._index[self.fn(int(a), int(b))] for a, b in zip(d1, d2)])

    def output_probs(self, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
        """Distribution over program outputs, shape (n, len(outputs))."""
        return np.einsum("kab,na,nb->nk", self.mask, p1, p2)

    def loss_and_grad(self, p1: np.ndarray, p2: np.ndarray, y: np.ndarray):
        """Mean of -log P(y) and its per-example gradient w.r.t. p1 and p2."""
        consistent = self.mask[y]                       # (n, a, b)
        py = np.einsum("nab,na,nb->n", consistent, p1, p2) + EPS
        loss = float(-np.log(py).mean())
        g1 = -np.einsum("nab,nb->na", consistent, p2) / py[:, None]
        g2 = -np.einsum("nab,na->nb", consistent, p1) / py[:, None]
        return loss, g1, g2

    def predict(self, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
        """Most probable program output for each pair."""
        best = self.output_probs(p1, p2).argmax(axis=1)
        return np.array(self.outputs)[best]

    def abduce(self, p1: np.ndarray, p2: np.ndarray, y: int):
        """Abduction: the most probable digits (a, b) for ONE image pair given
        that the program output is known to be `y`. Returns (a, b, posterior)."""
        joint = self.mask[self._index[y]] * np.outer(p1, p2)
        total = joint.sum()
        if total <= 0:
            raise ValueError(f"output {y!r} is impossible under the perceived digits")
        a, b = np.unravel_index(joint.argmax(), joint.shape)
        return int(a), int(b), float(joint[a, b] / total)


def chain_to_logits(p: np.ndarray, grad_p: np.ndarray) -> np.ndarray:
    """Convert dLoss/dprobs into dLoss/dlogits through the softmax Jacobian."""
    return p * (grad_p - (p * grad_p).sum(axis=1, keepdims=True))


TASKS = {
    "sum": SymbolicTask("sum", lambda a, b: a + b),
    "product": SymbolicTask("product", lambda a, b: a * b),
    "difference": SymbolicTask("difference", lambda a, b: a - b),
    "concat": SymbolicTask("concat", lambda a, b: 10 * a + b),
}
