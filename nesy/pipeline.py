"""Training loops and the perceive-then-reason pipeline."""

from __future__ import annotations

import numpy as np

from .data import IMAGE_DIM
from .logic import KnowledgeBase
from .perception import MLP, softmax
from .semantic import SymbolicTask, chain_to_logits

# Background knowledge applied on top of whatever the network perceives.
ARITHMETIC_RULES = [
    "sum(Z) :- left(X), right(Y), add(X, Y, Z).",
    "product(Z) :- left(X), right(Y), mul(X, Y, Z).",
    "even_sum :- sum(Z), mod(Z, 2, 0).",
    "odd_sum :- sum(Z), mod(Z, 2, 1).",
    "carry :- sum(Z), ge(Z, 10).",
    "left_greater :- left(X), right(Y), gt(X, Y).",
    "right_greater :- left(X), right(Y), lt(X, Y).",
    "same_digit :- left(X), right(X).",
    "zero_product :- product(0).",
]


def train_neurosymbolic(task: SymbolicTask, X1, X2, y, *, hidden: int = 128,
                        epochs: int = 30, batch: int = 32, lr: float = 3e-3,
                        seed: int = 0, log=None) -> MLP:
    """Train a single-digit classifier using only labels on the program output.

    One network is shared by both images. Its two predicted distributions are
    pushed through the symbolic task, and the gradient of -log P(y) flows back
    through the reasoning step into the network.
    """
    net = MLP([IMAGE_DIM, hidden, task.domain], seed=seed)
    rng = np.random.default_rng(seed)
    n = len(y)
    for epoch in range(epochs):
        order = rng.permutation(n)
        total = 0.0
        for start in range(0, n, batch):
            idx = order[start:start + batch]
            k = len(idx)
            probs = softmax(net.forward(np.concatenate([X1[idx], X2[idx]])))
            p1, p2 = probs[:k], probs[k:]
            loss, g1, g2 = task.loss_and_grad(p1, p2, y[idx])
            dlogits = np.concatenate([chain_to_logits(p1, g1), chain_to_logits(p2, g2)]) / k
            net.step(net.backward(dlogits), lr=lr)
            total += loss * k
        if log:
            log(epoch + 1, total / n)
    return net


def train_neural_baseline(task: SymbolicTask, X1, X2, y, *, hidden: int = 128,
                          epochs: int = 30, batch: int = 32, lr: float = 3e-3,
                          seed: int = 0, log=None) -> MLP:
    """Pure neural baseline: both images in, program output out, no symbols.

    It has to discover arithmetic itself from input/output examples.
    """
    X = np.concatenate([X1, X2], axis=1)
    net = MLP([2 * IMAGE_DIM, hidden, hidden, len(task.outputs)], seed=seed)
    rng = np.random.default_rng(seed)
    n = len(y)
    for epoch in range(epochs):
        order = rng.permutation(n)
        total = 0.0
        for start in range(0, n, batch):
            idx = order[start:start + batch]
            k = len(idx)
            probs = softmax(net.forward(X[idx]))
            total += -np.log(probs[np.arange(k), y[idx]] + 1e-12).sum()
            dlogits = probs.copy()
            dlogits[np.arange(k), y[idx]] -= 1.0
            net.step(net.backward(dlogits / k), lr=lr)
        if log:
            log(epoch + 1, total / n)
    return net


def evaluate_neurosymbolic(net: MLP, task: SymbolicTask, X1, X2, d1, d2) -> dict:
    p1, p2 = net.predict_proba(X1), net.predict_proba(X2)
    truth = np.array([task.fn(int(a), int(b)) for a, b in zip(d1, d2)])
    digit_acc = (np.concatenate([p1.argmax(1), p2.argmax(1)])
                 == np.concatenate([d1, d2])).mean()
    return {"task_accuracy": float((task.predict(p1, p2) == truth).mean()),
            "digit_accuracy": float(digit_acc)}


def evaluate_baseline(net: MLP, task: SymbolicTask, X1, X2, d1, d2) -> dict:
    pred = net.forward(np.concatenate([X1, X2], axis=1)).argmax(1)
    return {"task_accuracy": float((pred == task.encode(d1, d2)).mean())}


def perceive_and_reason(net: MLP, image1, image2, rules=ARITHMETIC_RULES):
    """Neural perception turns two images into symbols; the reasoner derives
    everything that follows from them. Returns (kb, derived facts, confidences)."""
    p1 = net.predict_proba(image1[None])[0]
    p2 = net.predict_proba(image2[None])[0]
    left, right = int(p1.argmax()), int(p2.argmax())
    kb = KnowledgeBase(rules)
    kb.add_fact(("left", left))
    kb.add_fact(("right", right))
    derived = kb.forward_chain()
    return kb, derived, (float(p1[left]), float(p2[right]))
