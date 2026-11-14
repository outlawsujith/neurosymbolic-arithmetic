"""Perceive two noisy digit images, reason about them, and explain the answer.

    python demo.py              # uses model.npz, training one first if missing
    python demo.py --seed 7 --examples 3
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from nesy import MLP, TASKS, make_pairs, to_ascii
from nesy.logic import format_atom
from nesy.pipeline import perceive_and_reason, train_neurosymbolic


def load_or_train(path: str) -> MLP:
    if os.path.exists(path):
        return MLP.load(path)
    print("No saved model found; training one from sum labels (a few seconds)...\n")
    task = TASKS["sum"]
    X1, X2, d1, d2 = make_pairs(2000, np.random.default_rng(0))
    net = train_neurosymbolic(task, X1, X2, task.encode(d1, d2), epochs=60)
    net.save(path)
    return net


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="model.npz")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--examples", type=int, default=2)
    args = parser.parse_args()

    net = load_or_train(args.model)
    X1, X2, d1, d2 = make_pairs(args.examples, np.random.default_rng(args.seed))

    for i in range(args.examples):
        print("=" * 64)
        print(f"Example {i + 1}: the network sees two noisy images")
        for a, b in zip(to_ascii(X1[i]), to_ascii(X2[i])):
            print(f"    {a}     {b}")

        kb, derived, (c1, c2) = perceive_and_reason(net, X1[i], X2[i])
        left = kb.query("left(X)")[0]["X"]
        right = kb.query("right(X)")[0]["X"]
        print(f"\nNeural perception:  left = {left} ({c1:.0%} confident), "
              f"right = {right} ({c2:.0%} confident)   [truth: {d1[i]}, {d2[i]}]")

        print("\nSymbolic reasoning derived:")
        for fact in sorted(derived, key=format_atom):
            print(f"    {format_atom(fact)}")

        total = kb.query("sum(Z)")[0]["Z"]
        target = "carry" if kb.holds("carry") else "even_sum" if kb.holds("even_sum") else "odd_sum"
        print(f"\nWhy does `{target}` hold?")
        for line in kb.explain(target):
            print(f"    {line}")

        # Abduction: reason backwards from a known answer to the likeliest inputs.
        p1 = net.predict_proba(X1[i][None])[0]
        p2 = net.predict_proba(X2[i][None])[0]
        a, b, post = TASKS["sum"].abduce(p1, p2, int(d1[i] + d2[i]))
        print(f"\nAbduction: told only that the true sum is {d1[i] + d2[i]}, the most "
              f"probable digits are ({a}, {b}) with posterior {post:.0%}")
        print(f"Predicted sum: {total}   True sum: {d1[i] + d2[i]}\n")


if __name__ == "__main__":
    main()
