"""Train the neurosymbolic model and compare it with a pure neural baseline.

    nesy-train                           # default: learn digits from sums
    nesy-train --task product --n-train 1000
    nesy-train --sweep                   # data-efficiency comparison table
"""

from __future__ import annotations

import argparse

import numpy as np

from nesy import TASKS, make_pairs
from nesy.pipeline import (evaluate_baseline, evaluate_neurosymbolic,
                           train_neural_baseline, train_neurosymbolic)


def run(task_name: str, n_train: int, n_test: int, epochs: int, seed: int,
        verbose: bool = True):
    task = TASKS[task_name]
    if epochs is None:  # same step budget for every training-set size
        epochs = max(30, 6000 * 32 // n_train)
    rng = np.random.default_rng(seed)
    X1, X2, d1, d2 = make_pairs(n_train, rng)
    T1, T2, t1, t2 = make_pairs(n_test, rng)
    y = task.encode(d1, d2)  # the ONLY supervision: the program's output

    def log(name):
        def _log(epoch, loss):
            if verbose and (epoch % 5 == 0 or epoch == 1):
                print(f"  [{name}] epoch {epoch:3d}  loss {loss:.4f}")
        return _log

    nesy_net = train_neurosymbolic(task, X1, X2, y, epochs=epochs, seed=seed,
                                   log=log("neurosymbolic"))
    base_net = train_neural_baseline(task, X1, X2, y, epochs=epochs, seed=seed,
                                     log=log("neural baseline"))
    nesy = evaluate_neurosymbolic(nesy_net, task, T1, T2, t1, t2)
    base = evaluate_baseline(base_net, task, T1, T2, t1, t2)
    return nesy_net, nesy, base


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", choices=sorted(TASKS), default="sum")
    parser.add_argument("--n-train", type=int, default=2000)
    parser.add_argument("--n-test", type=int, default=2000)
    parser.add_argument("--epochs", type=int, default=None,
                        help="default: enough epochs for about 6000 gradient steps")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save", default="model.npz", help="where to save the digit classifier")
    parser.add_argument("--sweep", action="store_true",
                        help="compare both models across training-set sizes")
    args = parser.parse_args()

    if args.sweep:
        print(f"task: {args.task}   (test accuracy on {args.n_test} unseen pairs)\n")
        print(f"{'train pairs':>11} | {'neurosymbolic':>13} | {'neural baseline':>15} | {'digit acc (NeSy)':>16}")
        print("-" * 66)
        for n in (100, 250, 500, 1000, 2000, 4000):
            _, nesy, base = run(args.task, n, args.n_test, args.epochs, args.seed, verbose=False)
            print(f"{n:>11} | {nesy['task_accuracy']:>12.1%} | {base['task_accuracy']:>14.1%} "
                  f"| {nesy['digit_accuracy']:>15.1%}")
        return

    print(f"Training on {args.n_train} image pairs labelled only with their {args.task}.\n")
    net, nesy, base = run(args.task, args.n_train, args.n_test, args.epochs, args.seed)
    print(f"\nTest results on {args.n_test} unseen pairs ({args.task}):")
    print(f"  neurosymbolic   task accuracy {nesy['task_accuracy']:.1%}   "
          f"digit accuracy {nesy['digit_accuracy']:.1%}  (digits were never labelled)")
    print(f"  neural baseline task accuracy {base['task_accuracy']:.1%}")
    net.save(args.save)
    print(f"\nSaved digit classifier to {args.save}")


if __name__ == "__main__":
    main()
