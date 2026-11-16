# Neurosymbolic Arithmetic

A small, dependency-light neurosymbolic AI system written from scratch in NumPy.
A neural network learns to **read digits** from noisy images while being told only
the **result of a symbolic program** run on those digits (for example, their sum).
No digit is ever labelled. A logic reasoner then draws conclusions from what the
network perceives and explains each one with a proof.

```
 image 1 ──► neural net ──► P(digit 1) ─┐
                                        ├─► symbolic program ──► P(sum) ──► loss vs. observed sum
 image 2 ──► neural net ──► P(digit 2) ─┘        (exact probabilistic inference,
              (shared)                             gradients flow back through it)
```

## Why this is neurosymbolic

| Part | File | Role |
|---|---|---|
| Neural perception | [nesy/perception.py](nesy/perception.py) | MLP with hand-written backprop and Adam; maps an image to a distribution over digits |
| Differentiable reasoning | [nesy/semantic.py](nesy/semantic.py) | Sums the probability of every world `(a, b)` that the symbolic program maps to the observed answer; `-log P(answer)` is the training loss |
| Symbolic reasoning | [nesy/logic.py](nesy/logic.py) | Datalog-style forward chainer with unification, arithmetic built-ins and proof traces |
| Pipeline | [nesy/pipeline.py](nesy/pipeline.py) | Training loops, a pure-neural baseline, and perceive-then-reason |
| Data | [nesy/data.py](nesy/data.py) | Synthetic noisy, randomly shifted digit glyphs (nothing to download) |

The training signal is the approach used by DeepProbLog and the semantic loss,
written out explicitly for programs over two digits:

```
P(y) = Σ  p1[a] · p2[b]      over all digits a, b with f(a, b) = y
loss = −log P(y_observed)
```

## Quick start

Requires Python 3.9+ and NumPy.

```bash
pip install git+https://github.com/outlawsujith/neurosymbolic-arithmetic.git
nesy-train                 # learn digits from sums, compare with a neural baseline
nesy-demo                  # perceive, reason, explain
```

To work on the code instead, clone the repo and install it in editable mode:

```bash
git clone https://github.com/outlawsujith/neurosymbolic-arithmetic.git
cd neurosymbolic-arithmetic
pip install -e .
python -m unittest discover -s tests -v
```

If the `nesy-train` command is not on your PATH (or is blocked on your machine), use
`python -m nesy.train` and `python -m nesy.demo` instead.

Other tasks: `nesy-train --task product` (also `difference`, `concat`).
Add your own in `TASKS` in [nesy/semantic.py](nesy/semantic.py) as any Python function of two digits.

## Results

`nesy-train --sweep` (seed 0, 2000 unseen test pairs, both models given the
same data and the same number of gradient steps):

| Training pairs | Neurosymbolic (sum accuracy) | Neural baseline (sum accuracy) | Digit accuracy (neurosymbolic) |
|---:|---:|---:|---:|
| 100  | 8.9%  | 8.3%  | 14.9% |
| 250  | 17.8% | 8.6%  | 35.0% |
| 500  | 23.1% | 9.7%  | 38.5% |
| 1000 | 95.3% | 10.8% | 97.6% |
| 2000 | 97.2% | 14.4% | 98.6% |
| 4000 | 97.7% | 26.9% | 98.9% |

The baseline is an MLP that takes both images and predicts the sum directly, so
it has to rediscover addition from examples. The neurosymbolic model is given
addition as knowledge and only has to learn perception. On the `product` task
with 2000 pairs the gap is 97.1% vs 37.9%.

Limits worth knowing: with fewer than about 1000 pairs the answer-only signal is
too weak and the neurosymbolic model also fails; these are single-seed numbers on
synthetic data; and exact inference enumerates all 100 digit pairs, which does not
scale to long programs without approximation.

## Demo output

```
Neural perception:  left = 4 (100% confident), right = 5 (100% confident)   [truth: 4, 5]

Symbolic reasoning derived:
    odd_sum
    product(20)
    right_greater
    sum(9)

Why does `odd_sum` hold?
    odd_sum   [by rule: odd_sum :- sum(Z), mod(Z, 2, 1).]
      sum(9)   [by rule: sum(Z) :- left(X), right(Y), add(X, Y, Z).]
        left(4)   [perceived]
        right(5)   [perceived]
        add(4, 5, 9)   [arithmetic]
      mod(9, 2, 1)   [arithmetic]

Abduction: told only that the true sum is 9, the most probable digits are (4, 5) with posterior 100%
```

## Using the reasoner on its own

```python
from nesy import KnowledgeBase

kb = KnowledgeBase([
    "path(X, Y) :- edge(X, Y).",
    "path(X, Z) :- edge(X, Y), path(Y, Z).",
])
for fact in ("edge(a, b)", "edge(b, c)"):
    kb.add_fact(fact)
kb.forward_chain()
print(kb.holds("path(a, c)"))            # True
print("\n".join(kb.explain("path(a, c)")))
```

## References

- Manhaeve et al., *DeepProbLog: Neural Probabilistic Logic Programming*, NeurIPS 2018
- Xu et al., *A Semantic Loss Function for Deep Learning with Symbolic Knowledge*, ICML 2018
- Garcez & Lamb, *Neurosymbolic AI: The 3rd Wave*, 2020

## License

MIT. See [LICENSE](LICENSE).

## Related project in this repository

[`energy-optimizer/`](energy-optimizer/) applies the same neurosymbolic ideas to
industrial energy optimisation: neural load forecasting, rule-based reasoning and
explainable scheduling, with an API, dashboard and Dockerfile. It is self-contained
and can be moved to its own repository as is.
